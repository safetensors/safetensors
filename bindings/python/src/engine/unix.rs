//! CUDA prefetch engine: load byte spans of a safetensors data section into
//! device memory the caller allocated, in the background, and hand each span's
//! owner back, once, as its bytes land.
//!
//! Two units, computed once from the spans the consumer requests, sorted by
//! start and disjoint:
//!
//! ```text
//! tensors  |t0 |  t1  |     t2     |  t3  |       t4        | t5 | t6  |
//! spans    |S0#|..|S1#|S2#|...............|#######S3########|#S4#|#S5##|
//! chunks   |c0 |  |  c1   |               |   c2   |   c3   |   c4   |c|
//! ```
//!
//! - spans [`LoadPlan::spans`]: a byte interval of the data section, copied to
//!   its destination, or, with a [`Gather`], packed into it by the gather's
//!   strided copies. The delivery unit: `take` hands out the span's owner.
//! - chunks ([`LoadPlan::chunks`]): at most [`CHUNK_SIZE`] of adjacent spans,
//!   never across a gap. The I/O unit: one `pread`, one slab, one event, one
//!   `memcpy` per run of spans whose destinations are adjacent. `span_chunks`
//!   and `chunk_spans` map one to the other.
//!
//! Data flow, per file:
//!
//! ```text
//! caller (one at a time)
//!   Feed::provide(dsts, stream)          the next spans' destinations, in plan
//!                                        order, and an event on `stream`
//! worker threads
//!   c = next_chunk.fetch_add(1)          next entry of LoadPlan::chunks
//!   wait for c's spans to be provided,   condvar, then the events of the
//!     then for their batches' events     batches that provided them
//!   lease = SlabPool::acquire()          a pinned host slab from the shared
//!                                        pool; waits for its previous copy
//!   pread(chunk c) -> lease              one read per chunk
//!   memcpyAsync(lease -> destinations)   one copy per run of adjacent
//!                                        destinations, on STREAMS[device];
//!                                        memcpy2DAsync for a gather's rows
//!   chunk_completion_events[c] = event   recorded after the copies; 0 while
//!                                        the chunk is still pending
//! consumer
//!   take_tensor(s) / TensorIter::next    wait the events of s's chunks, then
//!     -> owner of s's destination        handed out once (AlreadyDelivered after)
//! ```
//!
//! [`Loader`] owns the worker threads; [`LoaderInner`] (source, plan, sink, chunk
//! counter, first error) is shared with them and with [`TensorIter`]. Closing
//! signals the workers, joins them, waits for the copies in flight, then
//! returns the owners not taken: an outstanding iterator yields `Err(Closed)`
//! once and is exhausted.

use std::{
    collections::VecDeque,
    fmt::Display,
    fs::File,
    num::NonZeroUsize,
    ops::Range,
    os::unix::fs::FileExt,
    sync::{
        atomic::{AtomicBool, AtomicU64, AtomicUsize, Ordering},
        Arc, Condvar, Mutex, OnceLock,
    },
    thread::JoinHandle,
    time::Duration,
};

use crate::engine::cuda::{api, CudaApi, CudaError, DeviceGuard, Event, Stream};
use safetensors::slice::{Gather, StridedCopy};

#[derive(Debug)]
pub enum LoaderError {
    AlreadyDelivered,
    Closed,
    Cuda(CudaError),
    CudaRuntimeLoad,
    InvalidDevice(i32),
    Io(std::io::Error),
    WorkerFailed(String),
}

impl Display for LoaderError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::AlreadyDelivered => write!(
                f,
                "tensor already delivered (prefetch hands each tensor out once, via take or iteration)"
            ),
            Self::Closed => write!(f, "prefetch loader closed"),
            Self::Cuda(e) => write!(f, "{e}"),
            Self::CudaRuntimeLoad => write!(
                f,
                "no CUDA runtime is loaded in this process; import a CUDA-enabled \
                 framework (torch) before calling prefetch()"
            ),
            Self::InvalidDevice(d) => write!(
                f,
                "device index {d} is out of range (0..{MAX_DEVICES})"
            ),
            Self::Io(e) => write!(f, "io error: {e}"),
            Self::WorkerFailed(msg) => f.write_str(msg),
        }
    }
}

impl std::error::Error for LoaderError {}

impl LoaderError {
    /// The load failed, as opposed to a tensor taken twice or a closed loader
    pub fn is_failure(&self) -> bool {
        matches!(self, Self::WorkerFailed(_))
    }
}

impl From<CudaError> for LoaderError {
    fn from(value: CudaError) -> Self {
        Self::Cuda(value)
    }
}

impl From<std::io::Error> for LoaderError {
    fn from(value: std::io::Error) -> Self {
        Self::Io(value)
    }
}

const CHUNK_SIZE: NonZeroUsize = NonZeroUsize::new(16 * 1024 * 1024).unwrap();

/// The largest alignment of offset, length and buffer a direct read may need
const DIRECT_ALIGN: usize = 4096;

/// A chunk's superset aligned to [`DIRECT_ALIGN`] fits in a slab
const SLAB_SIZE: usize = CHUNK_SIZE.get() + 2 * DIRECT_ALIGN;
const _: () = assert!(SLAB_SIZE.is_multiple_of(DIRECT_ALIGN));

/// The file the workers read chunks from
struct Source {
    file: Arc<File>,
    #[cfg(target_os = "linux")]
    direct: Option<Direct>,
}

impl Source {
    /// `direct` as in [`Loader::load`]. Direct reads need `aligned_slabs` and a
    /// filesystem that supports them; reads go through the page cache otherwise.
    fn new(file: Arc<File>, direct: Option<bool>, aligned_slabs: bool) -> Self {
        #[cfg(target_os = "linux")]
        {
            use std::os::fd::AsRawFd;
            let direct = match direct {
                Some(true) | None if aligned_slabs => Direct::open(&file, direct.is_none()),
                _ => None,
            };
            if direct.is_none() {
                // Chunks are read out of order by several threads and a plan may
                // skip regions: kernel readahead would pull in bytes nobody asked
                // for. Best effort, the hint failing only costs throughput.
                // SAFETY: `file` holds an open descriptor for the call's duration.
                let _ =
                    unsafe { libc::posix_fadvise(file.as_raw_fd(), 0, 0, libc::POSIX_FADV_RANDOM) };
            }
            Self { file, direct }
        }
        #[cfg(not(target_os = "linux"))]
        {
            let _ = (direct, aligned_slabs);
            Self { file }
        }
    }

    /// Reads `len` bytes at `offset` into `slab`, aligned to [`DIRECT_ALIGN`] and
    /// at least `len + 2 * DIRECT_ALIGN` long, and returns where they start in it.
    fn read(&self, slab: &mut [u8], offset: u64, len: usize) -> std::io::Result<usize> {
        #[cfg(target_os = "linux")]
        if let Some(direct) = self
            .direct
            .as_ref()
            .filter(|d| d.ok.load(Ordering::Relaxed))
        {
            if direct.cached_first && read_cached(&self.file, &mut slab[..len], offset) == len {
                return Ok(0);
            }
            match direct.read(slab, offset, len) {
                Err(e) if e.raw_os_error() == Some(libc::EINVAL) => {
                    direct.ok.store(false, Ordering::Relaxed);
                }
                res => return res,
            }
        }
        self.file.read_exact_at(&mut slab[..len], offset)?;
        Ok(0)
    }
}

/// The file reopened `O_DIRECT`
#[cfg(target_os = "linux")]
struct Direct {
    file: File,
    /// of offsets, lengths and buffers; divides [`DIRECT_ALIGN`]
    align: usize,
    /// whether a chunk the page cache holds is read from it
    cached_first: bool,
    /// cleared when a direct read fails with `EINVAL`
    ok: AtomicBool,
}

#[cfg(target_os = "linux")]
impl Direct {
    fn open(file: &File, cached_first: bool) -> Option<Self> {
        use std::os::{fd::AsRawFd, unix::fs::OpenOptionsExt};
        let align = direct_align(file)?;
        let file = std::fs::OpenOptions::new()
            .read(true)
            .custom_flags(libc::O_DIRECT)
            .open(format!("/proc/self/fd/{}", file.as_raw_fd()))
            .ok()?;
        Some(Self {
            file,
            align,
            cached_first,
            ok: AtomicBool::new(true),
        })
    }

    /// Reads the [`Self::align`]ed superset of `len` bytes at `offset` into
    /// `slab` and returns where they start in it; the end of the file may cut the
    /// superset short.
    fn read(&self, slab: &mut [u8], offset: u64, len: usize) -> std::io::Result<usize> {
        let head = (offset % self.align as u64) as usize;
        let from = offset - head as u64;
        let buf = &mut slab[..(head + len).next_multiple_of(self.align)];
        let mut got = 0;
        while got < head + len {
            match self.file.read_at(&mut buf[got..], from + got as u64) {
                Ok(0) => return Err(std::io::ErrorKind::UnexpectedEof.into()),
                Ok(n) => got += n,
                Err(e) if e.kind() == std::io::ErrorKind::Interrupted => {}
                Err(e) => return Err(e),
            }
        }
        Ok(head)
    }
}

/// The alignment direct reads of `file` need, as `statx` reports it, or
/// [`DIRECT_ALIGN`] where it does not; `None` if it does not divide
/// [`DIRECT_ALIGN`] or the file takes no direct reads.
#[cfg(target_os = "linux")]
#[cfg_attr(not(target_env = "gnu"), allow(unused_variables))]
fn direct_align(file: &File) -> Option<usize> {
    #[cfg(target_env = "gnu")]
    {
        use std::os::fd::AsRawFd;
        // SAFETY: `statx` is plain integers, valid zeroed; `file` holds an open
        // descriptor, and `AT_EMPTY_PATH` with an empty path describes it
        let (r, stx) = unsafe {
            let mut stx: libc::statx = std::mem::zeroed();
            let r = libc::statx(
                file.as_raw_fd(),
                c"".as_ptr(),
                libc::AT_EMPTY_PATH,
                libc::STATX_DIOALIGN,
                &mut stx,
            );
            (r, stx)
        };
        if r == 0 && stx.stx_mask & libc::STATX_DIOALIGN != 0 {
            let align = stx.stx_dio_mem_align.max(stx.stx_dio_offset_align) as usize;
            return (align != 0 && DIRECT_ALIGN.is_multiple_of(align)).then_some(align);
        }
    }
    Some(DIRECT_ALIGN)
}

/// Reads what the page cache holds of `buf.len()` bytes at `offset`, up to the
/// first byte it does not, without waiting for the disk.
#[cfg(target_os = "linux")]
fn read_cached(file: &File, buf: &mut [u8], offset: u64) -> usize {
    use std::os::fd::AsRawFd;
    let Ok(offset) = libc::off_t::try_from(offset) else {
        return 0;
    };
    let iov = libc::iovec {
        iov_base: buf.as_mut_ptr().cast(),
        iov_len: buf.len(),
    };
    // SAFETY: `iov` describes `buf`, writable for the call's duration
    let n = unsafe { libc::preadv2(file.as_raw_fd(), &iov, 1, offset, libc::RWF_NOWAIT) };
    n.max(0) as usize
}

/// A byte interval `start..end` of the file's data section
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Span {
    pub start: usize,
    pub end: usize,
}

/// Geometry of one load. Every interval here is a byte range within the file's
/// data section; device addresses never appear in the plan.
struct LoadPlan {
    spans: Box<[Span]>,
    /// per span, the copies that pack its bytes into its destination; `None` for
    /// one plain copy
    gathers: Box<[Option<Gather>]>,
    /// per span, the `chunks` that carry its bytes; empty for a zero-size span
    span_chunks: Box<[Range<usize>]>,
    /// data section byte interval of each chunk: at most [`CHUNK_SIZE`] of
    /// adjacent spans, never across a gap
    chunks: Box<[Range<usize>]>,
    /// per chunk, the spans it carries bytes of
    chunk_spans: Box<[Range<usize>]>,
    in_file_offset: usize,
}

/// The part of `copy` whose source bytes lie in `window`, sources relative to
/// `window.start`: the rows inside it as one copy, and the rows its edges cut (at
/// most two) as single-row copies of their part inside it.
fn clip(copy: &StridedCopy, window: &Range<usize>) -> [Option<StridedCopy>; 3] {
    let row = |k: usize| copy.src + k * copy.src_pitch;
    let part = |k: usize| {
        let (start, end) = (
            row(k).max(window.start),
            (row(k) + copy.width).min(window.end),
        );
        (start < end).then(|| StridedCopy {
            src: start - window.start,
            dst: copy.dst + k * copy.dst_pitch + (start - row(k)),
            width: end - start,
            height: 1,
            src_pitch: 0,
            dst_pitch: 0,
        })
    };
    if copy.height == 1 {
        return [part(0), None, None];
    }
    if window.end <= copy.src {
        return [None; 3];
    }
    // rows are disjoint and in order (src_pitch >= width): the ones meeting the
    // window are consecutive
    let first = window
        .start
        .checked_sub(copy.src + copy.width)
        .map_or(0, |past| past / copy.src_pitch + 1);
    let last = ((window.end - copy.src - 1) / copy.src_pitch).min(copy.height - 1);
    if first > last {
        return [None; 3];
    }
    let whole = |k: usize| row(k) >= window.start && row(k) + copy.width <= window.end;
    let (mut lo, mut hi) = (first, last);
    let head = if whole(lo) {
        None
    } else {
        lo += 1;
        part(lo - 1)
    };
    let tail = if lo <= hi && !whole(hi) {
        hi -= 1;
        part(hi + 1)
    } else {
        None
    };
    let rows = (lo <= hi).then(|| StridedCopy {
        src: row(lo) - window.start,
        dst: copy.dst + lo * copy.dst_pitch,
        height: hi - lo + 1,
        ..*copy
    });
    [head, rows, tail]
}

impl LoadPlan {
    fn new(
        spans: Vec<Span>,
        gathers: Vec<Option<Gather>>,
        in_file_offset: usize,
        chunk_size: NonZeroUsize,
    ) -> Self {
        let chunk_size = chunk_size.get();
        assert!(
            spans.iter().all(|s| s.start <= s.end)
                && spans.windows(2).all(|w| w[0].end <= w[1].start),
            "spans must be sorted by start and disjoint, each with start <= end"
        );
        assert_eq!(gathers.len(), spans.len(), "one gather entry per span");

        let mut chunks: Vec<Range<usize>> = Vec::new();
        let mut chunk_spans: Vec<Range<usize>> = Vec::new();
        let mut span_chunks = vec![0..0; spans.len()];
        for (i, span) in spans.iter().enumerate() {
            let mut start = span.start;
            let mut first = chunks.len();
            while start < span.end {
                let n = chunks.len();
                match chunks.last_mut() {
                    // the previous span's last chunk ends here and has room left
                    Some(chunk) if chunk.end == start && chunk.len() < chunk_size => {
                        chunk.end = (chunk.start + chunk_size).min(span.end);
                        chunk_spans.last_mut().unwrap().end = i + 1;
                        first = first.min(n - 1);
                        start = chunk.end;
                    }
                    _ => {
                        let end = (start + chunk_size).min(span.end);
                        chunks.push(start..end);
                        chunk_spans.push(i..i + 1);
                        start = end;
                    }
                }
            }
            span_chunks[i] = first..chunks.len();
        }

        Self {
            spans: spans.into_boxed_slice(),
            gathers: gathers.into_boxed_slice(),
            span_chunks: span_chunks.into_boxed_slice(),
            chunks: chunks.into_boxed_slice(),
            chunk_spans: chunk_spans.into_boxed_slice(),
            in_file_offset,
        }
    }

    /// Issues the copies of chunk `c` from its slab, sources as offsets in the slab
    /// and destinations as addresses: `dst(s)` is span `s`'s destination and
    /// `allocation_end(d)` the end of the allocation holding `d`, if known.
    fn chunk_copies<E>(
        &self,
        c: usize,
        dst: impl Fn(usize) -> u64,
        mut allocation_end: impl FnMut(u64) -> Option<u64>,
        mut copy: impl FnMut(StridedCopy) -> Result<(), E>,
    ) -> Result<(), E> {
        let chunk = self.chunks[c].clone();
        let plain = |(dst, at, len, _): (u64, usize, usize, u64)| StridedCopy {
            src: at - chunk.start,
            dst: dst as usize,
            width: len,
            height: 1,
            src_pitch: 0,
            dst_pitch: 0,
        };
        // one copy per run of pieces adjacent in the file and on the device,
        // within one allocation: (device address, data section offset, length,
        // end of the allocation, looked up once the run grows)
        let mut run: Option<(u64, usize, usize, u64)> = None;
        for s in self.chunk_spans[c].clone() {
            let span = self.spans[s];
            let (start, end) = (span.start.max(chunk.start), span.end.min(chunk.end));
            if start >= end {
                continue;
            }
            if let Some(gather) = &self.gathers[s] {
                if let Some(run) = run.take() {
                    copy(plain(run))?;
                }
                let window = start - span.start..end - span.start;
                for piece in gather
                    .copies
                    .iter()
                    .flat_map(|g| clip(g, &window))
                    .flatten()
                {
                    copy(StridedCopy {
                        src: start - chunk.start + piece.src,
                        dst: dst(s) as usize + piece.dst,
                        ..piece
                    })?;
                }
                continue;
            }
            let dst = dst(s) + (start - span.start) as u64;
            if let Some((d0, at, len, limit)) = run.as_mut() {
                if *at + *len == start && *d0 + *len as u64 == dst {
                    if *limit == 0 {
                        *limit = allocation_end(*d0).unwrap_or(dst);
                    }
                    if dst + (end - start) as u64 <= *limit {
                        *len += end - start;
                        continue;
                    }
                }
            }
            if let Some(run) = run.replace((dst, start, end - start, 0)) {
                copy(plain(run))?;
            }
        }
        if let Some(run) = run {
            copy(plain(run))?;
        }
        Ok(())
    }
}

enum Sink<D> {
    Cuda(CudaSink<D>),
}

impl<D> Sink<D> {
    /// `read(slab, offset, len)` fills the chunk's slab from the source and
    /// returns where the chunk starts in it
    fn load_chunk(
        &self,
        chunk_idx: usize,
        read: impl FnOnce(&mut [u8], u64, usize) -> std::io::Result<usize>,
    ) -> Result<(), LoaderError> {
        match self {
            Self::Cuda(sink) => sink.load_chunk(chunk_idx, read),
        }
    }

    fn wait_ready(&self, span: usize) -> Result<(), LoaderError> {
        match self {
            Self::Cuda(sink) => sink.wait_ready(span),
        }
    }

    fn take(&self, idx: usize) -> Result<D, LoaderError> {
        match self {
            Self::Cuda(sink) => sink.take(idx),
        }
    }

    fn signal_stop(&self) {
        match self {
            Self::Cuda(sink) => sink.signal_stop(),
        }
    }

    fn stopped(&self) -> bool {
        match self {
            Self::Cuda(sink) => sink.closed.load(Ordering::Acquire),
        }
    }

    fn release(&self) -> Vec<D> {
        match self {
            Self::Cuda(sink) => sink.close(),
        }
    }

    fn fully_scheduled(&self, span: usize) -> bool {
        match self {
            Self::Cuda(sink) => sink.fully_scheduled(span),
        }
    }

    fn provide(
        &self,
        dsts: Vec<(u64, D)>,
        stream: Stream,
    ) -> Result<(), (LoaderError, Vec<(u64, D)>)> {
        match self {
            Self::Cuda(sink) => sink.provide(dsts, stream),
        }
    }

    fn provided(&self) -> usize {
        match self {
            Self::Cuda(sink) => sink.provided.load(Ordering::Acquire),
        }
    }

    fn scheduled(&self, claimed: usize) -> usize {
        match self {
            Self::Cuda(sink) => sink.scheduled(claimed),
        }
    }
}

#[derive(Clone, Copy)]
struct Slab {
    offset: usize,
    /// (completion event of the last host2device copy, device id)
    last_copy: Option<(Event, i32)>,
}

struct SlabLease {
    pool: &'static SlabPool,
    slab: Slab,
}

impl SlabLease {
    fn ptr(&self) -> *mut u8 {
        unsafe { self.pool.buffer_ptr.add(self.slab.offset) }
    }

    fn slab(&mut self) -> &mut [u8] {
        // SAFETY: the lease holds the slab, `slab_size` bytes of the pool's buffer
        unsafe { std::slice::from_raw_parts_mut(self.ptr(), self.pool.slab_size) }
    }

    fn release(mut self, d: &DeviceGuard<'_>, stream: Stream) -> Result<(), CudaError> {
        let event = match self.slab.last_copy.take() {
            Some((event, device)) if device == d.id() => event,
            Some((event, _)) => {
                let _ = d.api().event_destroy(event);
                d.event_create()?
            }
            None => d.event_create()?,
        };
        self.slab.last_copy = Some((event, d.id()));
        d.api().event_record(event, stream)
    }
}

impl Drop for SlabLease {
    fn drop(&mut self) {
        self.pool.put_back(self.slab);
    }
}

struct SlabPool {
    buffer_ptr: *mut u8,
    slab_size: usize,
    /// whether every slab is aligned to [`DIRECT_ALIGN`]
    aligned: bool,
    free: Mutex<VecDeque<Slab>>,
    available: Condvar,
}

unsafe impl Send for SlabPool {}
unsafe impl Sync for SlabPool {}

impl SlabPool {
    fn new(cuda: &CudaApi, n_slabs: usize, slab_size: usize) -> Result<Self, CudaError> {
        let buffer_ptr = cuda.host_alloc(n_slabs * slab_size)?;
        let free = (0..n_slabs)
            .map(|i| Slab {
                offset: i * slab_size,
                last_copy: None,
            })
            .collect();
        Ok(Self {
            buffer_ptr,
            slab_size,
            aligned: buffer_ptr.addr().is_multiple_of(DIRECT_ALIGN)
                && slab_size.is_multiple_of(DIRECT_ALIGN),
            free: Mutex::new(free),
            available: Condvar::new(),
        })
    }

    fn acquire(&'static self, cuda: &CudaApi) -> Result<SlabLease, CudaError> {
        // TODO: handle poisoning
        let mut free = self.free.lock().unwrap();
        let slab = loop {
            if let Some(s) = free.pop_front() {
                break s;
            }
            free = self.available.wait(free).unwrap();
        };
        drop(free);
        let lease = SlabLease { pool: self, slab };
        if let Some((event, _)) = lease.slab.last_copy {
            cuda.event_sync(event)?;
        }
        Ok(lease)
    }

    fn put_back(&self, slab: Slab) {
        self.free.lock().unwrap().push_back(slab);
        self.available.notify_one();
    }
}

struct CudaSink<D> {
    api: &'static CudaApi,
    device: i32,
    stream: Stream,
    pool: &'static SlabPool,
    plan: Arc<LoadPlan>,
    /// per span, the device address its bytes are copied to; set by `provide`
    dsts: Box<[AtomicU64]>,
    /// per span, the owner of its destination; `None` until provided, and once
    /// taken or released
    owners: Box<[Mutex<Option<D>>]>,
    /// per span, the event of the batch that provided it, recorded on the
    /// allocating stream: the destination is free to write once it completes
    ready: Box<[AtomicU64]>,
    /// the spans with a destination: a prefix of the plan
    provided: AtomicUsize,
    /// guards `provided` advancing and `released` turning on, so neither is missed
    /// by a waiter on `provision`
    provision_lock: Mutex<()>,
    provision: Condvar,
    batch_events: Mutex<Vec<Event>>,
    chunk_completion_events: Box<[AtomicU64]>,
    closed: AtomicBool,
    released: AtomicBool,
}

unsafe impl<D: Send> Send for CudaSink<D> {}
unsafe impl<D: Send> Sync for CudaSink<D> {}

impl<D> CudaSink<D> {
    fn new(
        api: &'static CudaApi,
        pool: &'static SlabPool,
        plan: Arc<LoadPlan>,
        device: i32,
    ) -> Result<Self, LoaderError> {
        let stream = device_stream(api, device)?;
        let n = plan.spans.len();
        Ok(Self {
            api,
            device,
            stream,
            pool,
            dsts: (0..n).map(|_| AtomicU64::new(0)).collect(),
            owners: (0..n).map(|_| Mutex::new(None)).collect(),
            ready: (0..n).map(|_| AtomicU64::new(0)).collect(),
            provided: AtomicUsize::new(0),
            provision_lock: Mutex::new(()),
            provision: Condvar::new(),
            batch_events: Mutex::new(Vec::new()),
            chunk_completion_events: (0..plan.chunks.len()).map(|_| AtomicU64::new(0)).collect(),
            closed: AtomicBool::new(false),
            released: AtomicBool::new(false),
            plan,
        })
    }

    /// Gives the next `dsts.len()` spans their destinations, allocated on
    /// `alloc_stream`. Hands `dsts` back if the sink is released or the event fails.
    fn provide(
        &self,
        dsts: Vec<(u64, D)>,
        alloc_stream: Stream,
    ) -> Result<(), (LoaderError, Vec<(u64, D)>)> {
        let _g = self.provision_lock.lock().unwrap();
        if self.released.load(Ordering::Acquire) {
            return Err((LoaderError::Closed, dsts));
        }
        let first = self.provided.load(Ordering::Acquire);
        assert!(
            first + dsts.len() <= self.plan.spans.len(),
            "more destinations than spans"
        );
        // the caller's allocator may hand out memory still in use by work queued
        // on the stream it allocated on: workers wait for that work before copying
        let ready = self.api.with_device(self.device, |d| {
            let e = d.event_create()?;
            if let Err(err) = self.api.event_record(e, alloc_stream) {
                let _ = self.api.event_destroy(e);
                return Err(err);
            }
            Ok(e)
        });
        let ready = match ready {
            Ok(e) => e,
            Err(err) => return Err((err.into(), dsts)),
        };
        self.batch_events.lock().unwrap().push(ready);
        let n = dsts.len();
        for (i, (ptr, owner)) in dsts.into_iter().enumerate() {
            self.dsts[first + i].store(ptr, Ordering::Relaxed);
            self.ready[first + i].store(ready as u64, Ordering::Relaxed);
            *self.owners[first + i].lock().unwrap() = Some(owner);
        }
        self.provided.store(first + n, Ordering::Release);
        self.provision.notify_all();
        Ok(())
    }

    /// Waits until the first `n` spans have a destination.
    fn wait_provided(&self, n: usize) -> Result<(), LoaderError> {
        if self.provided.load(Ordering::Acquire) >= n {
            return Ok(());
        }
        let mut g = self.provision_lock.lock().unwrap();
        loop {
            if self.provided.load(Ordering::Acquire) >= n {
                return Ok(());
            }
            if self.closed.load(Ordering::Acquire) {
                return Err(LoaderError::Closed);
            }
            g = self.provision.wait(g).unwrap();
        }
    }

    fn load_chunk(
        &self,
        chunk_idx: usize,
        read: impl FnOnce(&mut [u8], u64, usize) -> std::io::Result<usize>,
    ) -> Result<(), LoaderError> {
        let chunk = self.plan.chunks[chunk_idx].clone();
        let spans = self.plan.chunk_spans[chunk_idx].clone();
        self.wait_provided(spans.end)?;
        // host-side: a pending copy would hold its slab and stall other loaders
        let mut last = 0;
        for s in spans {
            let e = self.ready[s].load(Ordering::Relaxed);
            if e != last {
                self.api.event_sync(e as Event)?;
                last = e;
            }
        }
        let mut lease = self.pool.acquire(self.api)?;
        let head = read(
            lease.slab(),
            (self.plan.in_file_offset + chunk.start) as u64,
            chunk.len(),
        )?;

        let e = self.api.with_device(self.device, |d| {
            self.plan.chunk_copies(
                chunk_idx,
                |s| self.dsts[s].load(Ordering::Relaxed),
                |d0| self.api.allocation_end(d0),
                |c| {
                    let src = lease.ptr().wrapping_add(head + c.src);
                    match c.height {
                        1 => self
                            .api
                            .memcpy_h2d_async(c.dst as u64, src, c.width, self.stream),
                        _ => self.api.memcpy2d_h2d_async(
                            c.dst as u64,
                            c.dst_pitch,
                            src,
                            c.src_pitch,
                            c.width,
                            c.height,
                            self.stream,
                        ),
                    }
                },
            )?;
            lease.release(d, self.stream)?;
            let e = d.event_create()?;
            if let Err(err) = self.api.event_record(e, self.stream) {
                let _ = self.api.event_destroy(e);
                return Err(err);
            }
            Ok(e)
        })?;
        self.chunk_completion_events[chunk_idx].store(e as u64, Ordering::Release);

        Ok(())
    }

    fn wait_ready(&self, span: usize) -> Result<(), LoaderError> {
        // a zero-size span has no chunk: it is ready once provided
        self.wait_provided(span + 1)?;
        for chunk in self.plan.span_chunks[span].clone() {
            let mut i = 0;
            let e = loop {
                match self.chunk_completion_events[chunk].load(Ordering::Acquire) {
                    0 => {
                        if self.closed.load(Ordering::Acquire) {
                            return Err(LoaderError::Closed);
                        }
                        if i < 64 {
                            std::thread::yield_now();
                            i += 1;
                        } else {
                            std::thread::sleep(Duration::from_micros(200));
                        }
                    }
                    e => break e as Event,
                }
            };
            self.api.event_sync(e)?;
        }
        Ok(())
    }

    fn take(&self, idx: usize) -> Result<D, LoaderError> {
        match self.owners[idx].lock().unwrap().take() {
            Some(owner) => Ok(owner),
            None if self.released.load(Ordering::Acquire) => Err(LoaderError::Closed),
            None => Err(LoaderError::AlreadyDelivered),
        }
    }

    /// Called once the workers are joined: waits for the copies in flight, then
    /// returns the owners not taken, or leaks them if the wait fails.
    fn close(&self) -> Vec<D> {
        {
            let _g = self.provision_lock.lock().unwrap();
            self.closed.store(true, Ordering::Release);
            self.released.store(true, Ordering::Release);
            self.provision.notify_all();
        }
        let synced: Result<(), CudaError> = self.api.with_device(self.device, |d| {
            let e = d.event_create()?;
            let res = self
                .api
                .event_record(e, self.stream)
                .and_then(|()| self.api.event_sync(e));
            let _ = self.api.event_destroy(e);
            res
        });
        let owners: Vec<D> = self
            .owners
            .iter()
            .filter_map(|o| o.lock().unwrap().take())
            .collect();
        if synced.is_err() {
            // copies may still be queued into them: leak them rather than free them
            std::mem::forget(owners);
            return Vec::new();
        }
        owners
    }

    fn signal_stop(&self) {
        let _g = self.provision_lock.lock().unwrap();
        self.closed.store(true, Ordering::Release);
        self.provision.notify_all();
    }

    fn fully_scheduled(&self, span: usize) -> bool {
        self.provided.load(Ordering::Acquire) > span
            && self.plan.span_chunks[span]
                .clone()
                .all(|c| self.chunk_completion_events[c].load(Ordering::Acquire) != 0)
    }

    /// How many spans the claimed chunks reach
    fn scheduled(&self, claimed: usize) -> usize {
        match claimed.min(self.plan.chunks.len()) {
            n if n == self.plan.chunks.len() => self.plan.spans.len(),
            0 => 0,
            n => self.plan.chunk_spans[n - 1].end,
        }
    }
}

impl<D> Drop for CudaSink<D> {
    fn drop(&mut self) {
        self.close();
        for e in self.batch_events.get_mut().unwrap().drain(..) {
            let _ = self.api.event_destroy(e);
        }
        for e in self.chunk_completion_events.iter() {
            let e = e.swap(0, Ordering::AcqRel);
            if e != 0 {
                let _ = self.api.event_destroy(e as Event);
            }
        }
    }
}

/// Whether a CUDA runtime is loaded in this process, as [`Loader::load`] needs
pub fn available() -> bool {
    api().is_some()
}

/// This is a generous limit on the maximum number of devices that could be connected to a single host
const MAX_DEVICES: usize = 64;
static STREAMS: [AtomicU64; MAX_DEVICES] = [const { AtomicU64::new(0) }; MAX_DEVICES];

fn get_stream(stream_slots: &[AtomicU64], api: &CudaApi, device: i32) -> Result<Stream, CudaError> {
    let slot = stream_slots
        .get(device as usize)
        .unwrap_or_else(|| panic!("device index {device} exceeds MAX_DEVICES ({MAX_DEVICES})"));
    let stream = slot.load(Ordering::Acquire);
    if stream != 0 {
        return Ok(stream as Stream);
    }

    let new = api.with_device(device, |d| d.stream_create())?;
    match slot.compare_exchange(0, new as u64, Ordering::AcqRel, Ordering::Acquire) {
        Ok(_) => Ok(new),
        Err(existing) => {
            let _ = api.stream_destroy(new);
            Ok(existing as Stream)
        }
    }
}

fn device_stream(api: &CudaApi, device: i32) -> Result<Stream, CudaError> {
    get_stream(&STREAMS, api, device)
}

struct LoaderInner<D> {
    source: Source,
    plan: Arc<LoadPlan>,
    sink: Sink<D>,
    next_chunk: AtomicUsize,
    /// guards `next_chunk` advancing and the load stopping, so neither is missed
    /// by a waiter on `schedule`
    schedule_lock: Mutex<()>,
    schedule: Condvar,
    error: OnceLock<LoaderError>,
    workers: Mutex<Vec<JoinHandle<()>>>,
}

impl<D> LoaderInner<D> {
    fn signal_stop(&self) {
        self.sink.signal_stop();
        self.notify_schedule();
    }

    fn notify_schedule(&self) {
        drop(self.schedule_lock.lock().unwrap());
        self.schedule.notify_all();
    }

    /// Stops loading and returns the owners not taken. A `take_tensor` waiting in
    /// another thread returns its owner or an error: `Closed`, or what failed the load.
    fn close(&self) -> Vec<D> {
        self.halt();
        self.sink.release()
    }

    /// Signals the workers and joins them; the owners stay in the sink.
    fn halt(&self) {
        self.signal_stop();
        // held while joining: a concurrent halt returns once the workers are gone
        let mut workers = self.workers.lock().unwrap();
        for h in workers.drain(..) {
            let _ = h.join();
        }
    }

    fn take_tensor(&self, idx: usize) -> Result<D, LoaderError> {
        self.sink
            .wait_ready(idx)
            .and_then(|()| self.sink.take(idx))
            .map_err(|e| match (e, self.error.get()) {
                (LoaderError::Closed, Some(err)) => LoaderError::WorkerFailed(err.to_string()),
                (e, _) => e,
            })
    }
}

pub struct Loader<D> {
    inner: Arc<LoaderInner<D>>,
}

/// The side of a [`Loader`] that gives its spans their destinations, see
/// [`Loader::feed`].
pub struct Feed<D> {
    inner: Arc<LoaderInner<D>>,
}

impl<D> Feed<D> {
    /// Gives the next `dsts.len()` spans, in plan order, their device address
    /// and owner. They were allocated on `stream`: workers wait for the work
    /// queued on it so far before writing. One caller at a time. Hands `dsts`
    /// back when the loader is closed or the event can't be recorded.
    #[allow(clippy::type_complexity)]
    pub fn provide(
        &self,
        dsts: Vec<(u64, D)>,
        stream: u64,
    ) -> Result<(), (LoaderError, Vec<(u64, D)>)> {
        self.inner.sink.provide(dsts, stream as Stream)
    }

    /// How many spans have a destination
    pub fn provided(&self) -> usize {
        self.inner.sink.provided()
    }

    /// How many spans the read schedule has reached: at least the ones a
    /// worker is reading or waiting to read
    pub fn scheduled(&self) -> usize {
        let claimed = self.inner.next_chunk.load(Ordering::Relaxed);
        self.inner.sink.scheduled(claimed)
    }

    /// Whether the load stopped: closed, failed or halted
    pub fn stopped(&self) -> bool {
        self.inner.sink.stopped()
    }

    /// Stops the load as a worker error would: `take` of a span not loaded yet
    /// fails with `reason`. No-op once the load stopped.
    pub fn fail(&self, reason: String) {
        if self.inner.sink.stopped() {
            return;
        }
        let _ = self.inner.error.set(LoaderError::WorkerFailed(reason));
        self.inner.signal_stop();
    }

    /// Waits until the read schedule is past `scheduled` spans or the load stopped
    pub fn wait_scheduled(&self, scheduled: usize) {
        let mut g = self.inner.schedule_lock.lock().unwrap();
        while self.scheduled() == scheduled && !self.stopped() {
            g = self.inner.schedule.wait(g).unwrap();
        }
    }

    /// Stops the load without joining the workers
    pub fn stop(&self) {
        self.inner.signal_stop();
    }

    /// Stops the load and joins the workers; the owners stay until `close`.
    pub fn halt(&self) {
        self.inner.halt();
    }

    /// Stops the load, joins the workers and returns the owners not taken.
    pub fn close(&self) -> Vec<D> {
        self.inner.close()
    }
}

impl<D: Send + 'static> Loader<D> {
    /// Starts the workers. They copy a span once it has a destination: give
    /// them with [`Feed::provide`], in plan order, as the load advances.
    /// `gathers` has one entry per span: `Some` when its destination is
    /// [`Gather::len`] bytes packed by the gather's copies (see
    /// [`safetensors::slice::slice_layout`]) rather than the span's bytes.
    /// `direct`: `Some(false)` reads chunks through the page cache, `Some(true)`
    /// with `O_DIRECT`, `None` through the page cache when it holds the whole
    /// chunk and with `O_DIRECT` otherwise.
    pub fn load(
        file: Arc<File>,
        direct: Option<bool>,
        in_file_offset: usize,
        device: i32,
        threads: usize,
        spans: Vec<Span>,
        gathers: Vec<Option<Gather>>,
    ) -> Result<Self, LoaderError> {
        let cuda_api = api().ok_or(LoaderError::CudaRuntimeLoad)?;
        if !(0..MAX_DEVICES as i32).contains(&device) {
            return Err(LoaderError::InvalidDevice(device));
        }
        let plan = Arc::new(LoadPlan::new(spans, gathers, in_file_offset, CHUNK_SIZE));
        // avoid spawning too many threads when not needed, up to chunks.len()
        let threads = threads.clamp(1, plan.chunks.len().max(1));
        let pool = cuda_api.with_device(device, |_| pool(cuda_api))?;
        let inner = Arc::new(LoaderInner {
            source: Source::new(file, direct, pool.aligned),
            error: OnceLock::new(),
            plan: plan.clone(),
            sink: Sink::Cuda(CudaSink::new(cuda_api, pool, plan, device)?),
            next_chunk: AtomicUsize::new(0),
            schedule_lock: Mutex::new(()),
            schedule: Condvar::new(),
            workers: Mutex::new(Vec::with_capacity(threads)),
        });
        let loader = Self { inner };
        for _ in 0..threads {
            let inner = loader.inner.clone();
            let spawned = std::thread::Builder::new().spawn(move || {
                let _ = cuda_api.set_device(device);
                worker(inner)
            });
            // on failure, dropping the loader stops and joins the workers already started
            loader.inner.workers.lock().unwrap().push(spawned?);
        }
        Ok(loader)
    }
}

impl<D> Loader<D> {
    /// `idx` indexes [`Loader::spans`]
    pub fn take_tensor(&self, idx: usize) -> Result<D, LoaderError> {
        self.inner.take_tensor(idx)
    }

    pub fn feed(&self) -> Feed<D> {
        Feed {
            inner: self.inner.clone(),
        }
    }

    /// See [`LoaderInner::close`].
    pub fn close(&self) -> Vec<D> {
        self.inner.close()
    }

    pub fn iter(&self) -> TensorIter<D> {
        TensorIter {
            inner: self.inner.clone(),
            pending: (0..self.inner.plan.spans.len()).collect(),
        }
    }
}

impl<D> Drop for Loader<D> {
    fn drop(&mut self) {
        self.close();
    }
}

pub struct TensorIter<D> {
    inner: Arc<LoaderInner<D>>,
    pending: VecDeque<usize>,
}

impl<D> Iterator for TensorIter<D> {
    type Item = Result<(usize, D), LoaderError>;

    fn next(&mut self) -> Option<Self::Item> {
        loop {
            if self.pending.is_empty() {
                return None;
            }
            let pos = self
                .pending
                .iter()
                .take(64)
                .position(|&idx| self.inner.sink.fully_scheduled(idx))
                .unwrap_or(0);
            let idx = self.pending[pos];
            match self.inner.take_tensor(idx) {
                Ok(owner) => {
                    self.pending.remove(pos);
                    return Some(Ok((idx, owner)));
                }
                Err(LoaderError::AlreadyDelivered) => self.pending.remove(pos),
                Err(e) => {
                    self.pending.clear();
                    return Some(Err(e));
                }
            };
        }
    }
}

fn pool(cuda_api: &'static CudaApi) -> Result<&'static SlabPool, CudaError> {
    static POOL: OnceLock<SlabPool> = OnceLock::new();
    static INIT: Mutex<()> = Mutex::new(());
    if POOL.get().is_none() {
        let _g = INIT.lock().unwrap();
        if POOL.get().is_none() {
            let _ = POOL.set(SlabPool::new(cuda_api, 32, SLAB_SIZE)?);
        }
    }
    Ok(POOL.get().unwrap())
}

fn worker<D>(loader: Arc<LoaderInner<D>>) {
    loop {
        if loader.sink.stopped() {
            return;
        }
        let chunk_idx = loader.next_chunk.fetch_add(1, Ordering::Relaxed);
        loader.notify_schedule();
        if chunk_idx >= loader.plan.chunks.len() {
            return;
        }
        let res = loader.sink.load_chunk(chunk_idx, |slab, offset, len| {
            loader.source.read(slab, offset, len)
        });
        if let Err(err) = res {
            if !matches!(err, LoaderError::Closed) {
                let _ = loader.error.set(err);
            }
            loader.signal_stop();
            return;
        }
    }
}

#[cfg(test)]
#[allow(clippy::single_range_in_vec_init)] // one range per dimension
mod tests {
    use std::{num::NonZeroUsize, ops::Range};

    use super::{LoadPlan, Span, DIRECT_ALIGN, SLAB_SIZE};
    use safetensors::slice::{slice_layout, Gather, StridedCopy, TensorIndexer};
    use safetensors::Dtype;
    #[cfg(target_os = "linux")]
    use {
        super::Source,
        std::{fs::File, sync::Arc},
    };

    fn sp(start: usize, end: usize) -> Span {
        Span { start, end }
    }

    fn plan(spans: Vec<Span>, chunk_size: usize) -> LoadPlan {
        let gathers = vec![None; spans.len()];
        let plan = LoadPlan::new(spans, gathers, 0, NonZeroUsize::new(chunk_size).unwrap());
        check_plan(&plan, chunk_size);
        plan
    }

    /// A device of `len` bytes of 0xAA after the plan's copies of `file`, span `s`
    /// landing at `dsts[s]`, all in one allocation
    fn land(plan: &LoadPlan, file: &[u8], dsts: &[usize], len: usize) -> Vec<u8> {
        land_read(plan, dsts, len, |slab, offset, n| {
            let at = offset as usize - plan.in_file_offset;
            slab[..n].copy_from_slice(&file[at..at + n]);
            0
        })
    }

    /// [`land`] with each chunk read as the sink reads it: `read(slab, offset,
    /// len)` fills the slab and returns where the chunk starts in it
    fn land_read(
        plan: &LoadPlan,
        dsts: &[usize],
        len: usize,
        mut read: impl FnMut(&mut [u8], u64, usize) -> usize,
    ) -> Vec<u8> {
        let mut device = vec![0xAA; len];
        let (mut buf, at) = slab_buf(SLAB_SIZE, 0);
        for c in 0..plan.chunks.len() {
            let chunk = plan.chunks[c].clone();
            let offset = (plan.in_file_offset + chunk.start) as u64;
            let head = read(&mut buf[at.clone()], offset, chunk.len());
            let slab = &buf[at.start + head..];
            let copy = |cp: StridedCopy| -> Result<(), ()> {
                for r in 0..cp.height {
                    let (src, dst) = (cp.src + r * cp.src_pitch, cp.dst + r * cp.dst_pitch);
                    device[dst..dst + cp.width].copy_from_slice(&slab[src..src + cp.width]);
                }
                Ok(())
            };
            plan.chunk_copies(c, |s| dsts[s] as u64, |_| Some(len as u64), copy)
                .unwrap();
        }
        device
    }

    struct Rng(u64);
    impl Rng {
        fn below(&mut self, n: usize) -> usize {
            self.0 = self
                .0
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            ((self.0 >> 33) % n as u64) as usize
        }
    }

    /// The slice's bytes by brute force over element indices, from a tensor starting at byte 0 of `bytes`.
    fn slice_bytes(
        bytes: &[u8],
        shape: &[usize],
        elem: usize,
        intervals: &[Vec<Range<usize>>],
    ) -> Vec<u8> {
        fn rec(
            d: usize,
            at: usize,
            strides: &[usize],
            ivs: &[Vec<Range<usize>>],
            elem: usize,
            b: &[u8],
            out: &mut Vec<u8>,
        ) {
            if d == strides.len() {
                out.extend_from_slice(&b[at..at + elem]);
                return;
            }
            for r in &ivs[d] {
                for i in r.clone() {
                    rec(d + 1, at + i * strides[d], strides, ivs, elem, b, out);
                }
            }
        }
        let mut strides = vec![elem; shape.len()];
        for d in (0..shape.len().saturating_sub(1)).rev() {
            strides[d] = strides[d + 1] * shape[d + 1];
        }
        let mut out = Vec::new();
        rec(0, 0, &strides, intervals, elem, bytes, &mut out);
        out
    }

    #[test]
    fn test_random_slices_land_exactly() {
        let mut rng = Rng(7);
        for trial in 0..2000 {
            // a few tensors laid out back to back, as in a safetensors data section
            let n_tensors = 1 + rng.below(4);
            let mut layout = Vec::new(); // (start, shape, elem)
            let mut cursor = 0;
            for _ in 0..n_tensors {
                let shape: Vec<usize> = (0..1 + rng.below(4)).map(|_| 1 + rng.below(6)).collect();
                let elem = [1, 2, 4][rng.below(3)];
                layout.push((cursor, shape.clone(), elem));
                cursor += shape.iter().product::<usize>() * elem;
            }
            let file: Vec<u8> = (0..cursor).map(|_| rng.below(0xAA) as u8).collect();

            let (mut spans, mut gathers, mut dsts) = (Vec::new(), Vec::new(), Vec::new());
            // destinations back to back or a few bytes apart
            let mut expected = Vec::new();
            for (start, shape, elem) in &layout {
                if rng.below(4) == 0 {
                    continue; // left out of the plan
                }
                let intervals: Vec<Vec<Range<usize>>> = shape
                    .iter()
                    .map(|&n| match rng.below(3) {
                        0 => vec![0..n],
                        _ => {
                            let mut cuts: Vec<usize> = (0..2 * (1 + rng.below(2)))
                                .map(|_| rng.below(n + 1))
                                .collect();
                            cuts.sort_unstable();
                            cuts.dedup();
                            cuts.chunks(2)
                                .filter(|c| c.len() == 2)
                                .map(|c| c[0]..c[1])
                                .collect()
                        }
                    })
                    .collect();
                let dtype = [Dtype::U8, Dtype::U16, Dtype::U32]
                    [[1, 2, 4].iter().position(|e| e == elem).unwrap()];
                let indexers: Vec<Vec<TensorIndexer>> = intervals
                    .iter()
                    .map(|d| d.iter().map(|r| TensorIndexer::from(r.clone())).collect())
                    .collect();
                let r = slice_layout(dtype, shape, &indexers).unwrap();
                let nbytes = shape.iter().product::<usize>() * elem;
                let bytes = slice_bytes(&file[*start..*start + nbytes], shape, *elem, &intervals);
                expected.resize(expected.len() + [0, 3][rng.below(2)], 0xAA);
                dsts.push(expected.len());
                expected.extend(bytes);
                spans.push(sp(start + r.read.start, start + r.read.end));
                gathers.push(r.gather);
            }
            let chunk = 1 + rng.below(48);
            let plan = LoadPlan::new(spans, gathers, 0, NonZeroUsize::new(chunk).unwrap());
            assert_eq!(
                land(&plan, &file, &dsts, expected.len()),
                expected,
                "trial {trial}, chunk {chunk}"
            );
        }
    }

    #[test]
    fn test_gather_rows_cut_by_chunk_edges() {
        // columns 2..5 of a 4 x 10 tensor between two plain spans, adjacent in the
        // file and on the device: 7-byte chunks cut its rows, and the plain spans
        // don't join it
        let g = Gather {
            copies: vec![StridedCopy {
                src: 0,
                dst: 0,
                width: 3,
                height: 4,
                src_pitch: 10,
                dst_pitch: 3,
            }],
            len: 12,
        };
        let spans = vec![sp(0, 7), sp(7, 40), sp(40, 45)];
        let plan = LoadPlan::new(
            spans,
            vec![None, Some(g), None],
            0,
            NonZeroUsize::new(7).unwrap(),
        );
        let file: Vec<u8> = (0..45).collect();
        let rows = (0..4).flat_map(|r| 7 + r * 10..10 + r * 10);
        let expected: Vec<u8> = (0..7).chain(rows).chain(40..45).collect();
        assert_eq!(land(&plan, &file, &[0, 7, 19], 24), expected);
    }

    fn check_plan(plan: &LoadPlan, chunk_size: usize) {
        let covered = |at: usize| plan.spans.iter().any(|s| s.start <= at && at < s.end);
        // chunks: sorted, disjoint, <= chunk_size, cut only at gaps and full chunks
        for w in plan.chunks.windows(2) {
            assert!(
                w[0].end <= w[1].start,
                "chunks overlap or are unsorted: {w:?}"
            );
            if w[0].end == w[1].start {
                assert_eq!(
                    w[0].len(),
                    chunk_size,
                    "undersized cut without a gap: {w:?}"
                );
            }
        }
        for c in plan.chunks.iter() {
            assert!(!c.is_empty() && c.len() <= chunk_size, "bad chunk {c:?}");
            assert!(c.clone().all(covered), "chunk {c:?} reads a gap");
        }
        // every span byte is in a chunk
        for s in plan.spans.iter() {
            assert!(
                (s.start..s.end).all(|at| plan.chunks.iter().any(|c| c.contains(&at))),
                "span {s:?} not covered"
            );
        }
        // span_chunks and chunk_spans == the chunks and spans that intersect
        let meet = |s: &Span, c: &std::ops::Range<usize>| s.start < c.end && c.start < s.end;
        for (i, s) in plan.spans.iter().enumerate() {
            for (j, c) in plan.chunks.iter().enumerate() {
                assert_eq!(
                    plan.span_chunks[i].contains(&j),
                    meet(s, c),
                    "span {s:?} vs chunk {c:?}"
                );
                if meet(s, c) {
                    assert!(
                        plan.chunk_spans[j].contains(&i),
                        "chunk {c:?} misses span {s:?}"
                    );
                }
            }
        }
        for (j, spans) in plan.chunk_spans.iter().enumerate() {
            for i in spans.clone() {
                let s = &plan.spans[i];
                assert!(
                    s.start == s.end || meet(s, &plan.chunks[j]),
                    "chunk {j} lists span {s:?}"
                );
            }
        }
    }

    #[test]
    fn test_chunks_cross_adjacent_spans() {
        let p = plan(vec![sp(0, 5), sp(5, 12), sp(12, 18)], 5);
        assert_eq!(
            p.chunks,
            vec![0..5, 5..10, 10..15, 15..18].into_boxed_slice()
        );
        assert_eq!(p.chunk_spans[2], 1..3);
    }

    #[test]
    fn test_chunk_len_exact_multiple_of_chunk_size() {
        let p = plan(vec![sp(0, 10)], 5);
        assert_eq!(p.chunks.len(), 2);
    }

    #[test]
    fn test_multi_chunk_span() {
        let p = plan(vec![sp(0, 42)], 5);
        assert_eq!(p.span_chunks[0], 0..9);
    }

    #[test]
    fn test_many_spans_in_single_chunk() {
        let p = plan(
            vec![
                sp(0, 1),
                sp(1, 2),
                sp(2, 3),
                sp(3, 4),
                sp(4, 5),
                sp(5, 6),
                sp(6, 7),
                sp(7, 8),
                sp(8, 11),
                sp(11, 17),
                sp(17, 20),
            ],
            20,
        );
        assert_eq!((p.chunks.len(), p.chunk_spans[0].clone()), (1, 0..11));
    }

    #[test]
    fn test_chunk_size_larger_than_data() {
        let p = plan(vec![sp(0, 5), sp(5, 8)], 4242);
        assert_eq!(p.chunks.len(), 1);
    }

    #[test]
    fn test_empty_spans() {
        let p = plan(vec![sp(0, 0), sp(0, 5), sp(5, 5), sp(5, 8), sp(8, 8)], 5);
        assert!(p.span_chunks[0].is_empty() && p.span_chunks[2].is_empty());
    }

    #[test]
    fn test_spans_with_a_gap() {
        // no chunk covers the gap
        let p = plan(vec![sp(0, 5), sp(12, 18)], 5);
        assert_eq!(p.chunks, vec![0..5, 12..17, 17..18].into_boxed_slice());
        assert_eq!(p.span_chunks[0], 0..1);
        assert_eq!(p.span_chunks[1], 1..3);
    }

    #[test]
    fn test_single_sub_span() {
        let p = plan(vec![sp(14, 17)], 5);
        assert_eq!((p.chunks.len(), p.chunks[0].clone()), (1, 14..17));
    }

    #[test]
    #[should_panic(expected = "sorted by start and disjoint")]
    fn test_unsorted_spans_are_rejected() {
        plan(vec![sp(12, 18), sp(0, 5)], 5);
    }

    #[test]
    #[should_panic(expected = "sorted by start and disjoint")]
    fn test_overlapping_spans_are_rejected() {
        plan(vec![sp(0, 5), sp(3, 8)], 5);
    }

    /// `len` pseudo-random bytes in a new file, and the bytes
    #[cfg(target_os = "linux")]
    fn random_file(name: &str, len: usize) -> (std::path::PathBuf, Arc<File>, Vec<u8>) {
        let mut rng = Rng(len as u64);
        let data: Vec<u8> = (0..len).map(|_| rng.below(256) as u8).collect();
        let path = std::env::temp_dir().join(format!("st_{name}_{}", std::process::id()));
        std::fs::write(&path, &data).unwrap();
        (path.clone(), Arc::new(File::open(path).unwrap()), data)
    }

    /// Drops `file` from the page cache
    #[cfg(target_os = "linux")]
    fn evict(file: &File) {
        use std::os::fd::AsRawFd;
        file.sync_all().unwrap();
        // SAFETY: `file` holds an open descriptor for the call's duration
        unsafe { libc::posix_fadvise(file.as_raw_fd(), 0, 0, libc::POSIX_FADV_DONTNEED) };
    }

    /// A zeroed buffer of `len` bytes, `misalign` bytes past a [`DIRECT_ALIGN`] boundary
    fn slab_buf(len: usize, misalign: usize) -> (Vec<u8>, Range<usize>) {
        let buf = vec![0u8; len + DIRECT_ALIGN + misalign];
        let at = buf.as_ptr().align_offset(DIRECT_ALIGN) + misalign;
        (buf, at..at + len)
    }

    #[cfg(target_os = "linux")]
    #[test]
    fn test_source_reads_cold_and_cached() {
        // odd offsets, and a tail whose aligned superset reaches past the end of a
        // file of 125 blocks and 123 bytes
        let (path, file, data) = random_file("source", 125 * DIRECT_ALIGN + 123);
        let cases = [
            (0, 1),
            (1, 4095),
            (4095, 2),
            (4096, 8192),
            (4099, data.len() - 4099),
            (data.len() - 7, 7),
        ];
        let (mut buf, at) = slab_buf(SLAB_SIZE, 0);
        for direct in [None, Some(true), Some(false)] {
            let source = Source::new(file.clone(), direct, true);
            for cold in [true, false] {
                for (offset, len) in cases {
                    if cold {
                        evict(&file);
                    }
                    let head = source
                        .read(&mut buf[at.clone()], offset as u64, len)
                        .unwrap();
                    assert_eq!(
                        &buf[at.start + head..][..len],
                        &data[offset..offset + len],
                        "{direct:?}, cold {cold}, {offset}+{len}"
                    );
                }
            }
        }
        std::fs::remove_file(path).unwrap();
    }

    #[cfg(target_os = "linux")]
    #[test]
    fn test_source_falls_back_when_direct_is_refused() {
        // a direct read into a misaligned buffer fails with EINVAL
        let (path, file, data) = random_file("einval", 3 * DIRECT_ALIGN);
        let (mut buf, at) = slab_buf(SLAB_SIZE, 1);
        let source = Source::new(file.clone(), Some(true), true);
        if let Some(direct) = &source.direct {
            evict(&file);
            let head = source.read(&mut buf[at.clone()], 5, 5000).unwrap();
            assert_eq!(&buf[at.start + head..][..5000], &data[5..5005]);
            assert!(!direct.ok.load(std::sync::atomic::Ordering::Relaxed));
        } else {
            eprintln!("skipped: no direct reads in {:?}", std::env::temp_dir());
        }
        std::fs::remove_file(path).unwrap();
    }

    #[cfg(target_os = "linux")]
    #[test]
    fn test_sliced_plan_reads_cold_as_cached() {
        // rows and columns of a 64 x 300 U16 tensor, then a whole tensor, in a data
        // section at an odd offset, read in chunks that start mid-block
        let (offset, tensor) = (4099, 64 * 600);
        let (path, file, data) = random_file("sliced", offset + tensor + 777);
        let intervals = [vec![3..60], vec![10..50, 200..260]];
        let indexers: Vec<Vec<TensorIndexer>> = intervals
            .iter()
            .map(|d| d.iter().map(|r| TensorIndexer::from(r.clone())).collect())
            .collect();
        let r = slice_layout(Dtype::U16, &[64, 300], &indexers).unwrap();
        let packed = r.gather.as_ref().unwrap().len;
        let plan = LoadPlan::new(
            vec![sp(r.read.start, r.read.end), sp(tensor, tensor + 777)],
            vec![r.gather, None],
            offset,
            NonZeroUsize::new(5000).unwrap(),
        );
        let (dsts, len) = ([0, packed], packed + 777);
        let mut expected = slice_bytes(&data[offset..], &[64, 300], 2, &intervals);
        expected.extend_from_slice(&data[offset + tensor..]);
        for (direct, cold) in [(Some(true), true), (Some(false), false)] {
            let source = Source::new(file.clone(), direct, true);
            if cold {
                evict(&file);
            }
            let got = land_read(&plan, &dsts, len, |slab, at, n| {
                source.read(slab, at, n).unwrap()
            });
            assert_eq!(got, expected, "{direct:?}");
        }
        std::fs::remove_file(path).unwrap();
    }
}
