//! CUDA prefetch engine: load byte spans of a safetensors data section into
//! device memory in the background and hand tensors out, once each, as they
//! land.
//!
//! Three nested units, computed once from the spans the consumer requests,
//! sorted by start and disjoint:
//!
//! ```text
//! tensors  |t0 |  t1  |     t2     |  t3  |       t4        | t5 | t6  |
//! spans    |S0#|..|S1#|S2#|...............|#######S3########|#S4#|#S5##|
//! allocs   |A0 |  |  A1   |               |       A2        |    A3    |
//! chunks   |c0 |  |  c1   |               |   c2   |   c3   |   c4   |c|
//! ```
//!
//! - spans [`LoadPlan::spans`]: a byte interval of the data section. The
//!   delivery unit: `take` hands out one span as a device view. A region's span
//!   runs from its first byte to its last: all of it is read, only the region is
//!   copied to the device.
//! - allocation file ranges ([`LoadPlan::allocation_file_ranges`]): the allocation unit, the byte
//!   interval each device buffer [`Allocation`] covers. Contiguous spans are
//!   concatenated into one allocation up to the first span end at or past
//!   [`MIN_ALLOCATION_SIZE`], or up to the end of the contiguous run if that
//!   comes first: far fewer allocations than one per tensor, without hoarding
//!   a whole file in a single buffer, which keeps device memory use balanced.
//! - chunks ([`LoadPlan::chunks`]): slices of at most [`CHUNK_SIZE`] of one allocation.
//!   The I/O unit: one `pread`, one slab, one `memcpy`, one event.
//!   `LoadPlan::span_alloc` and `span_chunks` record, per span, the
//!   `Allocation` it lives in and the chunks that carry its bytes.
//!
//! Data flow, per file:
//!
//! ```text
//! worker threads
//!   c = next_chunk.fetch_add(1)          next entry of LoadPlan::chunks
//!   lease = SlabPool::acquire()          a pinned host slab from the shared
//!                                        pool; waits for its previous copy
//!   pread(chunk c) -> lease              one read per chunk
//!   memcpy(2D)Async(lease -> device)     LoadPlan::chunk_copies[c] into its
//!                                        Allocation, on STREAMS[device]: one
//!                                        copy for plain spans, the kept rows of
//!                                        a region (see safetensors::slice::slice_region) as
//!                                        strided copies straight into its
//!                                        compact place
//!   chunk_completion_events[c] = event   recorded after the copy; 0 while
//!                                        the chunk is still pending
//! consumer
//!   take_tensor(s) / TensorIter::next    wait the events of s's chunks, then
//!     -> CudaBuffer                      a view into Arc<Allocation>, handed
//!                                        out once (AlreadyDelivered after)
//!   take of the last span of an Allocation  the sink drops its reference: the
//!                                        consumer's views now own the memory
//!                                        (partly taken allocations are held
//!                                        until close)
//!   drop(last view of an Allocation)     free_async on FREE_STREAMS[device],
//!                                        fenced on the load stream, the legacy
//!                                        default stream and the streams the
//!                                        views were handed out on
//! ```
//!
//! [`Loader`] owns the worker threads; [`LoaderInner`] (file, plan, sink, chunk
//! counter, first error) is shared with them and with [`TensorIter`]. Closing
//! signals the workers, joins them, then releases the sink: an outstanding
//! iterator yields `Err(Closed)` once and is exhausted. [`CUDA_ACTIVE_SINKS`]
//! counts live sinks per device so the mempool keeps its memory while any
//! file is loading and is trimmed when the last one releases.

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
    InvalidPlan(String),
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
            Self::InvalidPlan(msg) => write!(f, "invalid prefetch plan: {msg}"),
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

const MIN_ALLOCATION_SIZE: NonZeroUsize = NonZeroUsize::new(256 * 1024 * 1024).unwrap();
const CHUNK_SIZE: NonZeroUsize = NonZeroUsize::new(16 * 1024 * 1024).unwrap();
/// CUDA's default stream (null handle): where torch/CuPy enqueue work
/// unless the caller uses an explicit stream context.
const DEFAULT_STREAM: Stream = std::ptr::null_mut();

/// One [`LoadPlan::allocation_file_ranges`] entry as device memory, the only
/// malloc/free site for tensor memory. Freed when the sink clears its slot and
/// the last [`CudaBuffer`] view is dropped.
struct Allocation {
    api: &'static CudaApi,
    device: i32,
    stream: Stream,
    ptr: u64,
    /// streams consumers were on when a view was handed out (see
    /// [`CudaBuffer::consumed_on`]); the free waits for them as well
    consumer_streams: Mutex<Vec<u64>>,
}

unsafe impl Send for Allocation {}
unsafe impl Sync for Allocation {}

impl Allocation {
    fn new(
        api: &'static CudaApi,
        device: i32,
        stream: Stream,
        bytes: usize,
    ) -> Result<Arc<Self>, LoaderError> {
        let ptr = api.with_device(device, |_| api.malloc_async(bytes, stream))?;
        Ok(Arc::new(Self {
            api,
            device,
            stream,
            ptr,
            consumer_streams: Mutex::new(Vec::new()),
        }))
    }
}

/// Frees on the per-device free stream once it has waited on the load stream
/// (our pending copies), the legacy default stream and every stream a view was
/// handed out on (the consumer's current stream at that time). A consumer that
/// reads on yet another stream must sync before dropping its last view.
impl Drop for Allocation {
    fn drop(&mut self) {
        let free = free_stream(self.api, self.device).unwrap_or(self.stream);
        let consumers = std::mem::take(&mut *self.consumer_streams.lock().unwrap());
        let _: Result<(), CudaError> = self.api.with_device(self.device, |d| {
            let fenced = [self.stream, DEFAULT_STREAM]
                .into_iter()
                .chain(consumers.into_iter().map(|s| s as Stream));
            for stream in fenced {
                if let Ok(e) = d.event_create() {
                    let _ = self.api.event_record(e, stream);
                    let _ = self.api.stream_wait_event(free, e);
                    let _ = self.api.event_destroy(e);
                }
            }
            self.api.free_async(self.ptr, free)
        });
    }
}

/// A byte interval `start..end` of the file's data section
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Span {
    pub start: usize,
    pub end: usize,
}

/// Geometry of one load. File intervals are byte ranges of the file's data section; device intervals are byte
/// ranges of an allocation. A span's device bytes are its file bytes, or, for a region, the compact result of
/// its [`Gather`] (fewer bytes than were read).
struct LoadPlan {
    spans: Box<[Span]>,
    /// per span, the [`Allocation`] holding it; `None` for a zero-size span
    span_alloc: Box<[Option<usize>]>,
    /// per span, its bytes in its allocation: what `take` hands out
    span_device: Box<[Range<usize>]>,
    /// per span, the `chunks` that carry its bytes; empty for a zero-size span
    span_chunks: Box<[Range<usize>]>,
    /// in file data section byte interval each [`Allocation`] covers
    allocation_file_ranges: Box<[Range<usize>]>,
    /// per allocation, its size on the device
    allocation_sizes: Box<[usize]>,
    /// in file data section byte interval of each chunk: at most [`CHUNK_SIZE`],
    /// inside one allocation
    chunks: Box<[Range<usize>]>,
    /// per chunk, the [`Allocation`] it lands in
    chunk_alloc: Box<[usize]>,
    /// per chunk, the copies from its staging slab (offsets from the chunk's first byte) into its allocation
    /// (offsets from the allocation's first byte)
    chunk_copies: Box<[Box<[StridedCopy]>]>,
    in_file_offset: usize,
}

/// Where a copy's last byte ends, `None` on overflow.
fn copy_end(offset: usize, pitch: usize, width: usize, height: usize) -> Option<usize> {
    height
        .checked_sub(1)?
        .checked_mul(pitch)?
        .checked_add(offset)?
        .checked_add(width)
}

/// Whether a copy stays inside a `src_len`-byte source and a `dst_len`-byte destination without overflowing,
/// and never writes a row over the previous one.
fn copy_fits(c: &StridedCopy, src_len: usize, dst_len: usize) -> bool {
    c.width > 0
        && c.height > 0
        && (c.height == 1 || (c.src_pitch >= c.width && c.dst_pitch >= c.width))
        && copy_end(c.src, c.src_pitch, c.width, c.height).is_some_and(|e| e <= src_len)
        && copy_end(c.dst, c.dst_pitch, c.width, c.height).is_some_and(|e| e <= dst_len)
}

/// The part of `copy` (source offsets absolute in the file, destination offsets absolute in the allocation)
/// whose bytes lie in `chunk`, as copies relative to the chunk: whole rows as one strided copy, and the rows cut
/// by the chunk's edges (at most two) as single-row copies of their part inside it.
fn clip(copy: &StridedCopy, chunk: &Range<usize>, out: &mut Vec<StridedCopy>) {
    let row = |k: usize| copy.src + k * copy.src_pitch;
    let part = |k: usize, out: &mut Vec<StridedCopy>| {
        let (start, end) = (row(k), row(k) + copy.width);
        let (x0, x1) = (start.max(chunk.start), end.min(chunk.end));
        if x0 < x1 {
            out.push(StridedCopy {
                src: x0 - chunk.start,
                dst: copy.dst + k * copy.dst_pitch + (x0 - start),
                width: x1 - x0,
                height: 1,
                src_pitch: x1 - x0,
                dst_pitch: x1 - x0,
            });
        }
    };
    if copy.height == 1 {
        part(0, out);
        return;
    }
    // rows are disjoint and ordered (pitch >= width): the ones touching the chunk are a contiguous run
    let first = if chunk.start < copy.src + copy.width {
        0
    } else {
        (chunk.start - copy.src - copy.width) / copy.src_pitch + 1
    };
    if chunk.end <= copy.src {
        return;
    }
    let last = ((chunk.end - copy.src - 1) / copy.src_pitch).min(copy.height - 1);
    if first > last {
        return;
    }
    let whole = |k: usize| row(k) >= chunk.start && row(k) + copy.width <= chunk.end;
    let (mut lo, mut hi) = (first, last);
    if !whole(lo) {
        part(lo, out);
        lo += 1;
    }
    if lo <= hi && !whole(hi) {
        part(hi, out);
        if hi == 0 {
            return;
        }
        hi -= 1;
    }
    if lo <= hi {
        out.push(StridedCopy {
            src: row(lo) - chunk.start,
            dst: copy.dst + lo * copy.dst_pitch,
            width: copy.width,
            height: hi - lo + 1,
            src_pitch: copy.src_pitch,
            dst_pitch: copy.dst_pitch,
        });
    }
}

impl LoadPlan {
    /// `gathers` has one entry per span: `Some` for a span read for a region of it. Refuses (rather than panics
    /// on) any span or copy that would read or write outside its bounds: plans come from user input.
    fn new(
        spans: Vec<Span>,
        gathers: Vec<Option<Gather>>,
        in_file_offset: usize,
        chunk_size: NonZeroUsize,
        min_allocation_size: NonZeroUsize,
    ) -> Result<Self, LoaderError> {
        let invalid = |msg: String| Err(LoaderError::InvalidPlan(msg));
        if spans.len() != gathers.len() {
            return invalid("one gather entry per span".to_string());
        }
        if !(spans.iter().all(|s| s.start <= s.end)
            && spans.windows(2).all(|w| w[0].end <= w[1].start))
        {
            return invalid(
                "spans must be sorted by start and disjoint, each with start <= end".to_string(),
            );
        }
        for (s, g) in spans.iter().zip(&gathers) {
            if let Some(g) = g {
                // a region is a subset of the bytes read for it, and its copies fill it exactly
                let copied = g
                    .copies
                    .iter()
                    .try_fold(0usize, |n, c| n.checked_add(c.width.checked_mul(c.height)?));
                if g.len > s.end - s.start
                    || copied != Some(g.len)
                    || !g
                        .copies
                        .iter()
                        .all(|c| copy_fits(c, s.end - s.start, g.len))
                {
                    return invalid(format!(
                        "a copy of span {s:?} falls outside it or does not fill its region"
                    ));
                }
            }
        }
        let device_len = |i: usize| {
            gathers[i]
                .as_ref()
                .map_or(spans[i].end - spans[i].start, |g| g.len)
        };
        let chunk_size = chunk_size.get();
        let min_allocation_size = min_allocation_size.get();

        let mut allocation_file_ranges: Vec<Range<usize>> = Vec::new();
        let mut allocation_sizes: Vec<usize> = Vec::new();
        let mut alloc_spans: Vec<Range<usize>> = Vec::new(); // per allocation, the spans it holds
        let mut span_alloc: Vec<Option<usize>> = vec![None; spans.len()];
        let mut span_device: Vec<Range<usize>> = vec![0..0; spans.len()];
        let (mut alloc_start, mut prev_end, mut device_size, mut first_span) = (0, 0, 0usize, 0);
        let close = |ranges: &mut Vec<Range<usize>>,
                     sizes: &mut Vec<usize>,
                     held: &mut Vec<Range<usize>>,
                     file: Range<usize>,
                     size: usize,
                     span_range: Range<usize>| {
            ranges.push(file);
            sizes.push(size);
            held.push(span_range);
        };
        for (i, span) in spans.iter().enumerate() {
            if span.start == span.end {
                continue; // no bytes: no range, no chunks
            }
            // gap between spans
            if span.start != prev_end {
                if prev_end > alloc_start {
                    close(
                        &mut allocation_file_ranges,
                        &mut allocation_sizes,
                        &mut alloc_spans,
                        alloc_start..prev_end,
                        device_size,
                        first_span..i,
                    );
                }
                alloc_start = span.start;
                device_size = 0;
                first_span = i;
            }
            span_alloc[i] = Some(allocation_file_ranges.len());
            // bounded by the file bytes read (a region never exceeds its span), so this cannot overflow
            let Some(end) = device_size.checked_add(device_len(i)) else {
                return invalid("device sizes overflow".to_string());
            };
            span_device[i] = device_size..end;
            device_size = end;
            prev_end = span.end;
            if prev_end - alloc_start >= min_allocation_size {
                close(
                    &mut allocation_file_ranges,
                    &mut allocation_sizes,
                    &mut alloc_spans,
                    alloc_start..prev_end,
                    device_size,
                    first_span..i + 1,
                );
                alloc_start = prev_end;
                device_size = 0;
                first_span = i + 1;
            }
        }
        if prev_end > alloc_start {
            close(
                &mut allocation_file_ranges,
                &mut allocation_sizes,
                &mut alloc_spans,
                alloc_start..prev_end,
                device_size,
                first_span..spans.len(),
            );
        }

        let mut chunks: Vec<Range<usize>> = Vec::new();
        let mut chunk_alloc: Vec<usize> = Vec::new();
        let mut chunk_copies: Vec<Box<[StridedCopy]>> = Vec::new();
        let mut first_chunk = Vec::with_capacity(allocation_file_ranges.len());
        for (alloc_idx, alloc_file_range) in allocation_file_ranges.iter().enumerate() {
            first_chunk.push(chunks.len());
            let mut start = alloc_file_range.start;
            while start < alloc_file_range.end {
                let end = (start + chunk_size).min(alloc_file_range.end);
                let chunk = start..end;
                let mut copies = Vec::new();
                for i in alloc_spans[alloc_idx].clone() {
                    let s = &spans[i];
                    if s.start >= chunk.end || s.end <= chunk.start {
                        continue;
                    }
                    let dev = span_device[i].start;
                    match &gathers[i] {
                        None => clip(
                            &StridedCopy {
                                src: s.start,
                                dst: dev,
                                width: s.end - s.start,
                                height: 1,
                                src_pitch: s.end - s.start,
                                dst_pitch: s.end - s.start,
                            },
                            &chunk,
                            &mut copies,
                        ),
                        Some(g) => {
                            for c in &g.copies {
                                let absolute = StridedCopy {
                                    src: s.start + c.src,
                                    dst: dev + c.dst,
                                    ..*c
                                };
                                clip(&absolute, &chunk, &mut copies);
                            }
                        }
                    }
                }
                if !copies
                    .iter()
                    .all(|c| copy_fits(c, chunk.len(), allocation_sizes[alloc_idx]))
                {
                    return invalid(format!(
                        "a copy of chunk {chunk:?} falls outside it or its allocation"
                    ));
                }
                chunks.push(chunk);
                chunk_alloc.push(alloc_idx);
                chunk_copies.push(copies.into_boxed_slice());
                start = end;
            }
        }

        if span_alloc
            .iter()
            .zip(&span_device)
            .any(|(a, d)| a.is_some_and(|a| d.end > allocation_sizes[a]))
        {
            return invalid("a span's device bytes fall outside its allocation".to_string());
        }

        let span_chunks = spans
            .iter()
            .zip(&span_alloc)
            .map(|(span, &alloc_idx)| match alloc_idx {
                Some(a) => {
                    let base = allocation_file_ranges[a].start;
                    let first = first_chunk[a];
                    first + (span.start - base) / chunk_size
                        ..first + (span.end - 1 - base) / chunk_size + 1
                }
                None => 0..0,
            })
            .collect();

        Ok(Self {
            spans: spans.into_boxed_slice(),
            span_alloc: span_alloc.into_boxed_slice(),
            span_device: span_device.into_boxed_slice(),
            span_chunks,
            allocation_file_ranges: allocation_file_ranges.into_boxed_slice(),
            allocation_sizes: allocation_sizes.into_boxed_slice(),
            chunks: chunks.into_boxed_slice(),
            chunk_alloc: chunk_alloc.into_boxed_slice(),
            chunk_copies: chunk_copies.into_boxed_slice(),
            in_file_offset,
        })
    }
}

pub struct CudaBuffer {
    _alloc: Option<Arc<Allocation>>,
    device: i32,
    ptr: u64,
    len: usize,
}

impl CudaBuffer {
    pub(crate) fn ptr(&self) -> u64 {
        self.ptr
    }

    pub(crate) fn len(&self) -> usize {
        self.len
    }

    pub(crate) fn device(&self) -> i32 {
        self.device
    }

    /// Records the stream the consumer is about to use this view on (its
    /// current stream at hand-out), so the allocation's free waits for that
    /// stream too. `0` is the legacy default stream, always fenced.
    pub(crate) fn consumed_on(&self, stream: u64) {
        if stream == 0 {
            return;
        }
        if let Some(alloc) = &self._alloc {
            let mut streams = alloc.consumer_streams.lock().unwrap();
            if !streams.contains(&stream) {
                streams.push(stream);
            }
        }
    }
}

pub enum DeviceBuffer {
    Cuda(CudaBuffer),
}

enum Sink {
    Cuda(CudaSink),
}

impl Sink {
    /// `read(buffer, offset)` fills the chunk's slab from the source
    fn load_chunk(
        &self,
        chunk_idx: usize,
        read: impl FnOnce(&mut [u8], u64) -> std::io::Result<()>,
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

    fn take(&self, idx: usize) -> Result<DeviceBuffer, LoaderError> {
        match self {
            Self::Cuda(sink) => Ok(DeviceBuffer::Cuda(sink.take(idx)?)),
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

    fn release(&self) {
        match self {
            Self::Cuda(sink) => sink.close(),
        }
    }

    fn fully_scheduled(&self, span: usize) -> bool {
        match self {
            Self::Cuda(sink) => sink.fully_scheduled(span),
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

    fn mut_slice(&mut self, len: usize) -> &mut [u8] {
        assert!(len <= self.pool.slab_size, "chunk does not fit in slab");
        unsafe { std::slice::from_raw_parts_mut(self.ptr(), len) }
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

/// Used to track live [`CudaSink`]s
/// We configure cuda's memory pool to retain allocated memory, meaning:
/// - across `safe_open` calls, allocated memory blocks can be reused
/// - we need to track when all active sinks are done to be able to release that retained memory
///
/// Once the active counter for a given device reaches 0, we call `cudaMemPoolTrimTo` releasing the
/// unused additional retained memory
static CUDA_ACTIVE_SINKS: [AtomicUsize; MAX_DEVICES] = [const { AtomicUsize::new(0) }; MAX_DEVICES];

struct CudaSink {
    api: &'static CudaApi,
    device: i32,
    stream: Stream,
    pool: &'static SlabPool,
    plan: Arc<LoadPlan>,
    allocations: Box<[Mutex<Option<Arc<Allocation>>>]>,
    /// spans not yet taken; at zero the sink drops its own
    /// reference so the memory lives exactly as long as the consumer's views
    in_alloc_pending: Box<[AtomicUsize]>,
    delivered: Box<[AtomicBool]>,
    chunk_completion_events: Box<[AtomicU64]>,
    closed: AtomicBool,
    released: AtomicBool,
}

unsafe impl Send for CudaSink {}
unsafe impl Sync for CudaSink {}

impl CudaSink {
    fn new(
        api: &'static CudaApi,
        pool: &'static SlabPool,
        plan: Arc<LoadPlan>,
        device: i32,
    ) -> Result<Self, LoaderError> {
        let stream = device_stream(api, device)?;
        CUDA_ACTIVE_SINKS[device as usize].fetch_add(1, Ordering::AcqRel);
        let _ = api.pool_set_release_threshold(device, u64::MAX);

        Ok(Self {
            api,
            device,
            stream,
            pool,
            allocations: (0..plan.allocation_file_ranges.len())
                .map(|_| Mutex::new(None))
                .collect(),
            in_alloc_pending: {
                let mut n = vec![0usize; plan.allocation_file_ranges.len()];
                for &a in plan.span_alloc.iter().flatten() {
                    n[a] += 1;
                }
                n.into_iter().map(AtomicUsize::new).collect()
            },
            delivered: (0..plan.spans.len())
                .map(|_| AtomicBool::new(false))
                .collect(),
            chunk_completion_events: (0..plan.chunks.len()).map(|_| AtomicU64::new(0)).collect(),
            closed: AtomicBool::new(false),
            released: AtomicBool::new(false),
            plan,
        })
    }

    fn allocation_ptr(&self, alloc_idx: usize) -> Result<u64, LoaderError> {
        let mut slot = self.allocations[alloc_idx].lock().unwrap();
        if let Some(a) = &*slot {
            return Ok(a.ptr);
        }
        if self.closed.load(Ordering::Acquire) {
            return Err(LoaderError::Closed);
        }
        let bytes = self.plan.allocation_sizes[alloc_idx];
        let alloc = Allocation::new(self.api, self.device, self.stream, bytes)?;
        let ptr = alloc.ptr;
        *slot = Some(alloc);
        Ok(ptr)
    }

    fn load_chunk(
        &self,
        chunk_idx: usize,
        read: impl FnOnce(&mut [u8], u64) -> std::io::Result<()>,
    ) -> Result<(), LoaderError> {
        let chunk = self.plan.chunks[chunk_idx].clone();
        let alloc_idx = self.plan.chunk_alloc[chunk_idx];
        let mut lease = self.pool.acquire(self.api)?;
        read(
            lease.mut_slice(chunk.len()),
            (self.plan.in_file_offset + chunk.start) as u64,
        )?;

        let dst = self.allocation_ptr(alloc_idx)?;
        let e = self.api.with_device(self.device, |d| {
            for c in self.plan.chunk_copies[chunk_idx].iter() {
                // `LoadPlan::new` checked every copy stays inside the chunk and the allocation
                let src = unsafe { lease.ptr().add(c.src) };
                if c.height == 1 {
                    self.api
                        .memcpy_h2d_async(dst + c.dst as u64, src, c.width, self.stream)?;
                } else {
                    self.api.memcpy2d_h2d_async(
                        dst + c.dst as u64,
                        c.dst_pitch,
                        src,
                        c.src_pitch,
                        c.width,
                        c.height,
                        self.stream,
                    )?;
                }
            }
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

    fn take(&self, idx: usize) -> Result<CudaBuffer, LoaderError> {
        if self.delivered[idx].load(Ordering::Acquire) {
            return Err(LoaderError::AlreadyDelivered);
        }
        let device_range = self.plan.span_device[idx].clone();
        let mut buffer = CudaBuffer {
            _alloc: None,
            device: self.device,
            ptr: 0,
            len: device_range.len(),
        };
        if let Some(alloc_idx) = self.plan.span_alloc[idx] {
            let Some(alloc) = self.allocations[alloc_idx].lock().unwrap().clone() else {
                return Err(if self.delivered[idx].load(Ordering::Acquire) {
                    LoaderError::AlreadyDelivered
                } else {
                    LoaderError::Closed
                });
            };
            buffer.ptr = alloc.ptr + device_range.start as u64;
            buffer._alloc = Some(alloc);
        }
        if self.delivered[idx].swap(true, Ordering::AcqRel) {
            return Err(LoaderError::AlreadyDelivered);
        }
        if let Some(alloc_idx) = self.plan.span_alloc[idx] {
            if self.in_alloc_pending[alloc_idx].fetch_sub(1, Ordering::AcqRel) == 1 {
                *self.allocations[alloc_idx].lock().unwrap() = None;
            }
        }
        Ok(buffer)
    }

    fn close(&self) {
        self.closed.store(true, Ordering::Release);
        if self.released.swap(true, Ordering::AcqRel) {
            return;
        }
        for slot in self.allocations.iter() {
            *slot.lock().unwrap() = None;
        }
        if CUDA_ACTIVE_SINKS[self.device as usize].fetch_sub(1, Ordering::AcqRel) == 1 {
            if let Ok(free) = free_stream(self.api, self.device) {
                // NOTE: `let _ =` is intentional, releasing retained memory is best effort
                let _: Result<(), CudaError> = self.api.with_device(self.device, |d| {
                    let e = d.event_create()?;
                    let _ = self.api.event_record(e, free);
                    let _ = self.api.event_sync(e);
                    self.api.event_destroy(e)
                });
            }
            let _ = self.api.pool_set_release_threshold(self.device, 0);
            let _ = self.api.pool_trim(self.device);
            // a sink opened concurrently may have lost its retention to the reset above
            if CUDA_ACTIVE_SINKS[self.device as usize].load(Ordering::Acquire) > 0 {
                let _ = self.api.pool_set_release_threshold(self.device, u64::MAX);
            }
        }
    }

    fn signal_stop(&self) {
        self.closed.store(true, Ordering::Release);
    }

    fn fully_scheduled(&self, span: usize) -> bool {
        self.plan.span_chunks[span]
            .clone()
            .all(|c| self.chunk_completion_events[c].load(Ordering::Acquire) != 0)
    }
}

impl Drop for CudaSink {
    fn drop(&mut self) {
        self.close();
        for e in self.chunk_completion_events.iter() {
            let e = e.swap(0, Ordering::AcqRel);
            if e != 0 {
                let _ = self.api.event_destroy(e as Event);
            }
        }
    }
}

/// This is a generous limit on the maximum number of devices that could be connected to a single host
const MAX_DEVICES: usize = 64;
static STREAMS: [AtomicU64; MAX_DEVICES] = [const { AtomicU64::new(0) }; MAX_DEVICES];
static FREE_STREAMS: [AtomicU64; MAX_DEVICES] = [const { AtomicU64::new(0) }; MAX_DEVICES];

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

fn free_stream(api: &CudaApi, device: i32) -> Result<Stream, CudaError> {
    get_stream(&FREE_STREAMS, api, device)
}

fn device_stream(api: &CudaApi, device: i32) -> Result<Stream, CudaError> {
    get_stream(&STREAMS, api, device)
}

struct LoaderInner {
    file: Arc<File>,
    plan: Arc<LoadPlan>,
    sink: Sink,
    next_chunk: AtomicUsize,
    error: OnceLock<LoaderError>,
}

impl LoaderInner {
    fn take_tensor(&self, idx: usize) -> Result<DeviceBuffer, LoaderError> {
        self.sink
            .wait_ready(idx)
            .and_then(|()| self.sink.take(idx))
            .map_err(|e| match (e, self.error.get()) {
                (LoaderError::Closed, Some(err)) => LoaderError::WorkerFailed(err.to_string()),
                (e, _) => e,
            })
    }
}

pub struct Loader {
    inner: Arc<LoaderInner>,
    workers: Box<[JoinHandle<()>]>,
}

impl Loader {
    /// Starts loading `spans` onto `device`. `gathers` has one entry per span: `Some` when the consumer asked for
    /// a region of the span rather than the span itself (see [`safetensors::slice::slice_region`]).
    pub fn load(
        file: Arc<File>,
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
        #[cfg(target_os = "linux")]
        {
            use std::os::fd::AsRawFd;
            // Chunks are read out of order by several threads and a plan may skip
            // regions: kernel readahead would pull in bytes nobody asked for.
            // Best effort, the hint failing only costs throughput.
            // SAFETY: `file` holds an open descriptor for the call's duration.
            let _ = unsafe { libc::posix_fadvise(file.as_raw_fd(), 0, 0, libc::POSIX_FADV_RANDOM) };
        }
        let plan = Arc::new(LoadPlan::new(
            spans,
            gathers,
            in_file_offset,
            CHUNK_SIZE,
            MIN_ALLOCATION_SIZE,
        )?);
        // avoid spawning too many threads when not needed, up to chunks.len()
        let threads = threads.clamp(1, plan.chunks.len().max(1));
        let pool = cuda_api.with_device(device, |_| pool(cuda_api))?;
        let inner = Arc::new(LoaderInner {
            file,
            error: OnceLock::new(),
            plan: plan.clone(),
            sink: Sink::Cuda(CudaSink::new(cuda_api, pool, plan, device)?),
            next_chunk: AtomicUsize::new(0),
        });
        let workers = (0..threads)
            .map(|_| {
                std::thread::spawn({
                    let inner = inner.clone();
                    move || {
                        let _ = cuda_api.set_device(device);
                        worker(inner)
                    }
                })
            })
            .collect();
        Ok(Self { inner, workers })
    }

    /// `idx` indexes [`Loader::spans`]
    pub fn take_tensor(&self, idx: usize) -> Result<DeviceBuffer, LoaderError> {
        self.inner.take_tensor(idx)
    }

    pub fn close(&mut self) {
        self.inner.sink.signal_stop();
        for h in std::mem::take(&mut self.workers) {
            let _ = h.join();
        }
        self.inner.sink.release();
    }

    pub fn iter(&self) -> TensorIter {
        TensorIter {
            inner: self.inner.clone(),
            pending: (0..self.inner.plan.spans.len()).collect(),
        }
    }
}

impl Drop for Loader {
    fn drop(&mut self) {
        self.close();
    }
}

pub struct TensorIter {
    inner: Arc<LoaderInner>,
    pending: VecDeque<usize>,
}

impl Iterator for TensorIter {
    type Item = Result<(usize, DeviceBuffer), LoaderError>;

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
                Ok(buffer) => {
                    self.pending.remove(pos);
                    return Some(Ok((idx, buffer)));
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
            let _ = POOL.set(SlabPool::new(cuda_api, 32, CHUNK_SIZE.get())?);
        }
    }
    Ok(POOL.get().unwrap())
}

fn worker(loader: Arc<LoaderInner>) {
    loop {
        if loader.sink.stopped() {
            return;
        }
        let chunk_idx = loader.next_chunk.fetch_add(1, Ordering::Relaxed);
        if chunk_idx >= loader.plan.chunks.len() {
            return;
        }
        let res = loader.sink.load_chunk(chunk_idx, |buffer, offset| {
            loader.file.read_exact_at(buffer, offset)
        });
        if let Err(err) = res {
            if !matches!(err, LoaderError::Closed) {
                let _ = loader.error.set(err);
            }
            loader.sink.signal_stop();
            return;
        }
    }
}

#[cfg(test)]
#[allow(clippy::single_range_in_vec_init)] // one interval per dimension
mod tests {
    use std::{collections::HashSet, num::NonZeroUsize, ops::Range};

    use super::{LoadPlan, Span};
    use safetensors::slice::{slice_region, Gather, StridedCopy, TensorIndexer};
    use safetensors::Dtype;

    /// Runs a plan's chunk copies on host buffers, as the device does: each chunk is read from `file` and its
    /// copies land in its allocation. Returns every span's device bytes.
    fn run(plan: &LoadPlan, file: &[u8]) -> Vec<Vec<u8>> {
        let mut allocs: Vec<Vec<u8>> = plan
            .allocation_sizes
            .iter()
            .map(|&n| vec![0xAA; n])
            .collect();
        for (c, chunk) in plan.chunks.iter().enumerate() {
            let slab = &file[chunk.clone()];
            let a = &mut allocs[plan.chunk_alloc[c]];
            for cp in plan.chunk_copies[c].iter() {
                for r in 0..cp.height {
                    let (src, dst) = (cp.src + r * cp.src_pitch, cp.dst + r * cp.dst_pitch);
                    a[dst..dst + cp.width].copy_from_slice(&slab[src..src + cp.width]);
                }
            }
        }
        (0..plan.spans.len())
            .map(|i| match plan.span_alloc[i] {
                Some(a) => allocs[a][plan.span_device[i].clone()].to_vec(),
                None => vec![],
            })
            .collect()
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

    /// The region's bytes by brute force over element indices, from a tensor starting at byte 0 of `bytes`.
    fn region_bytes(
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
    fn test_random_regions_land_exactly() {
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

            let (mut spans, mut gathers, mut expected) = (Vec::new(), Vec::new(), Vec::new());
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
                let r = slice_region(dtype, shape, &indexers).unwrap();
                let nbytes = shape.iter().product::<usize>() * elem;
                expected.push(region_bytes(
                    &file[*start..*start + nbytes],
                    shape,
                    *elem,
                    &intervals,
                ));
                spans.push(Span {
                    start: start + r.read.start,
                    end: start + r.read.end,
                });
                gathers.push(r.gather);
            }
            let chunk = 1 + rng.below(48);
            let min_alloc = 1 + rng.below(200);
            let plan = LoadPlan::new(
                spans,
                gathers,
                0,
                NonZeroUsize::new(chunk).unwrap(),
                NonZeroUsize::new(min_alloc).unwrap(),
            )
            .unwrap();
            assert_eq!(
                run(&plan, &file),
                expected,
                "trial {trial}, chunk {chunk}, min_alloc {min_alloc}"
            );
        }
    }

    #[test]
    fn test_garbage_plans_never_panic_and_stay_in_bounds() {
        // arbitrary spans and copies, as a hostile caller could build: `LoadPlan::new` must refuse them or accept
        // a plan whose every copy stays inside its chunk and allocation (`run` indexes host buffers, so any
        // out-of-bounds copy panics here)
        let mut rng = Rng(11);
        let value = |rng: &mut Rng| match rng.below(8) {
            0 => usize::MAX - rng.below(4),
            1 => usize::MAX / 2 + rng.below(4),
            _ => rng.below(80),
        };
        let mut accepted = 0;
        for _ in 0..20_000 {
            let n = rng.below(4);
            let mut spans = Vec::new();
            let mut gathers = Vec::new();
            for _ in 0..n {
                let (a, b) = (rng.below(256), rng.below(256));
                spans.push(sp(a.min(b), a.max(b)));
                gathers.push(match rng.below(3) {
                    0 => None,
                    _ => Some(Gather {
                        copies: (0..rng.below(4))
                            .map(|_| StridedCopy {
                                src: value(&mut rng),
                                dst: value(&mut rng),
                                width: value(&mut rng),
                                height: value(&mut rng),
                                src_pitch: value(&mut rng),
                                dst_pitch: value(&mut rng),
                            })
                            .collect(),
                        len: value(&mut rng),
                    }),
                });
            }
            let chunk = NonZeroUsize::new(1 + rng.below(64)).unwrap();
            let min_alloc = NonZeroUsize::new(1 + rng.below(128)).unwrap();
            if let Ok(plan) = LoadPlan::new(spans, gathers, 0, chunk, min_alloc) {
                accepted += 1;
                let file = vec![0u8; 256];
                let _ = run(&plan, &file);
            }
        }
        assert!(
            accepted > 0,
            "no garbage plan was valid: the generator never exercises acceptance"
        );
    }

    #[test]
    fn test_invalid_plans_are_refused() {
        let nz = |n| NonZeroUsize::new(n).unwrap();
        let new = |spans: Vec<Span>, gathers: Vec<Option<Gather>>| {
            LoadPlan::new(spans, gathers, 0, nz(8), nz(64))
        };
        // overlapping and unsorted spans
        assert!(new(vec![sp(0, 8), sp(4, 12)], vec![None, None]).is_err());
        assert!(new(vec![sp(8, 12), sp(0, 4)], vec![None, None]).is_err());
        assert!(new(vec![sp(0, 4)], vec![]).is_err());
        let copy = |src, dst, width, height, src_pitch, dst_pitch| StridedCopy {
            src,
            dst,
            width,
            height,
            src_pitch,
            dst_pitch,
        };
        let one = |c: StridedCopy, len| {
            vec![Some(Gather {
                copies: vec![c],
                len,
            })]
        };
        // reading past the span, writing past the region, overlapping rows, empty and overflowing copies
        assert!(new(vec![sp(0, 16)], one(copy(0, 0, 4, 5, 4, 4), 20)).is_err());
        assert!(new(vec![sp(0, 16)], one(copy(0, 0, 4, 4, 4, 4), 12)).is_err());
        assert!(new(vec![sp(0, 16)], one(copy(0, 0, 4, 2, 2, 4), 8)).is_err());
        assert!(new(vec![sp(0, 16)], one(copy(0, 0, 0, 1, 0, 0), 4)).is_err());
        assert!(new(vec![sp(0, 16)], one(copy(usize::MAX, 0, 4, 2, 4, 4), 8)).is_err());
        assert!(new(
            vec![sp(0, 16)],
            one(copy(0, 0, 4, usize::MAX, usize::MAX, 4), 8)
        )
        .is_err());
        // a region larger than the bytes read for it, or not filled by its copies
        assert!(new(vec![sp(0, 16)], one(copy(0, 0, 4, 4, 4, 4), 32)).is_err());
        assert!(new(vec![sp(0, 16)], one(copy(0, 0, 4, 2, 4, 4), 16)).is_err());
        assert!(new(vec![sp(0, 16)], one(copy(0, 0, 4, 4, 4, 4), usize::MAX)).is_err());
        // and a well-formed one goes through
        assert!(new(vec![sp(0, 16)], one(copy(0, 0, 4, 4, 4, 4), 16)).is_ok());
    }

    fn sp(start: usize, end: usize) -> Span {
        Span { start, end }
    }

    fn plan(spans: Vec<Span>, chunk_size: usize, min_allocation_size: usize) -> LoadPlan {
        let gathers = vec![None; spans.len()];
        let plan = LoadPlan::new(
            spans,
            gathers,
            0,
            NonZeroUsize::new(chunk_size).unwrap(),
            NonZeroUsize::new(min_allocation_size).unwrap(),
        )
        .unwrap();
        check_plan(&plan, chunk_size, min_allocation_size);
        plan
    }

    fn check_plan(plan: &LoadPlan, chunk_size: usize, min_allocation_size: usize) {
        let ranges = &plan.allocation_file_ranges;
        assert_eq!(plan.spans.len(), plan.span_alloc.len());
        assert_eq!(plan.spans.len(), plan.span_chunks.len());
        // spans: sorted, disjoint, each non-empty one inside its allocation
        for w in plan.spans.windows(2) {
            assert!(
                w[0].end <= w[1].start,
                "spans overlap or are unsorted: {w:?}"
            );
        }
        for (i, s) in plan.spans.iter().enumerate() {
            let (alloc, chunks) = (plan.span_alloc[i], &plan.span_chunks[i]);
            if s.start == s.end {
                assert!(
                    alloc.is_none() && chunks.is_empty(),
                    "empty span placed: {s:?}"
                );
                continue;
            }
            let r = &ranges[alloc.expect("non-empty span has an allocation")];
            assert!(
                r.start <= s.start && s.end <= r.end,
                "span {s:?} crosses {r:?}"
            );
        }
        // allocations: sorted, disjoint, bounded by span boundaries, cut at gaps and
        // at the first span end >= min_allocation_size and nowhere else
        let starts: HashSet<usize> = plan.spans.iter().map(|s| s.start).collect();
        let ends: HashSet<usize> = plan.spans.iter().map(|s| s.end).collect();
        for r in ranges.iter() {
            assert!(r.end > r.start, "empty range {r:?}");
            assert!(
                starts.contains(&r.start) && ends.contains(&r.end),
                "{r:?} not span-aligned"
            );
            for s in plan.spans.iter() {
                assert!(
                    !(s.end > r.start && s.end < r.end && s.end - r.start >= min_allocation_size),
                    "range {r:?} should have been cut at span end {}",
                    s.end
                );
            }
        }
        for w in ranges.windows(2) {
            assert!(
                w[0].end <= w[1].start,
                "ranges overlap or are unsorted: {w:?}"
            );
            if w[0].end == w[1].start {
                assert!(
                    w[0].len() >= min_allocation_size,
                    "undersized cut without a gap: {w:?}"
                );
            }
        }
        // chunks: tile every allocation exactly, <= chunk_size, never crossing one
        let mut c = 0;
        for (i, r) in ranges.iter().enumerate() {
            let mut cursor = r.start;
            while cursor < r.end {
                let chunk = &plan.chunks[c];
                assert_eq!(plan.chunk_alloc[c], i);
                assert_eq!(
                    chunk.start, cursor,
                    "chunk {c} does not continue allocation {i}"
                );
                assert!(chunk.len() <= chunk_size && chunk.end <= r.end);
                cursor = chunk.end;
                c += 1;
            }
        }
        assert_eq!(c, plan.chunks.len(), "chunks outside every allocation");
        assert_eq!(plan.chunks.len(), plan.chunk_alloc.len());
        // a span's chunks == the chunks it intersects
        for (s, chunks) in plan.spans.iter().zip(plan.span_chunks.iter()) {
            for (c, chunk) in plan.chunks.iter().enumerate() {
                assert_eq!(
                    chunks.contains(&c),
                    s.start != s.end && s.start < chunk.end && chunk.start < s.end,
                    "span {s:?} vs chunk {c} {chunk:?}"
                );
            }
        }
    }

    #[test]
    fn test_plan_allocations() {
        let p = plan(vec![sp(0, 5), sp(5, 12), sp(12, 18)], 5, 12);
        assert_eq!(
            p.allocation_file_ranges,
            vec![0..12, 12..18].into_boxed_slice()
        );
    }

    #[test]
    fn test_allocations_oversized_span() {
        // floor 8: cut at the first span end >= 8 bytes in; the 30-byte span
        // forces a large allocation; the 3-byte tail stays undersized
        let p = plan(vec![sp(0, 5), sp(5, 12), sp(12, 42), sp(42, 45)], 5, 8);
        assert_eq!(
            p.allocation_file_ranges,
            vec![0..12, 12..42, 42..45].into_boxed_slice()
        );
    }

    #[test]
    fn test_chunk_len_exact_multiple_of_chunk_size() {
        let p = plan(vec![sp(0, 10)], 5, 12);
        assert_eq!(p.chunks.len(), 2);
    }

    #[test]
    fn test_multi_chunk_span() {
        let p = plan(vec![sp(0, 42)], 5, 12);
        assert_eq!(p.span_chunks[0], 0..9);
    }

    #[test]
    fn test_many_spans_in_single_chunk() {
        plan(
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
            20,
        );
    }

    #[test]
    fn test_chunk_size_larger_than_data() {
        let p = plan(vec![sp(0, 5), sp(5, 8)], 4242, 4242);
        assert_eq!(p.chunks.len(), 1);
    }

    #[test]
    fn test_empty_spans() {
        plan(vec![sp(0, 0), sp(0, 5), sp(5, 5), sp(5, 8), sp(8, 8)], 5, 5);
    }

    #[test]
    fn test_spans_with_a_gap() {
        // two allocations, no chunk covers the gap
        let p = plan(vec![sp(0, 5), sp(12, 18)], 5, 12);
        assert_eq!(
            p.allocation_file_ranges,
            vec![0..5, 12..18].into_boxed_slice()
        );
        assert_eq!(p.chunks.len(), 3); // 0..5 | 12..17, 17..18
        assert_eq!(p.span_chunks[0], 0..1);
        assert_eq!(p.span_chunks[1], 1..3);
    }

    #[test]
    fn test_single_sub_span() {
        // the only allocation is the requested interval
        let p = plan(vec![sp(14, 17)], 5, 12);
        assert_eq!(p.allocation_file_ranges.len(), 1);
        assert_eq!(p.allocation_file_ranges[0], 14..17);
        assert_eq!(p.chunks.len(), 1);
        assert_eq!(p.span_alloc[0], Some(0));
    }

    #[test]
    fn test_adjacent_spans_share_a_range() {
        let p = plan(vec![sp(3, 5), sp(5, 7)], 5, 12);
        assert_eq!(p.allocation_file_ranges.len(), 1);
        assert_eq!(p.allocation_file_ranges[0], 3..7);
        assert_eq!(p.span_alloc[0], p.span_alloc[1]);
        assert_eq!(p.chunks.len(), 1);
    }

    #[test]
    #[should_panic(expected = "sorted by start and disjoint")]
    fn test_unsorted_spans_are_rejected() {
        plan(vec![sp(12, 18), sp(0, 5)], 5, 12);
    }

    #[test]
    #[should_panic(expected = "sorted by start and disjoint")]
    fn test_overlapping_spans_are_rejected() {
        plan(vec![sp(0, 5), sp(3, 8)], 5, 12);
    }

    #[test]
    fn test_size_cut_then_gap() {
        // first run reaches the floor at 12 and is cut there; the uncovered bytes
        // 12..20 open a new allocation; the tail run is undersized but closed by end
        let p = plan(vec![sp(0, 5), sp(5, 12), sp(20, 23), sp(23, 27)], 5, 12);
        assert_eq!(
            p.allocation_file_ranges,
            vec![0..12, 20..27].into_boxed_slice()
        );
    }
}
