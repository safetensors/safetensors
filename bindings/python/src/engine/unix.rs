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
//!   its destination. The delivery unit: `take` hands out the span's owner.
//! - chunks ([`LoadPlan::chunks`]): at most [`CHUNK_SIZE`] of adjacent spans,
//!   never across a gap. The I/O unit: one `pread`, one slab, one event, one
//!   `memcpy` per span it carries. `span_chunks` and `chunk_spans` map one to
//!   the other.
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
//!   memcpyAsync(lease -> destinations)   one copy per span in the chunk,
//!                                        on STREAMS[device]
//!   chunk_completion_events[c] = event   recorded after the copies; 0 while
//!                                        the chunk is still pending
//! consumer
//!   take_tensor(s) / TensorIter::next    wait the events of s's chunks, then
//!     -> owner of s's destination        handed out once (AlreadyDelivered after)
//! ```
//!
//! [`Loader`] owns the worker threads; [`LoaderInner`] (file, plan, sink, chunk
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
    /// per span, the `chunks` that carry its bytes; empty for a zero-size span
    span_chunks: Box<[Range<usize>]>,
    /// data section byte interval of each chunk: at most [`CHUNK_SIZE`] of
    /// adjacent spans, never across a gap
    chunks: Box<[Range<usize>]>,
    /// per chunk, the spans it carries bytes of
    chunk_spans: Box<[Range<usize>]>,
    in_file_offset: usize,
}

impl LoadPlan {
    fn new(spans: Vec<Span>, in_file_offset: usize, chunk_size: NonZeroUsize) -> Self {
        let chunk_size = chunk_size.get();
        assert!(
            spans.iter().all(|s| s.start <= s.end)
                && spans.windows(2).all(|w| w[0].end <= w[1].start),
            "spans must be sorted by start and disjoint, each with start <= end"
        );

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
            span_chunks: span_chunks.into_boxed_slice(),
            chunks: chunks.into_boxed_slice(),
            chunk_spans: chunk_spans.into_boxed_slice(),
            in_file_offset,
        }
    }
}

enum Sink<D> {
    Cuda(CudaSink<D>),
}

impl<D> Sink<D> {
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
        read: impl FnOnce(&mut [u8], u64) -> std::io::Result<()>,
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
        read(
            lease.mut_slice(chunk.len()),
            (self.plan.in_file_offset + chunk.start) as u64,
        )?;

        let e = self.api.with_device(self.device, |d| {
            for s in self.plan.chunk_spans[chunk_idx].clone() {
                let span = self.plan.spans[s];
                let (start, end) = (span.start.max(chunk.start), span.end.min(chunk.end));
                if start < end {
                    self.api.memcpy_h2d_async(
                        self.dsts[s].load(Ordering::Relaxed) + (start - span.start) as u64,
                        lease.ptr().wrapping_add(start - chunk.start),
                        end - start,
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
    file: Arc<File>,
    plan: Arc<LoadPlan>,
    sink: Sink<D>,
    next_chunk: AtomicUsize,
    error: OnceLock<LoaderError>,
}

impl<D> LoaderInner<D> {
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
    workers: Box<[JoinHandle<()>]>,
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
}

impl<D: Send + 'static> Loader<D> {
    /// Starts the workers. They copy a span once it has a destination: give
    /// them with [`Feed::provide`], in plan order, as the load advances.
    pub fn load(
        file: Arc<File>,
        in_file_offset: usize,
        device: i32,
        threads: usize,
        spans: Vec<Span>,
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
        let plan = Arc::new(LoadPlan::new(spans, in_file_offset, CHUNK_SIZE));
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

    /// Stops loading and returns the owners not taken.
    pub fn close(&mut self) -> Vec<D> {
        self.inner.sink.signal_stop();
        for h in std::mem::take(&mut self.workers) {
            let _ = h.join();
        }
        self.inner.sink.release()
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
            let _ = POOL.set(SlabPool::new(cuda_api, 32, CHUNK_SIZE.get())?);
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
mod tests {
    use std::num::NonZeroUsize;

    use super::{LoadPlan, Span};

    fn sp(start: usize, end: usize) -> Span {
        Span { start, end }
    }

    fn plan(spans: Vec<Span>, chunk_size: usize) -> LoadPlan {
        let plan = LoadPlan::new(spans, 0, NonZeroUsize::new(chunk_size).unwrap());
        check_plan(&plan, chunk_size);
        plan
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
}
