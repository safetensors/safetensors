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
//!   delivery unit: `take` hands out one span as a device view.
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
//!   memcpyAsync(lease -> device)         one copy per chunk into its Allocation,
//!                                        on STREAMS[device]
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

/// Geometry of one load. Every interval here is a byte range within the file's
/// data section; device addresses never appear in the plan.
struct LoadPlan {
    spans: Box<[Span]>,
    /// per span, the [`Allocation`] holding it; `None` for a zero-size span
    span_alloc: Box<[Option<usize>]>,
    /// per span, the `chunks` that carry its bytes; empty for a zero-size span
    span_chunks: Box<[Range<usize>]>,
    /// in file data section byte interval each [`Allocation`] covers
    allocation_file_ranges: Box<[Range<usize>]>,
    /// in file data section byte interval of each chunk: at most [`CHUNK_SIZE`],
    /// inside one allocation
    chunks: Box<[Range<usize>]>,
    /// per chunk, the [`Allocation`] it lands in
    chunk_alloc: Box<[usize]>,
    in_file_offset: usize,
}

impl LoadPlan {
    fn new(
        spans: Vec<Span>,
        in_file_offset: usize,
        chunk_size: NonZeroUsize,
        min_allocation_size: NonZeroUsize,
    ) -> Self {
        let chunk_size = chunk_size.get();
        let min_allocation_size = min_allocation_size.get();
        assert!(
            spans.iter().all(|s| s.start <= s.end)
                && spans.windows(2).all(|w| w[0].end <= w[1].start),
            "spans must be sorted by start and disjoint, each with start <= end"
        );

        let mut allocation_file_ranges: Vec<Range<usize>> = Vec::new();
        let mut span_alloc: Vec<Option<usize>> = vec![None; spans.len()];
        let (mut alloc_start, mut prev_end) = (0, 0);
        for (i, span) in spans.iter().enumerate() {
            if span.start == span.end {
                continue; // no bytes: no range, no chunks
            }
            // gap between spans
            if span.start != prev_end {
                if prev_end > alloc_start {
                    allocation_file_ranges.push(alloc_start..prev_end);
                }
                alloc_start = span.start;
            }
            span_alloc[i] = Some(allocation_file_ranges.len());
            prev_end = span.end;
            if prev_end - alloc_start >= min_allocation_size {
                allocation_file_ranges.push(alloc_start..prev_end);
                alloc_start = prev_end;
            }
        }
        if prev_end > alloc_start {
            allocation_file_ranges.push(alloc_start..prev_end);
        }

        let mut chunks: Vec<Range<usize>> = Vec::new();
        let mut chunk_alloc: Vec<usize> = Vec::new();
        let mut first_chunk = Vec::with_capacity(allocation_file_ranges.len());
        for (alloc_idx, alloc_file_range) in allocation_file_ranges.iter().enumerate() {
            first_chunk.push(chunks.len());
            let mut start = alloc_file_range.start;
            while start < alloc_file_range.end {
                let end = (start + chunk_size).min(alloc_file_range.end);
                chunks.push(start..end);
                chunk_alloc.push(alloc_idx);
                start = end;
            }
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

        Self {
            spans: spans.into_boxed_slice(),
            span_alloc: span_alloc.into_boxed_slice(),
            span_chunks,
            allocation_file_ranges: allocation_file_ranges.into_boxed_slice(),
            chunks: chunks.into_boxed_slice(),
            chunk_alloc: chunk_alloc.into_boxed_slice(),
            in_file_offset,
        }
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
        let bytes = &self.plan.allocation_file_ranges[alloc_idx];
        let alloc = Allocation::new(self.api, self.device, self.stream, bytes.len())?;
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

        let alloc_start = self.plan.allocation_file_ranges[alloc_idx].start;
        let dst = self.allocation_ptr(alloc_idx)?;
        let e = self.api.with_device(self.device, |d| {
            self.api.memcpy_h2d_async(
                dst + (chunk.start - alloc_start) as u64,
                lease.ptr(),
                chunk.len(),
                self.stream,
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
        let Span { start, end, .. } = self.plan.spans[idx];
        let mut buffer = CudaBuffer {
            _alloc: None,
            device: self.device,
            ptr: 0,
            len: end - start,
        };
        if let Some(alloc_idx) = self.plan.span_alloc[idx] {
            let Some(alloc) = self.allocations[alloc_idx].lock().unwrap().clone() else {
                return Err(if self.delivered[idx].load(Ordering::Acquire) {
                    LoaderError::AlreadyDelivered
                } else {
                    LoaderError::Closed
                });
            };
            buffer.ptr =
                alloc.ptr + (start - self.plan.allocation_file_ranges[alloc_idx].start) as u64;
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
        let plan = Arc::new(LoadPlan::new(
            spans,
            in_file_offset,
            CHUNK_SIZE,
            MIN_ALLOCATION_SIZE,
        ));
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
mod tests {
    use std::{collections::HashSet, num::NonZeroUsize};

    use super::{LoadPlan, Span};

    fn sp(start: usize, end: usize) -> Span {
        Span { start, end }
    }

    fn plan(spans: Vec<Span>, chunk_size: usize, min_allocation_size: usize) -> LoadPlan {
        let plan = LoadPlan::new(
            spans,
            0,
            NonZeroUsize::new(chunk_size).unwrap(),
            NonZeroUsize::new(min_allocation_size).unwrap(),
        );
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
