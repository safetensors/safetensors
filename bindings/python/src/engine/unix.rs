//! CUDA prefetch engine: load a whole safetensors data section into device
//! memory in the background and hand tensors out, once each, as they land.
//!
//! Two partitions of the data section that do not nest:
//!
//! ```text
//! tensors  |t0 |  t1  |     t2     |  t3  |        t4       | t5 | t6  |
//! chunks   |   c0   |   c1   |   c2   |   c3   |   c4   |   c5   | c6  |
//! ranges   |        range 0        |         range 1        |  range 2 |
//! ```
//!
//! - chunks: fixed `CHUNK_SIZE` grid, the unit of movement (one `pread`, one
//!   slab, one set of copies). `LoadPlan::tensor_chunks` maps a tensor to the
//!   chunks it needs.
//! - ranges: `LoadPlan::allocation_ranges`, the unit of residence (one
//!   `cudaMallocAsync` each). Cut at tensor ends so a tensor never crosses a
//!   range; every range but the last is >= `MIN_ALLOCATION_SIZE`. A chunk may
//!   overlap two ranges (c2 and c5 above), hence <= 2 copies per chunk. The
//!   layout is identity: a tensor's device address is its range base plus
//!   (file offset - range start), so tensors are views and are never copied
//!   again.
//!
//! Data flow, per file:
//!
//! ```text
//! worker threads
//!   c = next_chunk.fetch_add(1)
//!   lease = SlabPool::acquire()          pinned host slab (process-global
//!   pread(chunk c) -> lease              pool), reused once the copy that
//!   memcpyAsync(lease -> device)         last used it completes; <= 2 copies
//!                                        per chunk on STREAMS[device]
//!   chunk_completion_events[c] = event   recorded after the copies; 0 while
//!                                        the chunk is still pending
//! consumer
//!   take_tensor(t) / TensorIter::next    wait the events of t's chunks, then
//!     -> CudaBuffer                      a view into Arc<Allocation>, handed
//!                                        out once (AlreadyDelivered after)
//!   drop(last view of a range)           free_async on FREE_STREAMS[device],
//!                                        fenced on the load stream and the
//!                                        legacy default stream
//! ```
//!
//! `Loader` owns the worker threads; `LoaderInner` (file, plan, sink, chunk
//! counter, first error) is shared with them and with `TensorIter`. Closing
//! signals the workers, joins them, then releases the sink: an outstanding
//! iterator yields `Err(Closed)` once and is exhausted. `CUDA_ACTIVE_SINKS`
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

use safetensors::tensor::Metadata;

use crate::engine::cuda::{api, CudaApi, CudaError, DeviceGuard, Event, Stream};

#[derive(Debug)]
pub enum LoaderError {
    AlreadyDelivered,
    Closed,
    Cuda(CudaError),
    CudaRuntimeLoad,
    Io(std::io::Error),
    WorkerFailed(String),
}

impl Display for LoaderError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::AlreadyDelivered => write!(
                f,
                "tensor already delivered (prefetch hands each tensor out once, via get_tensor or tensor_stream)"
            ),
            Self::Closed => write!(f, "prefetch loader closed"),
            Self::Cuda(e) => write!(f, "{e}"),
            Self::CudaRuntimeLoad => write!(
                f,
                "no CUDA runtime is loaded in this process; import a CUDA-enabled \
                 framework (torch) before opening with prefetch=True"
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

/// One `allocation_range` of device memory — the only malloc/free site for
/// tensor memory. Freed when the sink clears its slot and the last
/// `CudaBuffer` view is dropped.
struct Allocation {
    api: &'static CudaApi,
    device: i32,
    stream: Stream,
    ptr: u64,
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
        }))
    }
}

/// Frees on the per-device free stream once it has waited on the load stream
/// (our pending copies) and the legacy default stream (consumer reads).
/// Consumers on other streams must sync before dropping their last view;
/// `__dlpack__(stream=)` negotiation is the planned replacement.
impl Drop for Allocation {
    fn drop(&mut self) {
        let free = free_stream(self.api, self.device).unwrap_or(self.stream);
        let _: Result<(), CudaError> = self.api.with_device(self.device, |d| {
            for stream in [self.stream, DEFAULT_STREAM] {
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

struct LoadPlan {
    tensor_offsets: Box<[(usize, usize)]>,
    data_len: usize,
    allocation_ranges: Box<[Range<usize>]>,
    in_file_offset: usize,
    n_chunks: usize,
    chunk_size: usize,
}

impl LoadPlan {
    fn new(
        metadata: &Metadata,
        in_file_offset: usize,
        chunk_size: NonZeroUsize,
        min_allocation_size: NonZeroUsize,
    ) -> Self {
        let chunk_size = chunk_size.get();
        let min_allocation_size = min_allocation_size.get();
        let tensor_offsets: Box<[(usize, usize)]> = metadata
            .tensor_infos()
            .iter()
            .map(|info| info.data_offsets)
            .collect();
        let data_len = metadata.data_len();

        let mut allocation_ranges = Vec::new();
        let mut alloc_start = 0;
        for &(_, end) in tensor_offsets.iter() {
            if end - alloc_start >= min_allocation_size {
                allocation_ranges.push(alloc_start..end);
                alloc_start = end;
            }
        }
        if alloc_start < data_len {
            allocation_ranges.push(alloc_start..data_len);
        }

        let n_chunks = data_len.div_ceil(chunk_size);

        Self {
            tensor_offsets,
            data_len,
            allocation_ranges: allocation_ranges.into_boxed_slice(),
            in_file_offset,
            n_chunks,
            chunk_size,
        }
    }

    fn chunk_len(&self, chunk_idx: usize) -> usize {
        self.chunk_size
            .min(self.data_len - chunk_idx * self.chunk_size)
    }

    fn chunk_file_offset(&self, chunk_idx: usize) -> usize {
        self.in_file_offset + chunk_idx * self.chunk_size
    }
}

impl LoadPlan {
    fn allocation_at(&self, offset: usize) -> usize {
        self.allocation_ranges.partition_point(|r| r.end <= offset)
    }

    fn chunk_allocations(&self, chunk_idx: usize) -> Range<usize> {
        let start = chunk_idx * self.chunk_size;
        let end = start + self.chunk_len(chunk_idx);
        self.allocation_at(start)..self.allocation_at(end - 1) + 1
    }

    /// Chunk idx list that contain slices of a given tensor
    fn tensor_chunks(&self, tensor: usize) -> Range<usize> {
        let (s, e) = &self.tensor_offsets[tensor];
        if s == e {
            return 0..0;
        }
        (s / self.chunk_size)..((e - 1) / self.chunk_size + 1)
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
}

pub enum DeviceBuffer {
    Cuda(CudaBuffer),
}

enum Sink {
    Cuda(CudaSink),
}

impl Sink {
    fn load_chunk(
        &self,
        chunk_idx: usize,
        len: usize,
        read: impl FnOnce(&mut [u8]) -> std::io::Result<()>,
    ) -> Result<(), LoaderError> {
        match self {
            Self::Cuda(sink) => sink.load_chunk(chunk_idx, len, read),
        }
    }

    fn wait_ready(&self, tensor: usize) -> Result<(), LoaderError> {
        match self {
            Self::Cuda(sink) => sink.wait_ready(tensor),
        }
    }

    fn take(&self, tensor: usize) -> Result<DeviceBuffer, LoaderError> {
        match self {
            Self::Cuda(sink) => Ok(DeviceBuffer::Cuda(sink.take(tensor)?)),
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

    fn fully_scheduled(&self, tensor: usize) -> bool {
        match self {
            Self::Cuda(sink) => sink.fully_scheduled(tensor),
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
        let event = match self.slab.last_copy {
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
            allocations: (0..plan.allocation_ranges.len())
                .map(|_| Mutex::new(None))
                .collect(),
            delivered: (0..plan.tensor_offsets.len())
                .map(|_| AtomicBool::new(false))
                .collect(),
            chunk_completion_events: (0..plan.n_chunks).map(|_| AtomicU64::new(0)).collect(),
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
        let range = &self.plan.allocation_ranges[alloc_idx];
        let alloc = Allocation::new(self.api, self.device, self.stream, range.end - range.start)?;
        let ptr = alloc.ptr;
        *slot = Some(alloc);
        Ok(ptr)
    }

    fn load_chunk(
        &self,
        chunk_idx: usize,
        len: usize,
        read: impl FnOnce(&mut [u8]) -> std::io::Result<()>,
    ) -> Result<(), LoaderError> {
        let mut lease = self.pool.acquire(self.api)?;
        read(lease.mut_slice(len))?;

        let chunk_start = chunk_idx * self.plan.chunk_size;
        for alloc in self.plan.chunk_allocations(chunk_idx) {
            let range = &self.plan.allocation_ranges[alloc];
            let dst = self.allocation_ptr(alloc)?;
            let start = chunk_start.max(range.start);
            let end = (chunk_start + len).min(range.end);
            self.api.memcpy_h2d_async(
                dst + (start - range.start) as u64,
                unsafe { lease.ptr().add(start - chunk_start) },
                end - start,
                self.stream,
            )?;
        }

        let e = self.api.with_device(self.device, |d| {
            lease.release(d, self.stream)?;
            d.event_create()
        })?;

        if let Err(err) = self.api.event_record(e, self.stream) {
            let _ = self.api.event_destroy(e);
            return Err(err.into());
        }
        self.chunk_completion_events[chunk_idx].store(e as u64, Ordering::Release);

        Ok(())
    }

    fn wait_ready(&self, tensor: usize) -> Result<(), LoaderError> {
        for chunk in self.plan.tensor_chunks(tensor) {
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

    fn take(&self, tensor: usize) -> Result<CudaBuffer, LoaderError> {
        if self.delivered[tensor].load(Ordering::Acquire) {
            return Err(LoaderError::AlreadyDelivered);
        }
        let (start, end) = self.plan.tensor_offsets[tensor];
        let mut buffer = CudaBuffer {
            _alloc: None,
            device: self.device,
            ptr: 0,
            len: end - start,
        };
        if start != end {
            let alloc_idx = self.plan.allocation_at(start);
            let Some(alloc) = self.allocations[alloc_idx].lock().unwrap().clone() else {
                return Err(LoaderError::Closed);
            };
            buffer.ptr = alloc.ptr + (start - self.plan.allocation_ranges[alloc_idx].start) as u64;
            buffer._alloc = Some(alloc);
        }
        if self.delivered[tensor].swap(true, Ordering::AcqRel) {
            return Err(LoaderError::AlreadyDelivered);
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

    fn fully_scheduled(&self, tensor: usize) -> bool {
        self.plan
            .tensor_chunks(tensor)
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
    fn take_tensor(&self, tensor: usize) -> Result<DeviceBuffer, LoaderError> {
        self.sink
            .wait_ready(tensor)
            .and_then(|()| self.sink.take(tensor))
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
        metadata: &Metadata,
        in_file_offset: usize,
        device: i32,
        threads: usize,
    ) -> Result<Self, LoaderError> {
        let cuda_api = api().ok_or(LoaderError::CudaRuntimeLoad)?;
        let plan = Arc::new(LoadPlan::new(
            metadata,
            in_file_offset,
            CHUNK_SIZE,
            MIN_ALLOCATION_SIZE,
        ));
        let pool = pool(cuda_api)?;
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
                    || worker(inner)
                })
            })
            .collect();
        Ok(Self { inner, workers })
    }

    pub fn take_tensor(&self, tensor: usize) -> Result<DeviceBuffer, LoaderError> {
        self.inner.take_tensor(tensor)
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
            pending: (0..self.inner.plan.tensor_offsets.len()).collect(),
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
        if chunk_idx >= loader.plan.n_chunks {
            return;
        }
        let res = loader
            .sink
            .load_chunk(chunk_idx, loader.plan.chunk_len(chunk_idx), |buffer| {
                loader
                    .file
                    .read_exact_at(buffer, loader.plan.chunk_file_offset(chunk_idx) as u64)
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

    use safetensors::{
        tensor::{Metadata, TensorInfo},
        Dtype,
    };

    use super::LoadPlan;

    fn t(name: &str, start: usize, len: usize) -> (String, TensorInfo) {
        (
            name.into(),
            TensorInfo {
                dtype: Dtype::U8,
                shape: vec![len],
                data_offsets: (start, start + len),
            },
        )
    }

    fn check_plan(plan: &LoadPlan, min_allocation_size: usize) {
        let mut cursor = 0;
        for r in plan.allocation_ranges.iter() {
            assert_eq!(r.start, cursor, "gap/overlap between allocation ranges");
            assert!(r.end > r.start);
            cursor = r.end;
        }
        assert_eq!(cursor, plan.data_len);
        let ends: HashSet<usize> = plan.tensor_offsets.iter().map(|&(_, e)| e).collect();
        for r in plan.allocation_ranges.iter().rev().skip(1) {
            assert!(
                ends.contains(&r.end),
                "cut at {} is not a tensor end",
                r.end
            );
            assert!(
                r.end - r.start >= min_allocation_size,
                "undersized non-tail range {r:?}"
            );
            for &(_, t) in plan.tensor_offsets.iter() {
                assert!(
                    !(t > r.start && t < r.end && t - r.start >= min_allocation_size),
                    "range {r:?} should have been cut earlier, at tensor end {t}"
                );
            }
        }
        for &(s, e) in plan.tensor_offsets.iter() {
            if s == e {
                continue;
            }
            let r = &plan.allocation_ranges[plan.allocation_at(s)];
            assert!(r.start <= s && e <= r.end, "tensor {s}..{e} crosses {r:?}");
        }

        for c in 0..plan.n_chunks {
            let (chunk_start, chunk_end) =
                (c * plan.chunk_size, c * plan.chunk_size + plan.chunk_len(c));
            let got = plan.chunk_allocations(c);
            for (i, r) in plan.allocation_ranges.iter().enumerate() {
                assert_eq!(
                    got.contains(&i),
                    r.start < chunk_end && chunk_start < r.end,
                    "chunk {c} vs range {i}"
                );
            }
            if min_allocation_size >= plan.chunk_size {
                assert!(got.len() <= 2, "chunk {c} spans {} allocations", got.len());
            }
            for (t, &(s, e)) in plan.tensor_offsets.iter().enumerate() {
                assert_eq!(
                    plan.tensor_chunks(t).contains(&c),
                    s != e && s < chunk_end && chunk_start < e,
                    "tensor {t} vs chunk {c}"
                );
            }
        }
    }

    #[test]
    fn test_plan_allocation_ranges() {
        let chunk_size = NonZeroUsize::new(5).unwrap();
        let min_allocation_size = NonZeroUsize::new(12).unwrap();
        let metadata = Metadata::new(
            None,
            vec![t("first", 0, 5), t("second", 5, 7), t("third", 12, 6)],
        )
        .unwrap();
        let load_plan = LoadPlan::new(&metadata, 0, chunk_size, min_allocation_size);

        assert_eq!(
            load_plan.allocation_ranges,
            vec![0..12, 12..18].into_boxed_slice(),
        );
        check_plan(&load_plan, 12);
    }

    #[test]
    fn test_allocation_ranges_oversized_tensor() {
        // floor 8: cut at the first tensor end >= 8 bytes in; the 30-byte
        // tensor forces a large range; the 3-byte tail stays undersized
        let chunk_size = NonZeroUsize::new(5).unwrap();
        let min_allocation_size = NonZeroUsize::new(8).unwrap();
        let metadata = Metadata::new(
            None,
            vec![t("a", 0, 5), t("b", 5, 7), t("big", 12, 30), t("c", 42, 3)],
        )
        .unwrap();
        let load_plan = LoadPlan::new(&metadata, 0, chunk_size, min_allocation_size);
        assert_eq!(
            load_plan.allocation_ranges,
            vec![0..12, 12..42, 42..45].into_boxed_slice(),
        );
        check_plan(&load_plan, 8);
    }

    #[test]
    fn test_chunk_len_exact_multiple_of_chunk_size() {
        let chunk_size = NonZeroUsize::new(5).unwrap();
        let min_allocation_size = NonZeroUsize::new(12).unwrap();
        let metadata = Metadata::new(None, vec![t("first", 0, 10)]).unwrap();
        let load_plan = LoadPlan::new(&metadata, 0, chunk_size, min_allocation_size);
        check_plan(&load_plan, 12);
    }

    #[test]
    fn test_multi_chunk_span() {
        let chunk_size = NonZeroUsize::new(5).unwrap();
        let min_allocation_size = NonZeroUsize::new(12).unwrap();
        let metadata = Metadata::new(None, vec![t("first", 0, 42)]).unwrap();
        let load_plan = LoadPlan::new(&metadata, 0, chunk_size, min_allocation_size);
        assert_eq!(load_plan.tensor_chunks(0), 0..9);
        check_plan(&load_plan, 12);
    }

    #[test]
    fn test_many_tensors_in_single_chunk() {
        let chunk_size = NonZeroUsize::new(20).unwrap();
        let min_allocation_size = NonZeroUsize::new(20).unwrap();
        let metadata = Metadata::new(
            None,
            vec![
                t("a", 0, 1),
                t("b", 1, 1),
                t("c", 2, 1),
                t("d", 3, 1),
                t("e", 4, 1),
                t("f", 5, 1),
                t("g", 6, 1),
                t("h", 7, 1),
                t("i", 8, 3),
                t("j", 11, 6),
                t("k", 17, 3),
            ],
        )
        .unwrap();
        let load_plan = LoadPlan::new(&metadata, 0, chunk_size, min_allocation_size);
        check_plan(&load_plan, 20);
    }

    #[test]
    fn test_chunk_size_larger_than_data() {
        let chunk_size = NonZeroUsize::new(4242).unwrap();
        let min_allocation_size = NonZeroUsize::new(4242).unwrap();
        let metadata = Metadata::new(None, vec![t("a", 0, 5), t("b", 5, 3)]).unwrap();
        let load_plan = LoadPlan::new(&metadata, 0, chunk_size, min_allocation_size);
        check_plan(&load_plan, 4242);
    }

    #[test]
    fn test_empty_slices() {
        let chunk_size = NonZeroUsize::new(5).unwrap();
        let min_allocation_size = NonZeroUsize::new(5).unwrap();
        let metadata = Metadata::new(
            None,
            vec![
                t("z0", 0, 0),
                t("a", 0, 5),
                t("z1", 5, 0),
                t("b", 5, 3),
                t("z2", 8, 0),
            ],
        )
        .unwrap();
        let load_plan = LoadPlan::new(&metadata, 0, chunk_size, min_allocation_size);
        check_plan(&load_plan, 5);
    }
}
