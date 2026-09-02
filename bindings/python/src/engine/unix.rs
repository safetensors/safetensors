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

use crate::engine::cuda::{api, CudaApi, CudaError, Event, Stream};

#[derive(Debug)]
pub enum LoaderError {
    AlreadyDelivered,
    Closed,
    Cuda(CudaError),
    CudaRuntimeLoad,
    Io(std::io::Error),
}

impl Display for LoaderError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::AlreadyDelivered => write!(f, "attempt to take already delivered tensor"),
            Self::Closed => write!(f, "load engine already shutdown"),
            Self::Cuda(e) => write!(f, "cuda error: {e}"),
            Self::CudaRuntimeLoad => write!(f, "could not load cuda runtime"),
            Self::Io(e) => write!(f, "io error: {e}"),
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
const DEFAULT_STREAM: Stream = std::ptr::null_mut();

struct DeviceAllocator {
    api: &'static CudaApi,
    device: i32,
    stream: Stream,
}

unsafe impl Send for DeviceAllocator {}
unsafe impl Sync for DeviceAllocator {}

impl DeviceAllocator {
    fn request(self: &Arc<Self>, bytes: usize) -> Result<Arc<Allocation>, LoaderError> {
        let _g = self.api.device_guard(self.device)?;
        let ptr = self.api.malloc_async(bytes, self.stream)?;
        Ok(Arc::new(Allocation {
            allocator: self.clone(),
            ptr,
        }))
    }

    fn release(&self, base: u64) {
        let Ok(_g) = self.api.device_guard(self.device) else {
            return;
        };
        let Ok(free) = free_stream(self.api, self.device) else {
            return;
        };
        for stream in [self.stream, DEFAULT_STREAM] {
            if let Ok(e) = self.api.event_create() {
                let _ = self.api.event_record(e, stream);
                let _ = self.api.stream_wait_event(free, e);
                let _ = self.api.event_destroy(e);
            }
        }
        let _ = self.api.free_async(base, free);
    }
}

struct Allocation {
    allocator: Arc<DeviceAllocator>,
    ptr: u64,
}

impl Drop for Allocation {
    fn drop(&mut self) {
        self.allocator.release(self.ptr);
    }
}

pub struct LoadPlan {
    tensor_offsets: Box<[(usize, usize)]>,
    data_len: usize,
    allocation_ranges: Box<[Range<usize>]>,
    in_file_offset: usize,
    n_chunks: usize,
    chunk_size: usize,
}

impl LoadPlan {
    pub fn new(
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

unsafe impl Send for CudaBuffer {}

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

    fn release(&self) -> Result<(), LoaderError> {
        match self {
            Self::Cuda(sink) => sink.release(),
        }
    }

    fn fully_scheduled(&self, tensor: usize) -> bool {
        match self {
            Self::Cuda(sink) => sink.fully_scheduled(tensor),
        }
    }
}

struct Slab {
    offset: usize,
    event: Event,
    sync_needed: bool,
}

struct SlabLease {
    pool: &'static SlabPool,
    offset: usize,
    event: Event,
    finished: bool,
}

impl SlabLease {
    fn ptr(&self) -> *mut u8 {
        unsafe { self.pool.buffer_ptr.add(self.offset) }
    }

    fn mut_slice(&mut self, len: usize) -> &mut [u8] {
        assert!(len <= self.pool.slab_size, "chunk does not fit in slab");
        unsafe { std::slice::from_raw_parts_mut(self.pool.buffer_ptr.add(self.offset), len) }
    }

    fn finish(mut self, cuda: &CudaApi, stream: Stream) -> Result<(), CudaError> {
        cuda.event_record(self.event, stream)?;
        self.finished = true;
        self.pool.put_back(Slab {
            offset: self.offset,
            event: self.event,
            sync_needed: true,
        });
        Ok(())
    }
}

impl Drop for SlabLease {
    fn drop(&mut self) {
        if !self.finished {
            self.pool.put_back(Slab {
                offset: self.offset,
                event: self.event,
                sync_needed: false,
            });
        }
    }
}

struct SlabPool {
    buffer_ptr: *mut u8,
    slab_size: usize,
    free: Mutex<Vec<Slab>>,
    available: Condvar,
}

unsafe impl Send for SlabPool {}
unsafe impl Sync for SlabPool {}

impl SlabPool {
    fn new(cuda: &CudaApi, n_slabs: usize, slab_size: usize) -> Result<Self, CudaError> {
        let buffer_ptr = cuda.host_alloc(n_slabs * slab_size)?;
        let free = (0..n_slabs)
            .map(|i| {
                Ok(Slab {
                    offset: i * slab_size,
                    event: cuda.event_create()?,
                    sync_needed: false,
                })
            })
            .collect::<Result<Vec<_>, _>>()?;
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
            if let Some(s) = free.pop() {
                break s;
            }
            free = self.available.wait(free).unwrap();
        };
        drop(free);
        if slab.sync_needed {
            cuda.event_sync(slab.event)?;
        }
        Ok(SlabLease {
            pool: self,
            offset: slab.offset,
            event: slab.event,
            finished: false,
        })
    }

    fn put_back(&self, slab: Slab) {
        self.free.lock().unwrap().push(slab);
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

pub struct CudaSink {
    api: &'static CudaApi,
    device: i32,
    stream: Stream,
    pool: &'static SlabPool,
    plan: Arc<LoadPlan>,
    allocator: Arc<DeviceAllocator>,
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
        let _g = api.device_guard(device)?;
        let stream = device_stream(api, device)?;
        let _ = api.pool_set_release_threshold(device, u64::MAX);
        CUDA_ACTIVE_SINKS[device as usize].fetch_add(1, Ordering::AcqRel);
        Ok(Self {
            api,
            device,
            stream,
            pool,
            allocator: Arc::new(DeviceAllocator {
                api,
                device,
                stream,
            }),
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
        let alloc = self.allocator.request(range.end - range.start)?;
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
        let _g = self.api.device_guard(self.device)?;
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
        let e = self.api.event_create()?;
        self.api.event_record(e, self.stream)?;
        self.chunk_completion_events[chunk_idx].store(e as u64, Ordering::Release);
        lease.finish(self.api, self.stream)?;
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
        if self.delivered[tensor].swap(true, Ordering::AcqRel) {
            return Err(LoaderError::AlreadyDelivered);
        }
        let (start, end) = self.plan.tensor_offsets[tensor];
        if start == end {
            return Ok(CudaBuffer {
                _alloc: None,
                device: self.device,
                ptr: 0,
                len: 0,
            });
        }
        let alloc_idx = self.plan.allocation_at(start);
        let Some(alloc) = self.allocations[alloc_idx].lock().unwrap().clone() else {
            return Err(LoaderError::Closed);
        };
        Ok(CudaBuffer {
            ptr: alloc.ptr + (start - self.plan.allocation_ranges[alloc_idx].start) as u64,
            len: end - start,
            device: self.device,
            _alloc: Some(alloc),
        })
    }

    fn release(&self) -> Result<(), LoaderError> {
        self.closed.store(true, Ordering::Release);
        if self.released.swap(true, Ordering::AcqRel) {
            return Ok(());
        }
        for slot in self.allocations.iter() {
            *slot.lock().unwrap() = None;
        }
        if CUDA_ACTIVE_SINKS[self.device as usize].fetch_sub(1, Ordering::AcqRel) == 1 {
            let Ok(_g) = self.api.device_guard(self.device) else {
                return Ok(());
            };
            // NOTE: `let _ =` is intentional, releasing retained memory is best effort
            if let Ok(free) = free_stream(self.api, self.device) {
                if let Ok(e) = self.api.event_create() {
                    let _ = self.api.event_record(e, free);
                    let _ = self.api.event_sync(e);
                    let _ = self.api.event_destroy(e);
                }
            }
            let _ = self.api.pool_set_release_threshold(self.device, 0);
            let _ = self.api.pool_trim(self.device);
        }
        Ok(())
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

    let new = api.stream_create()?;
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

pub struct LoaderInner {
    file: Arc<File>,
    plan: Arc<LoadPlan>,
    sink: Sink,
    next_chunk: AtomicUsize,
    cancelled: AtomicBool,
    error: OnceLock<LoaderError>,
}

impl LoaderInner {
    fn take_tensor(&self, tensor: usize) -> Result<DeviceBuffer, LoaderError> {
        self.sink.wait_ready(tensor).map_err(|e| match e {
            LoaderError::Closed => match self.error.get() {
                Some(err) => LoaderError::Io(std::io::Error::other(err.to_string())),
                None => LoaderError::Closed,
            },
            e => e,
        })?;
        self.sink.take(tensor)
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
        buffer_start_pos: usize,
        device: i32,
        threads: usize,
    ) -> Result<Self, LoaderError> {
        let cuda_api = api().ok_or(LoaderError::CudaRuntimeLoad)?;
        let plan = Arc::new(LoadPlan::new(
            metadata,
            buffer_start_pos,
            CHUNK_SIZE,
            MIN_ALLOCATION_SIZE,
        ));
        let pool = pool(cuda_api)?;
        let inner = Arc::new(LoaderInner {
            file,
            cancelled: AtomicBool::new(false),
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
        self.inner.cancelled.store(true, Ordering::Release);
        self.inner.sink.signal_stop();
        for h in std::mem::take(&mut self.workers) {
            let _ = h.join();
        }
        let _ = self.inner.sink.release();
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
        if loader.cancelled.load(Ordering::Acquire) {
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
            let _ = loader.error.set(err);
            loader.cancelled.store(true, Ordering::Release);
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

    use crate::engine::LoadPlan;

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
