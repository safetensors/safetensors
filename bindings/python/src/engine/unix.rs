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

const CHUNK_SIZE: NonZeroUsize = NonZeroUsize::new(16 * 1024 * 1024).unwrap();

#[derive(Debug, PartialEq)]
pub struct ChunkTensorSlice {
    tensor: usize,
    src: Range<usize>,
    dst_offset: usize,
}

pub struct LoadPlan {
    tensor_offsets: Box<[(usize, usize)]>,
    data_len: usize,
    /// Flat list of tensor-slice-within-a-chunk ([`ChunkTensorSlice`])
    intersections: Box<[ChunkTensorSlice]>,
    /// index into [`Self::intersections`] such that `intersections[chunk_intersections[chunk_idx]..chunk_intersections[chunk_idx + 1]]` -> gives all the intersections that are contained in `chunk_idx`
    chunk_intersections: Box<[usize]>,
    in_file_offset: usize,
    n_chunks: usize,
    chunk_size: usize,
}

impl LoadPlan {
    pub fn new(metadata: &Metadata, in_file_offset: usize, chunk_size: NonZeroUsize) -> Self {
        let chunk_size = chunk_size.get();
        let tensor_offsets: Box<[(usize, usize)]> = metadata
            .tensor_infos()
            .iter()
            .map(|info| info.data_offsets)
            .collect();
        let data_len = metadata.data_len();

        let n_chunks = data_len.div_ceil(chunk_size);

        let mut intersections = Vec::with_capacity(tensor_offsets.len() + n_chunks);
        let mut offsets = vec![0usize; n_chunks + 1];
        let mut cursor = 0;

        for (i, &(start, end)) in tensor_offsets.iter().enumerate() {
            if start == end {
                continue;
            }
            for c in (start / chunk_size)..=((end - 1) / chunk_size) {
                if cursor < c {
                    offsets[cursor + 1] = intersections.len();
                    cursor += 1;
                }
                debug_assert_eq!(c, cursor);
                let chunk_start = c * chunk_size;
                let overlap =
                    start.max(chunk_start)..end.min((chunk_start + chunk_size).min(data_len));
                intersections.push(ChunkTensorSlice {
                    tensor: i,
                    // convert to offsets within the chunk
                    src: (overlap.start - chunk_start)..(overlap.end - chunk_start),
                    // convert to offset within the destination tensor
                    dst_offset: overlap.start - start,
                });
            }
        }

        if cursor < n_chunks {
            offsets[cursor + 1] = intersections.len();
        }

        Self {
            tensor_offsets,
            data_len,
            intersections: intersections.into_boxed_slice(),
            chunk_intersections: offsets.into_boxed_slice(),
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
    /// Tensor slices contained in a chunk
    fn chunk_tensor_slices(&self, chunk_idx: usize) -> &[ChunkTensorSlice] {
        &self.intersections
            [self.chunk_intersections[chunk_idx]..self.chunk_intersections[chunk_idx + 1]]
    }

    /// Chunk idx list that contain slices of a given tensor
    fn tensor_chunks(&self, tensor: usize) -> Range<usize> {
        let (s, e) = &self.tensor_offsets[tensor];
        if s == e {
            return 0..0;
        }
        (s / self.chunk_size)..((e - 1) / self.chunk_size + 1)
    }

    fn tensor_size(&self, tensor: usize) -> usize {
        let (s, e) = self.tensor_offsets[tensor];
        e - s
    }
}

pub struct CudaBuffer {
    api: &'static CudaApi,
    device: i32,
    stream: Stream,
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

impl Drop for CudaBuffer {
    fn drop(&mut self) {
        if self.ptr != 0 {
            if let Ok(_g) = self.api.device_guard(self.device) {
                let _ = self.api.free_async(self.ptr, self.stream);
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
    fn buffer(&mut self, len: usize) -> &mut [u8] {
        assert!(len <= self.pool.slab_size, "chunk larger than slab size");
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
        let buffer = cuda.host_alloc(n_slabs * slab_size)?;
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
            buffer_ptr: buffer,
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

struct DestinationPtr(AtomicU64);
const DELIVERED: u64 = u64::MAX;

enum Take {
    Ptr(u64),
    Unallocated,
    AlreadyDelivered,
}

impl DestinationPtr {
    fn get_or_alloc(&self, len: usize, api: &CudaApi, stream: Stream) -> Result<u64, CudaError> {
        let ptr = self.0.load(Ordering::Acquire);
        if ptr != 0 {
            return Ok(ptr);
        }
        let new = api.malloc_async(len, stream)?;
        let ptr = match self
            .0
            .compare_exchange(0, new, Ordering::AcqRel, Ordering::Acquire)
        {
            Ok(_) => new,
            Err(existing) => {
                api.free_async(new, stream)?;
                existing
            }
        };
        Ok(ptr)
    }

    fn take(&self) -> Take {
        match self.0.swap(DELIVERED, Ordering::AcqRel) {
            0 => Take::Unallocated,
            DELIVERED => Take::AlreadyDelivered,
            p => Take::Ptr(p),
        }
    }

    /// reset the pointer to `0` and return the previous ptr if any
    fn reset(&self) -> Option<u64> {
        match self.0.swap(0, Ordering::AcqRel) {
            0 | DELIVERED => None,
            p => Some(p),
        }
    }
}

pub struct CudaSink {
    api: &'static CudaApi,
    device: i32,
    stream: Stream,
    pool: &'static SlabPool,
    plan: Arc<LoadPlan>,
    chunk_completion_events: Box<[AtomicU64]>,
    dests: Box<[DestinationPtr]>,
    closed: AtomicBool,
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
        let n_tensors = plan.tensor_offsets.len();
        Ok(Self {
            api,
            device,
            stream,
            pool,
            chunk_completion_events: (0..plan.n_chunks).map(|_| AtomicU64::new(0)).collect(),
            dests: (0..n_tensors)
                .map(|_| DestinationPtr(AtomicU64::new(0)))
                .collect(),
            closed: AtomicBool::new(false),
            plan,
        })
    }

    fn load_chunk(
        &self,
        chunk_idx: usize,
        len: usize,
        read: impl FnOnce(&mut [u8]) -> std::io::Result<()>,
    ) -> Result<(), LoaderError> {
        let mut lease = self.pool.acquire(self.api)?;
        read(lease.buffer(len))?;
        let _g = self.api.device_guard(self.device)?;
        let slab = unsafe { self.pool.buffer_ptr.add(lease.offset) };
        for slice in self.plan.chunk_tensor_slices(chunk_idx) {
            let dst = self.dests[slice.tensor].get_or_alloc(
                self.plan.tensor_size(slice.tensor),
                self.api,
                self.stream,
            )?;
            self.api.memcpy_h2d_async(
                dst + slice.dst_offset as u64,
                unsafe { slab.add(slice.src.start) },
                slice.src.end - slice.src.start,
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
        assert!(tensor < self.dests.len());
        match self.dests[tensor].take() {
            Take::Ptr(ptr) => Ok(CudaBuffer {
                api: self.api,
                device: self.device,
                stream: self.stream,
                ptr,
                len: self.plan.tensor_size(tensor),
            }),
            Take::Unallocated => Ok(CudaBuffer {
                api: self.api,
                device: self.device,
                stream: self.stream,
                ptr: 0,
                len: 0,
            }),
            Take::AlreadyDelivered => Err(LoaderError::AlreadyDelivered),
        }
    }

    fn release(&self) -> Result<(), LoaderError> {
        self.closed.store(true, Ordering::Release);
        let _g = self.api.device_guard(self.device)?;
        for dst in self.dests.iter() {
            if let Some(p) = dst.reset() {
                self.api.free_async(p, self.stream)?;
            }
        }
        for e in self.chunk_completion_events.iter() {
            let e = e.load(Ordering::Acquire);
            if e != 0 {
                self.api.event_sync(e as Event)?;
            }
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

fn device_stream(api: &CudaApi, device: i32) -> Result<Stream, CudaError> {
    let slot = STREAMS
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
        let plan = Arc::new(LoadPlan::new(metadata, buffer_start_pos, CHUNK_SIZE));
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
    use std::num::NonZeroUsize;

    use safetensors::{
        tensor::{Metadata, TensorInfo},
        Dtype,
    };

    use crate::engine::{ChunkTensorSlice, LoadPlan};

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

    fn check_plan(plan: &LoadPlan) {
        assert_eq!(plan.chunk_intersections[0], 0);
        assert_eq!(
            *plan.chunk_intersections.last().unwrap(),
            plan.intersections.len()
        );
        assert!(plan.chunk_intersections.windows(2).all(|w| w[0] <= w[1]));

        for c in 0..plan.n_chunks {
            let chunk_len = plan.chunk_len(c);
            let mut cursor = 0;
            for slice in plan.chunk_tensor_slices(c) {
                assert_eq!(slice.src.start, cursor, "gap/overlap inside chunk {c}");
                assert!(slice.src.end > slice.src.start);
                cursor = slice.src.end;
            }
            assert_eq!(
                cursor, chunk_len,
                "tensor chunk slices don't cover the full chunk"
            );
        }
        let mut tensor_slices = vec![vec![]; plan.tensor_offsets.len()];
        for slice in plan.intersections.iter() {
            tensor_slices[slice.tensor].push((slice.dst_offset, slice.src.end - slice.src.start))
        }
        for (i, (start, end)) in plan.tensor_offsets.iter().enumerate() {
            let size = end - start;
            let mut cursor = 0;
            for (dst, len) in &tensor_slices[i] {
                assert_eq!(
                    *dst, cursor,
                    "tensor {i}: gap/overlap at in destination buffer offset: {dst}"
                );
                cursor += len;
            }
            assert_eq!(
                cursor, size,
                "tensor {i}: missing last slice ({cursor} / {size})"
            );
        }
    }

    #[test]
    fn test_plan_valid_intersect() {
        let chunk_size = NonZeroUsize::new(5).unwrap();
        let metadata = Metadata::new(
            None,
            vec![t("first", 0, 5), t("second", 5, 7), t("third", 12, 6)],
        )
        .unwrap();
        let load_plan = LoadPlan::new(&metadata, 0, chunk_size);
        assert_eq!(
            load_plan.intersections,
            vec![
                ChunkTensorSlice {
                    tensor: 0,
                    src: 0..5,
                    dst_offset: 0,
                },
                ChunkTensorSlice {
                    tensor: 1,
                    src: 0..5,
                    dst_offset: 0,
                },
                ChunkTensorSlice {
                    tensor: 1,
                    src: 0..2,
                    dst_offset: 5,
                },
                ChunkTensorSlice {
                    tensor: 2,
                    src: 2..5,
                    dst_offset: 0,
                },
                ChunkTensorSlice {
                    tensor: 2,
                    src: 0..3,
                    dst_offset: 3,
                }
            ]
            .into_boxed_slice(),
        );
        assert_eq!(
            load_plan.chunk_intersections,
            vec![0, 1, 2, 4, 5].into_boxed_slice()
        );
        check_plan(&load_plan);
    }

    #[test]
    fn test_chunk_len_exact_multiple_of_chunk_size() {
        let chunk_size = NonZeroUsize::new(5).unwrap();
        let metadata = Metadata::new(None, vec![t("first", 0, 10)]).unwrap();
        let load_plan = LoadPlan::new(&metadata, 0, chunk_size);
        check_plan(&load_plan);
    }

    #[test]
    fn test_multi_chunk_span() {
        let chunk_size = NonZeroUsize::new(5).unwrap();
        let metadata = Metadata::new(None, vec![t("first", 0, 42)]).unwrap();
        let load_plan = LoadPlan::new(&metadata, 0, chunk_size);
        let offsets: Vec<_> = load_plan
            .intersections
            .iter()
            .map(|s| s.dst_offset)
            .collect();
        assert_eq!(offsets, vec![0, 5, 10, 15, 20, 25, 30, 35, 40]);
        check_plan(&load_plan);
    }

    #[test]
    fn test_many_tensors_in_single_chunk() {
        let chunk_size = NonZeroUsize::new(20).unwrap();
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
        let load_plan = LoadPlan::new(&metadata, 0, chunk_size);
        check_plan(&load_plan);
    }

    #[test]
    fn test_chunk_size_larger_than_data() {
        let chunk_size = NonZeroUsize::new(4242).unwrap();
        let metadata = Metadata::new(None, vec![t("a", 0, 5), t("b", 5, 3)]).unwrap();
        let load_plan = LoadPlan::new(&metadata, 0, chunk_size);
        check_plan(&load_plan);
    }

    #[test]
    fn test_empty_slices() {
        let chunk_size = NonZeroUsize::new(5).unwrap();
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
        let load_plan = LoadPlan::new(&metadata, 0, chunk_size);
        check_plan(&load_plan);
    }
}
