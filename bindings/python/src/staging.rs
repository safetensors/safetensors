//! Staged CUDA loading: chunked parallel `pread(2)` through a ring of
//! reusable pinned slabs, `cudaMemcpyAsync` to the device, event fences.
//!
//! Layering, designed so the read engine is swappable (an io_uring engine
//! could replace the pread workers without touching anything else):
//!
//! - **configuration**: [`LoadConfig`] + [`configure`], one-shot before the
//!   engine's first use; the engine lazily initializes with defaults
//!   otherwise and is immutable afterwards.
//! - **slab ring**: a process-global [`SlabPool`] of chunk-sized pinned
//!   buffers allocated once with `cudaHostAlloc` and reused for every load
//!   (per-load pinned alloc/free churn is what makes naive threaded CUDA
//!   loading slower than single-threaded). A slab is handed back to the
//!   ring tagged with the event of the copy that consumed it; the next
//!   acquirer synchronizes that fence before reuse. One ring serves every
//!   CUDA device in the process.
//! - **engine**: one process-global [`Engine`] worker pool serving every
//!   open file through a two-lane work queue of `(file state, chunk)`
//!   descriptors — consumer-demanded (forced) items strictly before
//!   opportunistic read-ahead. Destinations are per-tensor stream-ordered
//!   CUDA allocations (`cudaMallocAsync`), the whole pipeline is GIL-free,
//!   and each chunk is scattered straight into the final tensors' device
//!   memory, so a tensor is consumable as soon as its covering chunks
//!   land. Consumers receive raw [`AsyncDeviceBuf`]s; DLPack/framework
//!   wrapping happens on their own thread. Bulk `get_tensors()` and
//!   `prefetch=True` streaming both ride this engine. Idle workers top up
//!   the windows of live files in open order, keeping several files' reads
//!   in flight at once.
//!
//! Backpressure is two independent loops. The slab ring recycles at
//! H2D-completion speed, so a consumer sitting on delivered tensors can
//! never starve the readers of pinned buffers; the global [`Budget`]
//! separately bounds ready-but-unconsumed *device* bytes across all files
//! being prefetched at once.

use std::collections::{HashMap, VecDeque};
use std::fs::File;
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::sync::{Arc, Condvar, Mutex, OnceLock, Weak};
use std::time::Duration;

use crate::cuda::{self, CudaApi, Event, Stream};
use crate::read_exact_at;

/// Tunables of the CUDA loading engine. Set once per process with
/// [`configure`] before the engine's first use; afterwards immutable.
#[derive(Clone, Copy, Debug)]
pub struct LoadConfig {
    /// Chunk (= pinned slab) size in MiB.
    pub chunk_mb: usize,
    /// Global worker-thread count shared by every open file.
    pub workers: usize,
    /// Pinned slab ring depth.
    pub slabs: usize,
    /// Opportunistic read-ahead budget in MiB (allocated-but-unconsumed
    /// tensor bytes across all files). `0` means "derive from free device
    /// memory when the engine initializes" — see [`default_inflight_mb`].
    pub inflight_mb: usize,
}

impl Default for LoadConfig {
    fn default() -> Self {
        Self {
            chunk_mb: 16,
            workers: 24,
            slabs: 24,
            inflight_mb: 0,
        }
    }
}

/// Read-ahead budget chosen when the engine initializes unconfigured: a
/// sixteenth of the device's free memory, clamped to `[512 MiB, 8 GiB]`.
///
/// A fraction of *free* memory (not total) keeps staging proportional to
/// the headroom actually available on a GPU that is usually about to hold
/// the model being loaded, so the budget shrinks on a busy device instead
/// of competing with it. The floor is the historical default, below which
/// read-ahead stops covering the chunk pipeline; the ceiling is where the
/// measured gain flattens (74.8k-tensor NVFP4 warm: 2.45 s at 512 MiB,
/// 1.38 s at 8 GiB) and beyond which a load would reserve more transient
/// device memory than any consumer can drain usefully.
///
/// [`configure`] overrides it entirely. With no CUDA runtime (or a failing
/// `cudaMemGetInfo`) it degrades to the floor.
fn default_inflight_mb() -> usize {
    const MIN_MB: usize = 512;
    const MAX_MB: usize = 8 << 10;
    let Some(cuda) = cuda::api() else {
        return MIN_MB;
    };
    match cuda.mem_get_info() {
        Ok((free, _total)) => ((free >> 20) / 16).clamp(MIN_MB, MAX_MB),
        Err(_) => MIN_MB,
    }
}

static CONFIG: OnceLock<LoadConfig> = OnceLock::new();
static CONFIG_USED: AtomicBool = AtomicBool::new(false);

fn config() -> &'static LoadConfig {
    CONFIG_USED.store(true, Ordering::Release);
    CONFIG.get_or_init(LoadConfig::default)
}

/// The in-flight budget the engine actually runs with, in MiB: the
/// configured value, or the [`default_inflight_mb`] pick when unconfigured.
/// Reading it initializes the budget (and so freezes the pick).
pub fn effective_inflight_mb() -> usize {
    budget().limit() >> 20
}

/// Install the engine configuration. Fails if the engine already
/// initialized (lazily, on first CUDA load) or was already configured.
pub fn configure(cfg: LoadConfig) -> Result<(), String> {
    if cfg.chunk_mb == 0 || cfg.workers == 0 || cfg.slabs == 0 {
        return Err("chunk_mb, workers and slabs must be >= 1".to_string());
    }
    if CONFIG_USED.load(Ordering::Acquire) {
        return Err(
            "CUDA loading engine already initialized; configure it before the first load"
                .to_string(),
        );
    }
    CONFIG
        .set(cfg)
        .map_err(|_| "CUDA loading engine already configured".to_string())
}

/// A `cudaEvent_t` destroyed on drop.
pub struct OwnedEvent {
    cuda: &'static CudaApi,
    ev: Event,
}

// SAFETY: CUDA events are thread-safe handles.
unsafe impl Send for OwnedEvent {}
unsafe impl Sync for OwnedEvent {}

impl OwnedEvent {
    /// Create a sync-only event on the current device.
    pub fn new(cuda: &'static CudaApi) -> Result<Self, String> {
        Ok(Self {
            cuda,
            ev: cuda.event_create()?,
        })
    }

    /// Record this event on `stream`.
    pub fn record(&self, stream: Stream) -> Result<(), String> {
        self.cuda.event_record(self.ev, stream)
    }

    /// Block the host until this event completes.
    pub fn synchronize(&self) -> Result<(), String> {
        self.cuda.event_synchronize(self.ev)
    }

    /// Make all future work on `stream` wait for this event's most recent
    /// record.
    pub fn wait_on(&self, stream: Stream) -> Result<(), String> {
        self.cuda.stream_wait_event(stream, self.ev)
    }
}

impl Drop for OwnedEvent {
    fn drop(&mut self) {
        let _ = self.cuda.event_destroy(self.ev);
    }
}

struct Slab {
    ptr: usize,
    /// Event of the copy that last consumed this slab; `None` when the slab
    /// was never used or its copy was already synchronized.
    fence: Option<Arc<OwnedEvent>>,
}

/// Process-global ring of chunk-sized pinned host slabs (see module docs).
pub struct SlabPool {
    /// Slab (and therefore chunk) size in bytes.
    pub slab_size: usize,
    slabs: Mutex<VecDeque<Slab>>,
    available: Condvar,
}

impl SlabPool {
    fn new(cuda: &'static CudaApi, slab_size: usize, count: usize) -> Result<Self, String> {
        let mut q = VecDeque::with_capacity(count);
        for _ in 0..count {
            let ptr = cuda.host_alloc(slab_size, cuda::HOST_ALLOC_PORTABLE)?;
            q.push_back(Slab { ptr, fence: None });
        }
        Ok(Self {
            slab_size,
            slabs: Mutex::new(q),
            available: Condvar::new(),
        })
    }

    /// Pop a slab, waiting for one if all are lent out, and synchronize the
    /// fence of the copy that last used it before handing it over.
    pub fn acquire(&self) -> Result<usize, String> {
        let slab = {
            let mut q = self.slabs.lock().unwrap();
            loop {
                if let Some(slab) = q.pop_front() {
                    break slab;
                }
                q = self.available.wait(q).unwrap();
            }
        };
        if let Some(fence) = slab.fence {
            fence.synchronize()?;
        }
        Ok(slab.ptr)
    }

    /// Return a slab to the ring. `fence` is the event of the copy that
    /// consumed it (`None` if no copy was issued).
    pub fn release(&self, ptr: usize, fence: Option<Arc<OwnedEvent>>) {
        self.slabs.lock().unwrap().push_back(Slab { ptr, fence });
        self.available.notify_one();
    }
}

static POOL: OnceLock<Result<SlabPool, String>> = OnceLock::new();

/// The process-global slab ring, allocated once on first use per
/// [`LoadConfig`] (`chunk_mb` sets the slab size, `slabs` the ring depth).
pub fn pool(cuda: &'static CudaApi) -> Result<&'static SlabPool, String> {
    POOL.get_or_init(|| {
        let cfg = config();
        SlabPool::new(cuda, cfg.chunk_mb << 20, cfg.slabs)
    })
    .as_ref()
    .map_err(Clone::clone)
}

static LIVE_PREFETCHES: OnceLock<Mutex<HashMap<i32, usize>>> = OnceLock::new();

/// Register a live prefetch on `device`. While any is live, the default
/// mempool keeps freed blocks cached across syncs (release threshold
/// `u64::MAX`) so the per-tensor `cudaMallocAsync`/`cudaFreeAsync` churn of
/// a load is served from the pool instead of the driver.
pub fn pool_retain(cuda: &'static CudaApi, device: i32) {
    let mut live = LIVE_PREFETCHES
        .get_or_init(Default::default)
        .lock()
        .unwrap();
    let count = live.entry(device).or_insert(0);
    *count += 1;
    if *count == 1 {
        let _ = cuda.set_default_pool_release_threshold(device, u64::MAX);
    }
}

/// Unregister a live prefetch on `device`. When the last one closes, the
/// pool cache is trimmed to zero and the threshold restored, so every byte
/// of staging memory is visible as free to other allocators (torch/vLLM
/// KV-cache sizing) and later frees release at the next sync again.
pub fn pool_release_handle(cuda: &'static CudaApi, device: i32) {
    let mut live = LIVE_PREFETCHES
        .get_or_init(Default::default)
        .lock()
        .unwrap();
    let count = live.entry(device).or_insert(1);
    *count = count.saturating_sub(1);
    if *count == 0 {
        let _ = cuda.set_default_pool_release_threshold(device, 0);
        let _ = cuda.trim_default_pool(device);
    }
}

static STREAMS: OnceLock<Mutex<HashMap<i32, Stream>>> = OnceLock::new();

/// The dedicated copy stream for `device`, created on first use and kept
/// for the process lifetime — it outlives every DLPack capsule whose
/// deleter frees on it. The caller must have `device` current.
pub fn stream_for(cuda: &'static CudaApi, device: i32) -> Result<Stream, String> {
    let mut streams = STREAMS.get_or_init(Default::default).lock().unwrap();
    if let Some(&stream) = streams.get(&device) {
        return Ok(stream);
    }
    let stream = cuda.stream_create(cuda::STREAM_NON_BLOCKING)?;
    streams.insert(device, stream);
    Ok(stream)
}

/// One fixed-size piece of the file's tensor-data section.
pub struct Chunk {
    /// Absolute file offset of the first byte.
    pub file_off: u64,
    /// Offset within the tensor-data section.
    pub data_off: usize,
    /// Length in bytes (`<= chunk_size`; the tail chunk may be shorter).
    pub len: usize,
}

/// Split `[data_start, data_start + data_len)` into `chunk_size`d [`Chunk`]s.
pub fn plan_chunks(data_start: u64, data_len: usize, chunk_size: usize) -> Vec<Chunk> {
    let mut chunks = Vec::with_capacity(data_len.div_ceil(chunk_size.max(1)));
    let mut off = 0usize;
    while off < data_len {
        let len = chunk_size.min(data_len - off);
        chunks.push(Chunk {
            file_off: data_start + off as u64,
            data_off: off,
            len,
        });
        off += len;
    }
    chunks
}

/// Process-global budget for prefetch in-flight bytes: destination tensors
/// allocated ahead of consumption but not yet delivered to the caller
/// ([`LoadConfig::inflight_mb`]). Pinned memory is already bounded by the
/// slab ring, so this bounds device-side transients when many files are
/// opened with `prefetch=True` at once.
///
/// The budget never blocks and bounds only *opportunistic* read-ahead:
/// refills simply stop widening when it is exhausted (backpressure is
/// "stop issuing", so a consumer thread can never deadlock on it).
/// Consumer-demanded (forced) allocations bypass the counter entirely —
/// counting them would starve every other file's pipelining while a
/// consumer drains its lookahead window; their overshoot is structurally
/// bounded by the per-consumer lookahead plus one oversized tensor. The
/// forced/opportunistic split also orders the I/O itself: forced work
/// rides the engine queue's front lane.
pub struct Budget {
    limit: usize,
    used: Mutex<usize>,
}

impl Budget {
    /// Reserve `n` bytes if they fit (or nothing else is in flight, so one
    /// oversized tensor can still enter an empty window).
    pub fn try_acquire(&self, n: usize) -> bool {
        let mut used = self.used.lock().unwrap();
        if *used + n <= self.limit || *used == 0 {
            *used += n;
            true
        } else {
            false
        }
    }

    /// Return `n` reserved bytes.
    pub fn release(&self, n: usize) {
        let mut used = self.used.lock().unwrap();
        *used = used.saturating_sub(n);
    }

    /// Currently reserved bytes (for diagnostics).
    pub fn in_flight(&self) -> usize {
        *self.used.lock().unwrap()
    }

    /// The configured ceiling in bytes.
    pub fn limit(&self) -> usize {
        self.limit
    }
}

static BUDGET: OnceLock<Budget> = OnceLock::new();

/// The process-global prefetch budget. Initialized on first use; the
/// caller must have the loading device current so an unconfigured budget
/// derives from the right device's free memory (see
/// [`default_inflight_mb`]).
pub fn budget() -> &'static Budget {
    BUDGET.get_or_init(|| {
        let mb = match config().inflight_mb {
            0 => default_inflight_mb(),
            mb => mb,
        };
        Budget {
            limit: mb << 20,
            used: Mutex::new(0),
        }
    })
}

/// Byte span of one tensor within the data section, plus the chunk range
/// covering it. Empty tensors carry an empty chunk range.
pub struct TensorSpan {
    /// First byte within the data section.
    pub begin: usize,
    /// One past the last byte.
    pub end: usize,
    /// Index of the first covering chunk (meaningless when `begin == end`).
    pub first_chunk: usize,
    /// Index of the last covering chunk (meaningless when `begin == end`).
    pub last_chunk: usize,
}

/// Compute [`TensorSpan`]s from offset-ordered `(begin, end)` pairs.
pub fn tensor_spans(offsets: &[(usize, usize)], chunk_size: usize) -> Vec<TensorSpan> {
    offsets
        .iter()
        .map(|&(begin, end)| TensorSpan {
            begin,
            end,
            first_chunk: begin / chunk_size,
            last_chunk: if end > begin {
                (end - 1) / chunk_size
            } else {
                0
            },
        })
        .collect()
}

/// One unit of engine work: read one chunk of one file and scatter it.
/// Holding the `Arc` keeps the state struct alive while queued; a
/// cancelled state's items are refused at claim time.
struct WorkItem {
    state: Arc<PrefetchState>,
    chunk: usize,
}

struct Lanes {
    /// Consumer-demanded work: strictly served before read-ahead.
    forced: VecDeque<WorkItem>,
    /// Opportunistic read-ahead.
    opportunistic: VecDeque<WorkItem>,
}

/// The process-global loading engine: [`LoadConfig::workers`] threads
/// serving every open file through the two-lane queue. Idle workers top up
/// the windows of live files in open order (see module docs).
pub struct Engine {
    lanes: Mutex<Lanes>,
    work_cv: Condvar,
    /// Live prefetch states in open order, for idle window top-ups.
    registry: Mutex<Vec<Weak<PrefetchState>>>,
}

static ENGINE: OnceLock<&'static Engine> = OnceLock::new();

/// The process-global engine, spawning its worker threads on first use.
pub fn engine(cuda: &'static CudaApi) -> &'static Engine {
    ENGINE.get_or_init(|| {
        let _ = cuda; // reserved: engine is per-process, CUDA api is global
        let engine: &'static Engine = Box::leak(Box::new(Engine {
            lanes: Mutex::new(Lanes {
                forced: VecDeque::new(),
                opportunistic: VecDeque::new(),
            }),
            work_cv: Condvar::new(),
            registry: Mutex::new(Vec::new()),
        }));
        for _ in 0..config().workers {
            std::thread::spawn(move || engine.worker_loop());
        }
        engine
    })
}

impl Engine {
    /// Track a new file's state for idle window top-ups (open order).
    fn register(&self, state: &Arc<PrefetchState>) {
        self.registry.lock().unwrap().push(Arc::downgrade(state));
    }

    /// Queue chunk descriptors; forced work jumps the read-ahead lane.
    fn push(&self, items: Vec<WorkItem>, forced: bool) {
        if items.is_empty() {
            return;
        }
        let mut lanes = self.lanes.lock().unwrap();
        if forced {
            lanes.forced.extend(items);
        } else {
            lanes.opportunistic.extend(items);
        }
        drop(lanes);
        self.work_cv.notify_all();
    }

    /// Drop every queued-but-unstarted descriptor of `state` (handle
    /// close); items already claimed by a worker run to completion and are
    /// awaited via the state's in-flight counter.
    fn purge(&self, state: &Arc<PrefetchState>) {
        let mut lanes = self.lanes.lock().unwrap();
        lanes.forced.retain(|it| !Arc::ptr_eq(&it.state, state));
        lanes
            .opportunistic
            .retain(|it| !Arc::ptr_eq(&it.state, state));
    }

    /// Wake idle workers (budget was freed; windows may widen).
    fn nudge(&self) {
        self.work_cv.notify_all();
    }

    /// Top up the windows of live files in open order; prunes dead states.
    fn sweep_refill(&self) {
        let states: Vec<Arc<PrefetchState>> = {
            let mut registry = self.registry.lock().unwrap();
            registry.retain(|w| w.strong_count() > 0);
            registry.iter().filter_map(Weak::upgrade).collect()
        };
        for state in states {
            let _ = PrefetchState::try_refill(&state, None);
        }
    }

    fn worker_loop(&'static self) {
        loop {
            let item = {
                let mut lanes = self.lanes.lock().unwrap();
                loop {
                    if let Some(item) = lanes
                        .forced
                        .pop_front()
                        .or_else(|| lanes.opportunistic.pop_front())
                    {
                        break Some(item);
                    }
                    drop(lanes);
                    // Idle: top up live windows (may queue work), then wait.
                    // The wait is timed so budget freed without a nudge
                    // (e.g. tensors dropped by GC) is still picked up.
                    self.sweep_refill();
                    lanes = self.lanes.lock().unwrap();
                    if lanes.forced.is_empty() && lanes.opportunistic.is_empty() {
                        let (guard, _) = self
                            .work_cv
                            .wait_timeout(lanes, Duration::from_millis(100))
                            .unwrap();
                        lanes = guard;
                    }
                }
            };
            if let Some(item) = item {
                self.process(item);
            }
        }
    }

    fn process(&self, item: WorkItem) {
        let state = item.state;
        if !state.begin_chunk(item.chunk) {
            return; // cancelled, or a duplicate of an already-claimed chunk
        }
        let result = state.load_chunk(item.chunk);
        state.finish_chunk(result);
        // Top up every live file's window as chunks complete (open order):
        // freed budget must flow to not-yet-consumed files DURING a load,
        // not only when the queue drains — several files streaming at once
        // is where the old per-handle-thread engine got its depth (worth
        // 10-15% cold on multi-shard loads).
        drop(state);
        self.sweep_refill();
    }
}

struct Progress {
    /// Tensors allocated so far (allocation happens in offset order).
    tensors_alloced: usize,
}

struct DoneState {
    /// Per-chunk fence event, set when the chunk's copies are issued.
    events: Vec<Option<Arc<OwnedEvent>>>,
    /// Chunk indices in copy-enqueue order: the order in which workers took
    /// the `issue` lock and pushed their `cudaMemcpyAsync`es onto the
    /// per-device copy stream. The stream is FIFO, so this is also the
    /// order the chunks' events fire in — the backbone of
    /// [`StreamCursor`], which walks it position by position.
    issue_order: Vec<usize>,
    error: Option<String>,
}

/// Allocation cursor, guarded by one mutex so refill batches serialize.
struct AllocCursor {
    /// Next tensor (offset order) without a destination.
    next_tensor: usize,
    /// Chunks handed to the engine queue so far (monotone prefix).
    chunks_queued: usize,
}

/// An owned stream-ordered device allocation, freed on drop in stream
/// order (`cudaFreeAsync`). This is what rides inside a DLPack capsule's
/// manager context, so the deleter can run on any thread (GC, another
/// device current): the drop sets the owning device first. The per-device
/// copy stream is process-lived, so it outlives every capsule.
pub struct AsyncDeviceBuf {
    cuda: &'static CudaApi,
    ptr: usize,
    device: i32,
    stream: Stream,
}

// SAFETY: `ptr` is device memory only touched through the CUDA API, which
// is thread-safe; the fields themselves are plain values.
unsafe impl Send for AsyncDeviceBuf {}
unsafe impl Sync for AsyncDeviceBuf {}

impl crate::dlpack::AsDevicePtr for AsyncDeviceBuf {
    fn as_device_ptr(&self) -> *mut std::ffi::c_void {
        self.ptr as *mut std::ffi::c_void
    }
}

impl Drop for AsyncDeviceBuf {
    fn drop(&mut self) {
        // Best-effort: during interpreter teardown the CUDA context may
        // already be gone.
        if let Ok(_guard) = self.cuda.device_guard(self.device) {
            let _ = self.cuda.free_device(self.ptr, self.stream);
        }
    }
}

/// Shared state of one file's prefetch: the chunk plan, per-tensor
/// destination pointers, per-chunk claim flags and completion events, and
/// failure/cancel flags. All I/O runs on the process-global [`Engine`];
/// this struct only describes the file and tracks its progress.
///
/// The pipeline is GIL-free. Window refills ([`try_refill`](Self::try_refill))
/// run on engine workers (idle sweeps) and on detached consumer threads:
/// stream-ordered-allocate destinations, then queue the newly covered
/// chunks. Consumers wait per tensor on the events of every covering chunk
/// (chunks complete out of order, so the last chunk's event alone is not a
/// readiness proof) and receive the buffer itself via
/// [`take_buf`](Self::take_buf); framework wrapping (DLPack) happens on
/// the consumer's own thread.
pub struct PrefetchState {
    cuda: &'static CudaApi,
    device: i32,
    stream: Stream,
    file: Arc<File>,
    chunks: Vec<Chunk>,
    spans: Vec<TensorSpan>,
    /// Device data pointer per tensor; published before the covering
    /// chunks are queued, swapped to 0 when the buffer is handed over.
    dest: Vec<AtomicUsize>,
    /// Whether tensor `t`'s bytes were counted against the budget
    /// (opportunistic) or forced past it (consumer-demanded lookahead).
    counted: Vec<AtomicBool>,
    /// Per-chunk claim: a chunk is processed exactly once even when queued
    /// twice (lane promotion pushes duplicates).
    claimed: Vec<AtomicBool>,
    alloc: Mutex<AllocCursor>,
    /// Highest tensor index a consumer take has already forced (allocated,
    /// queued front-lane, promoted). Takes below it skip the forced-refill
    /// and promotion probes entirely — at tiny-tensor cadence those probes
    /// dominated (74.8k-tensor NVFP4: 3.7s vs 1.7s warm).
    forced_upto: AtomicUsize,
    /// Work items this file pushed onto the engine's forced lane. A pure
    /// [`StreamCursor`] drain consumes in completion order and must never
    /// need the forced lane, so this stays 0 there; random-access
    /// [`take_buf`](Self::take_buf) and the cursor's liveness escape hatch
    /// are what move it.
    forced_pushes: AtomicUsize,
    /// Budget bytes released by deliveries since the last window refill;
    /// see [`take_completed`](Self::take_completed).
    released: AtomicUsize,
    /// Orders the copy stream after the consumer's (legacy default) stream
    /// at each allocation batch: the pool can reuse a freed block whose
    /// previous owner still has reads in flight there.
    fence: OwnedEvent,
    progress: Mutex<Progress>,
    progress_cv: Condvar,
    done: Mutex<DoneState>,
    done_cv: Condvar,
    /// Serializes each chunk's (memcpys, record) pair on the shared stream.
    issue: Mutex<()>,
    /// Chunks claimed by a worker and not yet finished.
    inflight: Mutex<usize>,
    inflight_cv: Condvar,
    cancel: AtomicBool,
}

impl PrefetchState {
    /// Build the state for one file's prefetch and register it with the
    /// engine. `fence` must have been created with `device` current. The
    /// initial window is opened immediately: I/O starts here.
    pub fn start(
        cuda: &'static CudaApi,
        device: i32,
        stream: Stream,
        file: Arc<File>,
        chunks: Vec<Chunk>,
        spans: Vec<TensorSpan>,
        fence: OwnedEvent,
    ) -> Result<Arc<Self>, String> {
        let n_chunks = chunks.len();
        let n_tensors = spans.len();
        let state = Arc::new(Self {
            cuda,
            device,
            stream,
            file,
            chunks,
            spans,
            dest: (0..n_tensors).map(|_| AtomicUsize::new(0)).collect(),
            counted: (0..n_tensors).map(|_| AtomicBool::new(false)).collect(),
            claimed: (0..n_chunks).map(|_| AtomicBool::new(false)).collect(),
            alloc: Mutex::new(AllocCursor {
                next_tensor: 0,
                chunks_queued: 0,
            }),
            forced_upto: AtomicUsize::new(0),
            forced_pushes: AtomicUsize::new(0),
            released: AtomicUsize::new(0),
            fence,
            progress: Mutex::new(Progress { tensors_alloced: 0 }),
            progress_cv: Condvar::new(),
            done: Mutex::new(DoneState {
                events: (0..n_chunks).map(|_| None).collect(),
                issue_order: Vec::with_capacity(n_chunks),
                error: None,
            }),
            done_cv: Condvar::new(),
            issue: Mutex::new(()),
            inflight: Mutex::new(0),
            inflight_cv: Condvar::new(),
            cancel: AtomicBool::new(false),
        });
        engine(cuda).register(&state);
        Self::try_refill(&state, None)?;
        Ok(state)
    }

    /// Stop the engine from starting more of this file's work and wake all
    /// state waiters.
    fn cancel(&self) {
        // Under the inflight lock so a concurrent claim either completes
        // before the flag (and is awaited) or observes it (and is refused).
        let _guard = self.inflight.lock().unwrap();
        self.cancel.store(true, Ordering::Relaxed);
        self.progress_cv.notify_all();
        self.done_cv.notify_all();
    }

    fn cancelled(&self) -> bool {
        self.cancel.load(Ordering::Relaxed)
    }

    /// Number of tensors whose destinations are published (allocation is
    /// strictly in offset order).
    fn tensors_allocated(&self) -> usize {
        self.progress.lock().unwrap().tensors_alloced
    }

    /// Publish tensor `t`'s device destination pointer. Called in offset
    /// order under the `alloc` lock; the covering chunks stay unqueued
    /// until the refill batch ends.
    fn publish_dest(&self, t: usize, ptr: usize) {
        self.dest[t].store(ptr, Ordering::Release);
        let mut progress = self.progress.lock().unwrap();
        debug_assert_eq!(progress.tensors_alloced, t);
        progress.tensors_alloced = t + 1;
        drop(progress);
        self.progress_cv.notify_all();
    }

    /// Widen the allocation window: stream-ordered-allocate destinations in
    /// offset order while the budget admits them (unconditionally through
    /// `force_until` — the consumer demanded that tensor, so capped
    /// read-ahead must not become a refused read), then fence and queue the
    /// newly covered chunks with the engine (front lane when forced).
    /// GIL-free; runs on engine workers and detached consumer threads.
    pub fn try_refill(self: &Arc<Self>, force_until: Option<usize>) -> Result<(), String> {
        let budget = budget();
        let mut cursor = self.alloc.lock().unwrap();
        if cursor.next_tensor >= self.spans.len() && cursor.chunks_queued >= self.chunks.len() {
            return Ok(());
        }
        // The device guard costs CUDA calls; take it only once there is
        // actual allocation work (takes probe this path at tensor cadence).
        let mut guard = None;
        let mut allocated_any = false;
        let mut result = Ok(());
        while cursor.next_tensor < self.spans.len() && !self.cancelled() {
            let t = cursor.next_tensor;
            let nbytes = self.spans[t].end - self.spans[t].begin;
            let counted = if force_until.is_some_and(|idx| t <= idx) {
                false // consumer-demanded: budget-exempt
            } else if budget.try_acquire(nbytes) {
                true
            } else {
                break; // window edge: backpressure is "stop issuing"
            };
            if guard.is_none() {
                match self.cuda.device_guard(self.device) {
                    Ok(g) => guard = Some(g),
                    Err(e) => {
                        if counted {
                            budget.release(nbytes);
                        }
                        self.fail(e.clone());
                        return Err(e);
                    }
                }
            }
            // Zero-byte tensors get a 1-byte clamp so the pointer is valid.
            let alloc = match self.cuda.alloc_device(nbytes.max(1), self.stream) {
                Ok(ptr) => Ok(ptr),
                // Consumer-demanded: the caller is blocked on this tensor,
                // so make one real effort to find room before giving up.
                Err(e) if !counted => self.alloc_after_reclaim(nbytes.max(1), &e),
                Err(e) => Err(e),
            };
            match alloc {
                Ok(ptr) => {
                    self.counted[t].store(counted, Ordering::Release);
                    self.publish_dest(t, ptr);
                    cursor.next_tensor = t + 1;
                    allocated_any = true;
                }
                Err(e) => {
                    if counted {
                        // Opportunistic lane: silent backpressure, exactly
                        // like an exhausted budget. Read-ahead nobody asked
                        // for must never surface as an error; the window
                        // stops widening and the next refill (chunk
                        // completion, idle sweep, consumer take) retries as
                        // device memory frees.
                        budget.release(nbytes);
                        break;
                    }
                    result = Err(e);
                    break;
                }
            }
        }
        if allocated_any {
            debug_assert!(guard.is_some());
            if let Err(e) = self
                .fence
                .record(0)
                .and_then(|()| self.fence.wait_on(self.stream))
            {
                self.fail(e.clone());
                return result.and(Err(e));
            }
        }
        // Queue every chunk now fully covered by allocated tensors.
        let boundary = match self.spans.get(cursor.next_tensor) {
            Some(span) => span.begin,
            None => usize::MAX,
        };
        let mut items = Vec::new();
        while cursor.chunks_queued < self.chunks.len() {
            let c = cursor.chunks_queued;
            if self.chunks[c].data_off + self.chunks[c].len > boundary {
                break;
            }
            items.push(WorkItem {
                state: self.clone(),
                chunk: c,
            });
            cursor.chunks_queued = c + 1;
        }
        if force_until.is_some() {
            self.forced_pushes
                .fetch_add(1 + items.len(), Ordering::Relaxed);
        }
        engine(self.cuda).push(items, force_until.is_some());
        if let Err(e) = &result {
            self.fail(e.clone());
        }
        result
    }

    /// Last-ditch allocation for a consumer-demanded tensor: retire every
    /// pending stream-ordered free on the copy stream (delivered tensors
    /// the consumer already dropped free asynchronously), hand the
    /// mempool's cache back to the driver, then retry once. A second
    /// failure names the knob that bounds this engine's own footprint and
    /// how many bytes it is holding, instead of a bare
    /// `cudaErrorMemoryAllocation`.
    fn alloc_after_reclaim(&self, nbytes: usize, first: &str) -> Result<usize, String> {
        let _ = self.cuda.stream_synchronize(self.stream);
        let _ = self.cuda.trim_default_pool(self.device);
        self.cuda
            .alloc_device(nbytes, self.stream)
            .map_err(|retry| {
                let mib = |b: usize| b as f64 / (1usize << 20) as f64;
                let budget = budget();
                format!(
                    "out of device memory allocating {:.1} MiB for a requested tensor on cuda:{} \
                 ({retry}; first attempt: {first}). The loader is holding {:.0} MiB of \
                 read-ahead against a {:.0} MiB budget: lower it with \
                 safetensors.configure_cuda_loading(inflight_mb=...) before the first load \
                 of the process, or free device memory before loading.",
                    mib(nbytes),
                    self.device,
                    mib(budget.in_flight()),
                    mib(budget.limit()),
                )
            })
    }

    /// The tensor through which a consumer take forces allocation: at
    /// least tensor `t`'s full chunk coverage, extended to a byte horizon
    /// (`FORCE_LOOKAHEAD` past `t`) so the file being consumed always has a
    /// guaranteed read-ahead window — with many files open, opportunistic
    /// budget is contended and can starve the consumed file down to
    /// chunk-latency lockstep (measured 8.2 vs 13.1 GB/s on an 18.9k-tensor
    /// MoE).
    fn force_horizon(&self, t: usize) -> usize {
        /// Guaranteed per-consumer read-ahead (bytes, budget-exempt).
        const FORCE_LOOKAHEAD: usize = 256 << 20;
        let horizon = self.spans[t].end + FORCE_LOOKAHEAD;
        let by_bytes = self.spans.partition_point(|s| s.begin < horizon);
        self.force_target((by_bytes.max(t + 1) - 1).min(self.spans.len() - 1))
    }

    /// Move any already-queued-but-unclaimed chunks covering
    /// `[t..=target]` to the front lane by pushing duplicates (claims
    /// dedupe), so a consumer never waits behind other files' read-ahead.
    fn promote(self: &Arc<Self>, t: usize, target: usize) {
        let span = &self.spans[t];
        if span.begin == span.end {
            return;
        }
        let last = self.spans[target].last_chunk.max(span.last_chunk);
        let queued = self.alloc.lock().unwrap().chunks_queued;
        let mut items = Vec::new();
        for c in span.first_chunk..=last.min(queued.saturating_sub(1)) {
            if !self.claimed[c].load(Ordering::Acquire) {
                items.push(WorkItem {
                    state: self.clone(),
                    chunk: c,
                });
            }
        }
        self.forced_pushes.fetch_add(items.len(), Ordering::Relaxed);
        engine(self.cuda).push(items, true);
    }

    /// Work items this file has pushed onto the engine's forced lane; see
    /// [`forced_pushes`](Self::forced_pushes).
    pub fn forced_pushes(&self) -> usize {
        self.forced_pushes.load(Ordering::Relaxed)
    }

    /// Deliver tensor `t`'s device buffer: force the window through its
    /// covering chunks plus a read-ahead horizon, block until its bytes are
    /// resident, then hand over ownership (freed on drop, in stream order).
    /// `None` means it was already delivered once.
    pub fn take_buf(self: &Arc<Self>, t: usize) -> Result<Option<AsyncDeviceBuf>, String> {
        if self.dest[t].load(Ordering::Acquire) == 0 && self.tensors_allocated() > t {
            return Ok(None);
        }
        // A refill failure past `t` must not block delivering `t` itself;
        // wait_tensor reports the failure if it actually affects `t`.
        let target = self.force_horizon(t);
        // `forced_upto` is exclusive: a previous take already forced and
        // promoted through `forced_upto - 1`.
        if self.forced_upto.load(Ordering::Acquire) <= target || self.tensors_allocated() <= target
        {
            let _ = Self::try_refill(self, Some(target));
            self.promote(t, target);
            self.forced_upto.fetch_max(target + 1, Ordering::AcqRel);
        }
        self.wait_tensor(t)?;
        let ptr = self.dest[t].swap(0, Ordering::AcqRel);
        if ptr == 0 {
            return Ok(None);
        }
        if self.counted[t].swap(false, Ordering::AcqRel) {
            budget().release(self.spans[t].end - self.spans[t].begin);
        }
        // Freed budget: widen own window, then wake idle workers for the
        // rest.
        let _ = Self::try_refill(self, None);
        engine(self.cuda).nudge();
        Ok(Some(AsyncDeviceBuf {
            cuda: self.cuda,
            ptr,
            device: self.device,
            stream: self.stream,
        }))
    }

    /// Shut this file's loading down: refuse new claims, purge queued
    /// descriptors, await claimed chunks, sync every recorded event, then
    /// free undelivered destinations. Safe against a worker mid-pread into
    /// a slab destined for these tensors: the in-flight wait covers it.
    pub fn close(self: &Arc<Self>) {
        self.cancel();
        engine(self.cuda).purge(self);
        {
            let mut inflight = self.inflight.lock().unwrap();
            while *inflight > 0 {
                inflight = self.inflight_cv.wait(inflight).unwrap();
            }
        }
        self.sync_all_events();
        self.reclaim_undelivered();
        engine(self.cuda).nudge();
    }

    /// Free every allocated-but-undelivered destination and drop this
    /// handle's pool retention (the last close per device trims the pool
    /// cache back to the driver).
    fn reclaim_undelivered(&self) {
        let budget = budget();
        let cursor = self.alloc.lock().unwrap();
        let Ok(_guard) = self.cuda.device_guard(self.device) else {
            return;
        };
        for t in 0..cursor.next_tensor {
            let ptr = self.dest[t].swap(0, Ordering::AcqRel);
            if ptr != 0 {
                let _ = self.cuda.free_device(ptr, self.stream);
                if self.counted[t].swap(false, Ordering::AcqRel) {
                    budget.release(self.spans[t].end - self.spans[t].begin);
                }
            }
        }
        pool_release_handle(self.cuda, self.device);
    }

    /// The tensor through which allocation must proceed so every chunk
    /// covering tensor `t` can be queued (a chunk is queued only once all
    /// its overlapping tensors have destinations).
    fn force_target(&self, t: usize) -> usize {
        let span = &self.spans[t];
        if span.begin == span.end {
            return t;
        }
        let chunk = &self.chunks[span.last_chunk];
        let chunk_end = chunk.data_off + chunk.len;
        // Index of the last tensor starting before the chunk's end (at
        // least `t` itself).
        self.spans
            .partition_point(|s| s.begin < chunk_end)
            .max(t + 1)
            - 1
    }

    /// Record a failure (first error wins) and stop everything.
    fn fail(&self, e: String) {
        {
            let mut done = self.done.lock().unwrap();
            if done.error.is_none() {
                done.error = Some(e);
            }
        }
        self.cancel();
    }

    fn stop_reason(&self) -> String {
        self.done
            .lock()
            .unwrap()
            .error
            .clone()
            .unwrap_or_else(|| "file was closed while tensors were still loading".to_string())
    }

    /// Block until tensor `t`'s destination is allocated and every covering
    /// chunk's copies have completed on the device.
    pub fn wait_tensor(&self, t: usize) -> Result<(), String> {
        {
            let mut progress = self.progress.lock().unwrap();
            while progress.tensors_alloced <= t {
                if self.cancelled() {
                    return Err(self.stop_reason());
                }
                progress = self.progress_cv.wait(progress).unwrap();
            }
        }
        let span = &self.spans[t];
        if span.begin == span.end {
            return Ok(());
        }
        for c in span.first_chunk..=span.last_chunk {
            let event = {
                let mut done = self.done.lock().unwrap();
                loop {
                    if let Some(e) = &done.events[c] {
                        break e.clone();
                    }
                    if self.cancelled() {
                        return Err(self.stop_reason());
                    }
                    done = self.done_cv.wait(done).unwrap();
                }
            };
            event.synchronize()?;
        }
        Ok(())
    }

    /// Synchronize every recorded chunk event so device buffers can be
    /// freed with no copy in flight.
    fn sync_all_events(&self) {
        let events: Vec<Arc<OwnedEvent>> = {
            let done = self.done.lock().unwrap();
            done.events.iter().flatten().cloned().collect()
        };
        for e in events {
            let _ = e.synchronize();
        }
    }

    /// Claim chunk `c` for processing; refuses duplicates and cancelled
    /// states, and counts the claim in the in-flight counter awaited by
    /// [`close`](Self::close).
    fn begin_chunk(&self, c: usize) -> bool {
        let mut inflight = self.inflight.lock().unwrap();
        if self.cancelled() || self.claimed[c].swap(true, Ordering::AcqRel) {
            return false;
        }
        *inflight += 1;
        true
    }

    /// Release chunk `c`'s in-flight claim. The success path already
    /// published the chunk's event and its position in the copy-enqueue
    /// order from inside [`load_chunk`](Self::load_chunk)'s issue section.
    fn finish_chunk(&self, result: Result<(), String>) {
        if let Err(e) = result {
            self.fail(e);
        }
        let mut inflight = self.inflight.lock().unwrap();
        *inflight -= 1;
        drop(inflight);
        self.inflight_cv.notify_all();
    }

    /// Read chunk `c` into a pinned slab and scatter it to the overlapping
    /// tensors' device allocations, publishing its fence event and its
    /// position in the copy-enqueue order. Runs on an engine worker with no
    /// claim on any lock.
    fn load_chunk(&self, c: usize) -> Result<(), String> {
        let _guard = self.cuda.device_guard(self.device)?;
        let pool = pool(self.cuda)?;
        let chunk = &self.chunks[c];
        let event = Arc::new(OwnedEvent::new(self.cuda)?);
        let slab = pool.acquire()?;
        // SAFETY: `slab` names `slab_size >= chunk.len` bytes of pinned
        // host memory exclusively leased to this worker.
        let buf = unsafe { std::slice::from_raw_parts_mut(slab as *mut u8, chunk.len) };
        if let Err(e) = read_exact_at(&self.file, buf, chunk.file_off) {
            pool.release(slab, None);
            return Err(format!("pread of chunk at {}: {e}", chunk.file_off));
        }
        {
            let _issue_guard = self.issue.lock().unwrap();
            for (dst, slab_off, len) in self.copy_ops(c) {
                if let Err(e) = self
                    .cuda
                    .memcpy_h2d_async(dst, slab + slab_off, len, self.stream)
                {
                    pool.release(slab, None);
                    return Err(e);
                }
            }
            // On record failure the copies were issued with no fence;
            // retiring the slab could let another worker overwrite it
            // mid-copy, so `?` leaks it from the ring instead (the load is
            // failing anyway).
            event.record(self.stream)?;
            // Still inside the issue section: appending here is what makes
            // `issue_order` the copy stream's own FIFO order, so position
            // `k`'s event is guaranteed to fire before position `k + 1`'s.
            let mut done = self.done.lock().unwrap();
            done.events[c] = Some(event.clone());
            done.issue_order.push(c);
            drop(done);
            self.done_cv.notify_all();
        }
        pool.release(slab, Some(event));
        Ok(())
    }

    /// The plan's chunk size (every chunk but the tail is full-sized).
    fn chunk_size(&self) -> usize {
        self.chunks.first().map_or(1, |c| c.len)
    }

    /// Index range of the tensors whose bytes overlap chunk `c`. Spans are
    /// offset-ordered and non-overlapping, so both `begin` and `end` are
    /// sorted and the overlap set is one contiguous run. Zero-byte tensors
    /// can fall inside the range; they own none of the chunk's bytes.
    fn covered_tensors(&self, c: usize) -> std::ops::Range<usize> {
        let c0 = self.chunks[c].data_off;
        let c1 = c0 + self.chunks[c].len;
        self.spans.partition_point(|s| s.end <= c0)..self.spans.partition_point(|s| s.begin < c1)
    }

    /// Copy ops for chunk `i`: `(device dst, offset within slab, len)` per
    /// overlapping tensor.
    fn copy_ops(&self, i: usize) -> Vec<(usize, usize, usize)> {
        let c0 = self.chunks[i].data_off;
        let c1 = c0 + self.chunks[i].len;
        let range = self.covered_tensors(i);
        let mut ops = Vec::with_capacity(range.len());
        for idx in range {
            let span = &self.spans[idx];
            if span.begin == span.end {
                continue;
            }
            let ov0 = span.begin.max(c0);
            let ov1 = span.end.min(c1);
            let base = self.dest[idx].load(Ordering::Acquire);
            ops.push((base + (ov0 - span.begin), ov0 - c0, ov1 - ov0));
        }
        ops
    }

    /// Hand over tensor `t`'s device buffer with no forcing and no waiting:
    /// only [`StreamCursor`] calls this, and only once it has synchronized
    /// every chunk covering `t`. `None` means the tensor was already
    /// delivered (a `get_tensor` raced the stream for it).
    fn take_completed(self: &Arc<Self>, t: usize) -> Option<AsyncDeviceBuf> {
        let ptr = self.dest[t].swap(0, Ordering::AcqRel);
        if ptr == 0 {
            return None;
        }
        if self.counted[t].swap(false, Ordering::AcqRel) {
            let nbytes = self.spans[t].end - self.spans[t].begin;
            budget().release(nbytes);
            // Freed budget reaches the readers only through a refill.
            // Chunk completions sweep every live window, but when the
            // consumer has drained ahead of the readers nothing is in
            // flight to trigger one and the pipeline waits out an idle
            // worker's sweep. Probing per tensor is what dominated the
            // 74.8k-tensor warm load, so amortize it over a chunk's worth
            // of delivered bytes: big-tensor files refill on every take,
            // tiny-tensor files once per chunk.
            let freed = self.released.fetch_add(nbytes, Ordering::Relaxed) + nbytes;
            if freed >= self.chunk_size() {
                self.released.store(0, Ordering::Relaxed);
                let _ = Self::try_refill(self, None);
                engine(self.cuda).nudge();
            }
        }
        Some(AsyncDeviceBuf {
            cuda: self.cuda,
            ptr,
            device: self.device,
            stream: self.stream,
        })
    }

    /// The chunk at position `pos` of the copy-enqueue order, or `None` if
    /// nothing reached that position within `timeout`.
    fn issued_at(&self, pos: usize, timeout: Duration) -> Result<Option<usize>, String> {
        let mut done = self.done.lock().unwrap();
        loop {
            if let Some(&c) = done.issue_order.get(pos) {
                return Ok(Some(c));
            }
            if self.cancelled() {
                return Err(self.stop_reason());
            }
            let (guard, wait) = self.done_cv.wait_timeout(done, timeout).unwrap();
            done = guard;
            if wait.timed_out() {
                return Ok(done.issue_order.get(pos).copied());
            }
        }
    }

    /// True when nothing this file has queued is still waiting to be issued
    /// while chunks remain unqueued: the copy stream cannot produce another
    /// completion until the allocation window widens.
    fn window_starved(&self) -> bool {
        let queued = self.alloc.lock().unwrap().chunks_queued;
        queued < self.chunks.len() && self.done.lock().unwrap().issue_order.len() >= queued
    }

    /// Synchronize chunk `c`'s fence. Position `k` of the copy-enqueue
    /// order fires no later than position `k + 1`, so a cursor walking the
    /// order in sequence blocks exactly once per chunk, always on the event
    /// that is next to complete.
    fn sync_chunk(&self, c: usize) -> Result<(), String> {
        let event = {
            let mut done = self.done.lock().unwrap();
            loop {
                if let Some(e) = &done.events[c] {
                    break e.clone();
                }
                if self.cancelled() {
                    return Err(self.stop_reason());
                }
                done = self.done_cv.wait(done).unwrap();
            }
        };
        event.synchronize()
    }
}

/// Completion-order cursor over one file's tensors: the reading half of
/// `safe_open(..., prefetch=True).tensor_stream()`.
///
/// The copy stream is FIFO per device, so chunks complete in the order
/// their copies were enqueued ([`DoneState::issue_order`]). The cursor
/// walks that order one position at a time, blocking on the event that is
/// next to fire — no polling, no callbacks, no event queries — and yields
/// each tensor as its last covering chunk lands. Consumption therefore
/// never sits on tensor *n* while *n+1..n+k* are already resident, which is
/// what offset-ordered consumption did: it idled on the head of the line
/// and then had to force the very chunk it was waiting for onto the
/// engine's front lane, churning the queue against every other file's
/// read-ahead.
///
/// Because it only ever consumes what the stream already produced, a pure
/// drain issues no forced work at all (see
/// [`PrefetchState::forced_pushes`]). The one exception is liveness: if the
/// global budget is held by *other* files' undelivered read-ahead, this
/// file's window cannot widen on its own, so a starved cursor falls back to
/// a budget-exempt forced refill — the same escape hatch a random-access
/// `get_tensor` takes.
pub struct StreamCursor {
    state: Arc<PrefetchState>,
    /// Next position in the copy-enqueue order to synchronize.
    pos: usize,
    /// Covering chunks not yet known-complete, per tensor.
    remaining: Vec<u32>,
    /// Tensors whose covering chunks have all completed, in completion
    /// order, waiting to be yielded.
    ready: VecDeque<usize>,
    /// Tensors yielded so far.
    yielded: usize,
    /// Monotone lower bound for [`next_pending`](Self::next_pending):
    /// `remaining` only ever falls to 0, never back up.
    scan_from: usize,
}

impl StreamCursor {
    /// Open a cursor over `state`. Zero-byte tensors have no covering
    /// chunks and are ready immediately.
    pub fn new(state: Arc<PrefetchState>) -> Self {
        let mut remaining = Vec::with_capacity(state.spans.len());
        let mut ready = VecDeque::new();
        for (t, span) in state.spans.iter().enumerate() {
            if span.begin == span.end {
                remaining.push(0);
                ready.push_back(t);
            } else {
                remaining.push((span.last_chunk - span.first_chunk + 1) as u32);
            }
        }
        Self {
            state,
            pos: 0,
            remaining,
            ready,
            yielded: 0,
            scan_from: 0,
        }
    }

    /// The next tensor to become resident, as `(index, buffer)`. A `None`
    /// buffer means the tensor was already delivered through `get_tensor`
    /// and the caller must fall back to a plain read. `None` ends the
    /// stream. Blocks; call with the GIL released.
    pub fn next(&mut self) -> Result<Option<(usize, Option<AsyncDeviceBuf>)>, String> {
        loop {
            if let Some(t) = self.ready.pop_front() {
                self.yielded += 1;
                return Ok(Some((t, self.state.take_completed(t))));
            }
            if self.yielded >= self.remaining.len() {
                return Ok(None);
            }
            self.advance()?;
        }
    }

    /// Synchronize the next enqueued chunk and mark every tensor it
    /// completes as ready.
    fn advance(&mut self) -> Result<(), String> {
        let chunk = loop {
            /// Starvation probe cadence. A healthy pipeline answers on the
            /// condvar long before this; when it does fire, the probe is
            /// two uncontended mutexes, and this is the latency a genuinely
            /// starved cursor pays before it demands work.
            const POLL: Duration = Duration::from_millis(5);
            if let Some(c) = self.state.issued_at(self.pos, POLL)? {
                break c;
            }
            if self.state.window_starved() {
                // Budget may simply have freed without a nudge reaching a
                // worker; widening on this thread is what an idle worker
                // sweep would have done anyway.
                let _ = PrefetchState::try_refill(&self.state, None);
                if self.state.window_starved() {
                    // Genuinely out of budget (other files hold it): demand
                    // the tensor the consumer is actually blocked on.
                    let t = self.next_pending();
                    let target = self.state.force_horizon(t);
                    if self.state.forced_upto.load(Ordering::Acquire) <= target {
                        let _ = PrefetchState::try_refill(&self.state, Some(target));
                        self.state
                            .forced_upto
                            .fetch_max(target + 1, Ordering::AcqRel);
                    }
                }
            }
        };
        self.pos += 1;
        self.state.sync_chunk(chunk)?;
        for t in self.state.covered_tensors(chunk) {
            if self.remaining[t] == 0 {
                continue; // zero-byte tensor inside the chunk's range
            }
            self.remaining[t] -= 1;
            if self.remaining[t] == 0 {
                self.ready.push_back(t);
            }
        }
        Ok(())
    }

    /// Lowest tensor index still waiting on a chunk: the one a starved
    /// cursor must demand to make progress.
    fn next_pending(&mut self) -> usize {
        while self.scan_from + 1 < self.remaining.len() && self.remaining[self.scan_from] == 0 {
            self.scan_from += 1;
        }
        self.scan_from
    }
}
