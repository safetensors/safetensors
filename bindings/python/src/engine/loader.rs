//! CUDA prefetch engine: load byte spans of a safetensors data section into
//! device memory the caller allocated, in the background, and hand each span's
//! owner back, once, as its bytes land.
//!
//! Data flow, per file:
//!
//! ```text
//! caller (one at a time)
//!   Feed::provide(dsts, stream)          the next spans' destinations, in plan
//!                                        order, and an event on `stream`
//! worker threads, transfer::work from the DirectFile to the CudaSink
//!   c = the Plan's next chunk            next entry of LoadPlan::chunks
//!   CudaSink::wait_ready(c)              c's spans provided (condvar), then
//!                                        the events of their batches
//!   slab = SlabPool::acquire()           a pinned host slab from the shared
//!                                        pool; waits for its previous copy
//!   DirectFile::fill(c, slab)            one read per chunk
//!   CudaSink::drain: memcpyAsync         one copy per run of adjacent
//!     (slab -> destinations)             destinations, on STREAMS[device];
//!                                        memcpy2DAsync for a gather's rows
//!   chunk_completion_events[c] = event   recorded after the copies; 0 while
//!                                        the chunk is still pending
//! consumer
//!   take_tensor(s) / TensorIter::next    wait the events of s's chunks, then
//!     -> owner of s's destination        handed out once (AlreadyDelivered after)
//! ```
//!
//! [`Loader`] owns the worker threads; [`LoaderInner`] (source, plan, sink, the
//! [`Plan`] of chunks with its first error) is shared with them and with
//! [`TensorIter`]. Closing
//! signals the workers, joins them, waits for the copies in flight, then
//! returns the owners not taken: an outstanding iterator yields `Err(Closed)`
//! once and is exhausted.

use std::{
    collections::VecDeque,
    fmt::Display,
    fs::File,
    sync::{atomic::Ordering, Arc, Mutex},
    thread::JoinHandle,
};

use super::{
    cuda::{api, CudaError, Stream},
    cuda_memory::{pinned_pool, MAX_DEVICES},
    file::DirectFile,
    memory::{CudaSink, LoadPlan, Span},
    pool::SlabPool,
    transfer::{self, Error, Plan, CHUNK_SIZE},
};
use safetensors::slice::Gather;

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

struct LoaderInner<D> {
    source: DirectFile,
    plan: Arc<LoadPlan>,
    sink: CudaSink<D>,
    pool: Arc<SlabPool>,
    /// the plan's chunks as the workers claim them, and the first error
    transfer: Plan,
    workers: Mutex<Vec<JoinHandle<()>>>,
}

impl<D> LoaderInner<D> {
    /// Stops loading and returns the owners not taken. A `take_tensor` waiting in
    /// another thread returns its owner or an error: `Closed`, or what failed the load.
    fn close(&self) -> Vec<D> {
        self.halt();
        self.sink.close()
    }

    /// Signals the workers and joins them; the owners stay in the sink.
    fn halt(&self) {
        self.transfer.stop();
        self.sink.signal_stop();
        // held while joining: a concurrent halt returns once the workers are gone
        let mut workers = self.workers.lock().unwrap();
        for h in workers.drain(..) {
            let _ = h.join();
        }
    }

    fn take_tensor(&self, idx: usize) -> Result<D, LoaderError> {
        self.sink
            .wait_landed(idx)
            .and_then(|()| self.sink.take(idx))
            .map_err(|e| match (e, self.transfer.error()) {
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
        self.inner.sink.provided.load(Ordering::Acquire)
    }

    /// How many spans the read schedule has reached: at least the ones a
    /// worker is reading or waiting to read
    pub fn scheduled(&self) -> usize {
        self.inner.sink.scheduled(self.inner.transfer.claimed())
    }

    /// Whether the load stopped: closed, failed or halted
    pub fn stopped(&self) -> bool {
        self.inner.transfer.stopped()
    }

    /// Stops the load as a worker error would: `take` of a span not loaded yet
    /// fails with `reason`. No-op once the load stopped.
    pub fn fail(&self, reason: String) {
        if self.inner.transfer.stopped() {
            return;
        }
        self.inner.transfer.fail(Error::Other(reason.into()));
        self.inner.sink.signal_stop();
    }

    /// Waits until the read schedule is past `scheduled` spans or the load stopped
    pub fn wait_scheduled(&self, scheduled: usize) {
        self.inner
            .transfer
            .wait_until(|| self.scheduled() != scheduled);
    }

    /// Stops the load without joining the workers
    pub fn stop(&self) {
        self.inner.transfer.stop();
        self.inner.sink.signal_stop();
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
        let pool = cuda_api.with_device(device, |_| pinned_pool(cuda_api))?;
        let inner = Arc::new(LoaderInner {
            source: DirectFile::new(file, direct, pool.aligned()),
            transfer: Plan::new(plan.file_chunks()),
            plan: plan.clone(),
            sink: CudaSink::new(cuda_api, plan, device)?,
            pool,
            workers: Mutex::new(Vec::with_capacity(threads)),
        });
        let loader = Self { inner };
        for _ in 0..threads {
            let inner = loader.inner.clone();
            let spawned = std::thread::Builder::new().spawn(move || {
                let _ = cuda_api.set_device(device);
                transfer::work(&inner.transfer, &inner.source, &inner.sink, &inner.pool)
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
