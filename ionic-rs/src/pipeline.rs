//! CUDA pipeline orchestration — two-thread design ported from
//! `~/ionic/lib/pipelines/cuda.c`. Same concurrency story:
//!
//! - **Scheduler thread** (the caller's thread): drives the io_uring engine.
//!   Submits reads, drains CQEs, forwards completed slots to the DMA worker,
//!   recycles slots when the worker signals their H2D is done.
//! - **DMA worker thread** (`std::thread::scope`-spawned per [`run`] call):
//!   keeps up to two H2D copies in flight (one per CUDA stream — matches
//!   NVIDIA's two-async-copy-engine hardware reality), polls
//!   `cuEventQuery` non-blocking, signals done back to the scheduler.
//! - **Two streams + two events** (`cudaStreamCreateWithFlags(...,
//!   cudaStreamNonBlocking)` in ionic; `CuStream::new()` here, which uses
//!   the same flag). Round-robin across them so reads and H2Ds overlap.
//!
//! ionic uses two atomic bitmasks (`pending` and `done`) for cross-thread
//! signalling. We use channels — same semantics with less unsafe surface.
//! The scheduler-side per-iteration loop is essentially ionic's
//! `iouring.c:441-448` rearranged around `mpsc::Receiver::try_recv` and a
//! state-machine choice between blocking on io_uring vs. blocking on the
//! done channel.
//!
//! ## Lifetimes / safety story
//!
//! - `std::thread::scope` ensures the worker is joined before [`run`]
//!   returns, even on panic. No `'static` requirement on shared data.
//! - `PinnedBuf` staging buffers live inside the engine; their addresses
//!   stay stable for the engine's lifetime. The scheduler hands the worker
//!   a `*const u8 + bytes_read` snapshot for each completed CQE; that
//!   pointer is valid until the worker signals done and the slot is
//!   recycled. The slot can't be re-issued for a new io_uring read until
//!   the worker has signalled done, so the worker's read of the staging
//!   bytes can't race with a kernel write into the same buffer.
//! - Both the scheduler and the worker push the primary context on entry;
//!   primary contexts are safe to make current on multiple threads
//!   simultaneously.

use std::path::Path;
use std::sync::mpsc;
use std::sync::Mutex;

use crate::cuda::{self, CuContext, CuDevice, CuEvent, CuStream, DeviceBuf, PinnedBuf};
use crate::error::{Error, Result};
use crate::iouring::{coalesce_chunks, ChunkRead, LogicalSegment, ReadEngine};

/// One unit of work the scheduler hands to the DMA worker after an
/// io_uring read completes.
///
/// `buf_ptr` snapshots the staging buffer's address for the duration of
/// this slot's H2D dispatch. The slot is held in flight until the worker
/// signals back via `done_tx`; only then does the scheduler recycle it
/// and risk reusing the same staging buffer for a new read.
#[derive(Clone, Copy)]
struct DmaJob {
    slot: u32,
    chunk_idx: usize,
    buf_ptr: *const u8,
    bytes_read: usize,
}

// SAFETY: `buf_ptr` is a snapshot into a `PinnedBuf` owned by the engine.
// The engine outlives the scoped DMA worker; the slot is not recycled
// until the worker signals done, so no aliasing with kernel writes.
unsafe impl Send for DmaJob {}

/// CUDA pipeline tying the io_uring engine to a pair of streams.
/// Constructed once per file (or once per pipeline lifetime, with
/// repeated `run` calls); construction allocates QD pinned staging
/// buffers, two non-blocking streams, two timing-disabled events, and
/// retains the device's primary context.
pub struct CudaPipeline {
    /// Order matters — fields drop in declaration order. Engine, streams,
    /// events all hold pointers into the primary context, so they must
    /// drop before the context-release runs in `ctx`'s `Drop`.
    engine: ReadEngine<PinnedBuf>,
    streams: [CuStream; 2],
    events: [CuEvent; 2],
    ctx: CuContext,
}

impl CudaPipeline {
    pub fn new(device_ordinal: i32, qd: u32, chunk_bytes: usize) -> Result<Self> {
        let dev = CuDevice::get(device_ordinal)?;
        let ctx = CuContext::primary_retain(dev)?;
        // PinnedBuf::alloc + cuStreamCreate + cuEventCreate all need a
        // current context. Hold it for construction.
        let (engine, streams, events) = ctx.with_current(|| {
            let engine = ReadEngine::<PinnedBuf>::new(qd, chunk_bytes)?;
            let streams = [CuStream::new()?, CuStream::new()?];
            let events = [CuEvent::new()?, CuEvent::new()?];
            Ok((engine, streams, events))
        })?;
        Ok(Self {
            engine,
            streams,
            events,
            ctx,
        })
    }

    pub fn register_file(&mut self, path: &Path) -> Result<u32> {
        self.engine.register_file(path)
    }

    /// Allocate a device buffer on this pipeline's primary context. Use to
    /// pre-allocate per-tensor target buffers that subsequent `run` calls
    /// will DMA into. The caller doesn't need to manage context push/pop —
    /// we do it here.
    pub fn alloc_device_buf(&self, bytes: usize) -> Result<DeviceBuf> {
        self.ctx.with_current(|| DeviceBuf::alloc(bytes))
    }

    /// Run the given logical reads through the pipeline. Each segment's
    /// `dst_offset` field is interpreted as a `CUdeviceptr` (cast from
    /// `usize`) — the H2D scatter lands at `dst_device_ptr +
    /// chunk_overlap_offset` for each scatter entry the coalescer
    /// produces.
    ///
    /// Returns when every byte requested is resident on the device. The
    /// scoped DMA worker is joined before return; on worker error, the
    /// error surfaces here.
    pub fn run(&mut self, segments: &[LogicalSegment]) -> Result<()> {
        let chunks = coalesce_chunks(segments, self.engine.chunk_bytes());
        let total = chunks.len();
        if total == 0 {
            return Ok(());
        }

        let qd = self.engine.queue_depth() as usize;
        let mut slot_to_chunk: Vec<usize> = vec![0; qd];

        // Scheduler ↔ worker channels. mpsc::channel is unbounded but
        // bounded de facto by QD: at most QD slots can be in flight, and
        // each occupies at most one DmaJob in transit.
        let (job_tx, job_rx) = mpsc::channel::<DmaJob>();
        let (done_tx, done_rx) = mpsc::channel::<u32>();

        // Worker error surfaces here on join.
        let worker_err: Mutex<Option<Error>> = Mutex::new(None);

        // Borrow split: scheduler takes `&mut self.engine`, worker gets
        // immutable refs to the other fields. Disjoint field borrows allow
        // both to coexist — but only via field access, not method syntax,
        // so we destructure here.
        let Self {
            engine,
            streams,
            events,
            ctx,
        } = self;
        let chunks_ref = &chunks;
        let worker_err_ref = &worker_err;

        std::thread::scope(|scope| -> Result<()> {
            scope.spawn(move || {
                if let Err(e) = run_dma_worker(
                    job_rx, done_tx, chunks_ref, streams, events, ctx,
                ) {
                    *worker_err_ref.lock().unwrap() = Some(e);
                }
            });

            let reader_result = run_scheduler_loop(
                engine,
                &chunks,
                &mut slot_to_chunk,
                &job_tx,
                &done_rx,
                total,
            );

            // Drop our sender — when the worker finishes its current jobs
            // and pulls again, it'll observe Disconnected and exit.
            drop(job_tx);
            // (thread::scope joins the worker before the closure returns.)

            reader_result
        })?;

        // Worker error has priority over a clean reader return — if the
        // worker silently failed mid-stream, the GPU buffer is incomplete
        // and we mustn't return Ok.
        if let Some(e) = worker_err.into_inner().unwrap() {
            return Err(e);
        }

        // Final stream sync. Per-event polling proved each H2D landed on
        // its producer-side stream; this barrier promotes that to a
        // happens-before edge any consumer stream can rely on. Without
        // it, a downstream tensor handed off via DLPack and read on
        // torch's default stream could see partial writes (events
        // establish ordering only along the same stream by default).
        self.ctx.with_current(|| {
            self.streams[0].synchronize()?;
            self.streams[1].synchronize()
        })?;
        Ok(())
    }

}

/// Scheduler loop on the calling thread. Mirrors ionic's
/// `iouring.c:373-460` `fetch` body modulo Rust idioms.
///
/// Free-function (not a method) so we can pass `&mut engine` alongside
/// `&streams`/`&events` without re-borrowing through `self`.
fn run_scheduler_loop(
    engine: &mut ReadEngine<PinnedBuf>,
    chunks: &[ChunkRead],
    slot_to_chunk: &mut [usize],
    job_tx: &mpsc::Sender<DmaJob>,
    done_rx: &mpsc::Receiver<u32>,
    total: usize,
) -> Result<()> {
    let mut next_chunk: usize = 0; // index of next chunk to submit
    let mut io_outstanding: usize = 0; // submitted, awaiting CQE
    let mut dma_outstanding: usize = 0; // forwarded to worker, awaiting done signal
    let mut completed: usize = 0; // slots fully recycled

    while completed < total {
        // 1. Drain done signals from worker → recycle slots.
        while let Ok(slot) = done_rx.try_recv() {
            engine.release_slot(slot);
            dma_outstanding -= 1;
            completed += 1;
        }
        if completed == total {
            break;
        }

        // 2. Submit reads while we have free slots and unsubmitted chunks.
        while next_chunk < total {
            let slot = match engine.acquire_slot() {
                Some(s) => s,
                None => break, // all slots in flight (io OR dma)
            };
            if let Err(e) = engine.submit_read(slot, &chunks[next_chunk]) {
                engine.release_slot(slot);
                return Err(e);
            }
            slot_to_chunk[slot as usize] = next_chunk;
            next_chunk += 1;
            io_outstanding += 1;
        }
        engine.flush_submissions()?;

        // 3. Block for progress. Pick which side based on what's
        //    outstanding: prefer io_uring (returns fast when work is
        //    pending); fall through to done_rx when only DMA is left.
        if io_outstanding > 0 {
            // submit_and_wait(1) is cheap if CQEs are already pending.
            engine.submit_and_wait_one()?;
        } else if dma_outstanding > 0 {
            // No io_uring work outstanding — block on the worker.
            let slot = done_rx.recv().map_err(|_| {
                Error::Other("DMA worker disconnected with chunks still pending".into())
            })?;
            engine.release_slot(slot);
            dma_outstanding -= 1;
            completed += 1;
            continue; // back to top, drain any other ready signals
        } else {
            return Err(Error::Other(format!(
                "pipeline state inconsistent: completed={} total={} io=0 dma=0",
                completed, total
            )));
        }

        // 4. Drain newly-ready CQEs → forward to worker.
        for c in engine.peek_completions()? {
            let chunk_idx = slot_to_chunk[c.slot as usize];
            let buf_ptr = engine.slot_buffer(c.slot, c.bytes_read).as_ptr();
            let job = DmaJob {
                slot: c.slot,
                chunk_idx,
                buf_ptr,
                bytes_read: c.bytes_read,
            };
            io_outstanding -= 1;
            if job_tx.send(job).is_err() {
                return Err(Error::Other(
                    "DMA worker disconnected mid-stream".into(),
                ));
            }
            dma_outstanding += 1;
        }
    }
    Ok(())
}

/// DMA worker thread body. Pulls jobs, dispatches H2D on alternating
/// streams, polls events to detect H2D completion, signals slots back to
/// the scheduler.
///
/// Same shape as ionic's `cuda.c:152-247`. Uses `cuEventQuery` to recycle
/// slots without ever blocking on `cuStreamSynchronize` — the whole point
/// of the two-thread design is to keep io_uring submission and CUDA
/// dispatch progressing independently.
fn run_dma_worker(
    job_rx: mpsc::Receiver<DmaJob>,
    done_tx: mpsc::Sender<u32>,
    chunks: &[ChunkRead],
    streams: &[CuStream; 2],
    events: &[CuEvent; 2],
    ctx: &CuContext,
) -> Result<()> {
    ctx.with_current(|| -> Result<()> {
        // ionic: `struct ionic_io_fetch_result *dmas[2] = { NULL, NULL };`
        // We hold the same shape — one in-flight DmaJob per stream.
        let mut in_flight: [Option<DmaJob>; 2] = [None, None];

        loop {
            // 1. Poll events for completed H2Ds → signal slot back.
            for stream_idx in 0..2 {
                if let Some(job) = in_flight[stream_idx] {
                    if events[stream_idx].query()? {
                        if done_tx.send(job.slot).is_err() {
                            // Scheduler is gone; nothing left to do.
                            return Ok(());
                        }
                        in_flight[stream_idx] = None;
                    }
                }
            }

            // 2. Try to fill any empty stream with a new job.
            let mut got_any = false;
            for stream_idx in 0..2 {
                if in_flight[stream_idx].is_some() {
                    continue;
                }
                let job = match job_rx.try_recv() {
                    Ok(j) => Some(j),
                    Err(mpsc::TryRecvError::Empty) => {
                        // No work to grab right now.
                        if in_flight[0].is_none()
                            && in_flight[1].is_none()
                            && stream_idx == 0
                        {
                            // Both streams idle, no jobs in queue — block
                            // on a recv. Disconnect = clean shutdown.
                            match job_rx.recv() {
                                Ok(j) => Some(j),
                                Err(_) => return Ok(()),
                            }
                        } else {
                            None
                        }
                    }
                    Err(mpsc::TryRecvError::Disconnected) => {
                        // No more jobs coming. Drain in-flight and exit.
                        if in_flight[0].is_none() && in_flight[1].is_none() {
                            return Ok(());
                        }
                        None
                    }
                };
                if let Some(job) = job {
                    dispatch_h2d(&job, chunks, &streams[stream_idx])?;
                    events[stream_idx].record(&streams[stream_idx])?;
                    in_flight[stream_idx] = Some(job);
                    got_any = true;
                }
            }

            if !got_any {
                // Nothing to dispatch this iteration; events still pending.
                // PAUSE / YIELD — same as ionic_cpu_relax.
                std::hint::spin_loop();
            }
        }
    })
}

/// Fire one `cuMemcpyHtoDAsync` per scatter entry of the chunk that this
/// job's slot received. Mirrors the inner loop of ionic's DMA worker
/// (`cuda.c:179-198`), with the same short-read trimming the scheduler
/// already applies to the scatter list.
fn dispatch_h2d(job: &DmaJob, chunks: &[ChunkRead], stream: &CuStream) -> Result<()> {
    let chunk = &chunks[job.chunk_idx];
    for s in &chunk.scatter {
        if s.staging_offset >= job.bytes_read {
            // Scatter entry falls entirely past the kernel's short read.
            continue;
        }
        let avail = job.bytes_read - s.staging_offset;
        let len = s.len.min(avail);
        // SAFETY: buf_ptr is a snapshot of the staging buffer that the
        // engine guarantees is valid for `bytes_read` bytes until this
        // job's slot is signalled done.
        let src = unsafe { job.buf_ptr.add(s.staging_offset) };
        cuda::memcpy_h2d_async(s.dst_offset as u64, src, len, stream)?;
    }
    Ok(())
}
