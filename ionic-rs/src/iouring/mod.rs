//! io_uring read engine with registered files + registered buffers + a
//! slot pool. Port of ionic's `iouring.c` engine surface (`fetch` + the
//! companion file/buffer registration helpers), simplified for P3:
//! synchronous slot recycling — a slot returns to the free pool the moment
//! its CQE is processed, since the chunk's bytes are consumed via the
//! `on_chunk` callback before we pop the next CQE.
//!
//! In P4 this changes: the DMA worker becomes the consumer and the slot is
//! held until the H2D copy completes (ionic's `pending` / `done` atomic
//! bitmasks). The engine's submit/poll loop will be the same shape; only
//! the slot-release plumbing changes.
//!
//! ## Lifetime invariants
//!
//! - Registered fds must outlive the engine — we own `Vec<File>`.
//! - Registered buffers must not move and must outlive any in-flight read.
//!   We own `Vec<Box<[u8]>>` and explicitly `unregister_buffers` in `Drop`
//!   before the boxes are freed.
//! - Submitting before `register_files`/`register_buffers` is a kernel-side
//!   error; the constructor finishes registration before returning.

mod coalesce;
pub use coalesce::{coalesce_chunks, ChunkRead, LogicalSegment, ScatterEntry};

/// One io_uring read completion. Returned by `ReadEngine::peek_completions`.
/// `slot` identifies the staging buffer holding the bytes; `bytes_read` is
/// the actual count returned by the kernel (may be less than the requested
/// chunk length when reading past EOF).
#[derive(Debug, Clone, Copy)]
pub struct Completion {
    pub slot: u32,
    pub bytes_read: usize,
}

use std::collections::VecDeque;
use std::fs::File;
use std::os::fd::{AsRawFd, RawFd};
use std::path::Path;

use io_uring::{opcode, types, IoUring};

use crate::error::{Error, Result};

pub const DEFAULT_QUEUE_DEPTH: u32 = 64;
/// 512 KiB. Empirically chosen — between 128 KiB and 4 MiB the warm-cache
/// throughput on a single-tensor read is roughly flat, but the per-chunk
/// fixed overhead (slot bookkeeping + stream sync) amortizes better at
/// 512 KiB than at 128 KiB. Cold-cache stays bandwidth-bound on either.
pub const DEFAULT_CHUNK_BYTES: usize = 512 * 1024;

/// Storage type for io_uring's registered staging buffers.
///
/// The engine accepts any backing that exposes raw byte storage. Two
/// implementations matter:
///
/// - [`Box<[u8]>`] — pageable host memory. Used by the standalone tests and
///   any non-CUDA consumer. Reads land here, but the H2D path that follows
///   has to bounce through the driver's own pinned buffer, capping
///   throughput at ~6–7 GB/s effective.
/// - [`crate::cuda::PinnedBuf`] — page-locked host memory allocated via
///   `cuMemHostAlloc`. The CUDA pipeline uses this so `cuMemcpyHtoDAsync`
///   can DMA directly without a bounce, hitting ~25–50 GB/s on PCIe Gen5.
pub trait StagingBuffer: Send {
    fn alloc(bytes: usize) -> Result<Self>
    where
        Self: Sized;
    fn as_slice(&self) -> &[u8];
    fn as_mut_slice(&mut self) -> &mut [u8];
}

impl StagingBuffer for Box<[u8]> {
    fn alloc(bytes: usize) -> Result<Self> {
        Ok(vec![0u8; bytes].into_boxed_slice())
    }
    fn as_slice(&self) -> &[u8] {
        self
    }
    fn as_mut_slice(&mut self) -> &mut [u8] {
        self
    }
}

impl StagingBuffer for crate::cuda::PinnedBuf {
    fn alloc(bytes: usize) -> Result<Self> {
        Self::alloc(bytes)
    }
    fn as_slice(&self) -> &[u8] {
        self.as_slice()
    }
    fn as_mut_slice(&mut self) -> &mut [u8] {
        self.as_mut_slice()
    }
}

/// io_uring read engine generic over its staging buffer type. Default is
/// `Box<[u8]>` so existing call sites and tests work unchanged; the CUDA
/// pipeline parametrizes with `PinnedBuf` for direct-DMA H2D.
pub struct ReadEngine<B: StagingBuffer = Box<[u8]>> {
    ring: IoUring,
    /// Owned files. Index in this Vec equals the index into the kernel's
    /// registered-files table (`Fixed(idx)` in opcodes).
    files: Vec<File>,
    /// One staging buffer per slot, length == `qd`. Index equals
    /// `buf_index` in the registered-buffers table.
    bufs: Vec<B>,
    /// Free slot indices. `pop_front()` to claim, `push_back()` to release.
    free: VecDeque<u32>,
    chunk_bytes: usize,
    qd: u32,
}

impl<B: StagingBuffer> ReadEngine<B> {
    /// Open a ring with `qd` SQEs/CQEs and pre-allocate `qd` chunk-sized
    /// staging buffers via `B::alloc`, then register them with the kernel.
    /// Files are added later via `register_file`.
    pub fn new(qd: u32, chunk_bytes: usize) -> Result<Self> {
        if qd == 0 || chunk_bytes == 0 {
            return Err(Error::Other(format!(
                "invalid ReadEngine config: qd={qd} chunk_bytes={chunk_bytes}"
            )));
        }
        let ring = IoUring::new(qd).map_err(|e| io_err("IoUring::new", &e))?;

        // Allocate one buffer per slot. The chosen `B::alloc` controls
        // whether these are pageable (`Box<[u8]>`) or pinned (`PinnedBuf`).
        let mut bufs: Vec<B> = Vec::with_capacity(qd as usize);
        for _ in 0..qd {
            bufs.push(B::alloc(chunk_bytes)?);
        }

        // Build iovecs pointing at each buffer. The kernel doesn't move the
        // memory; we just hand it the address + length so it can do the read.
        let iovecs: Vec<libc::iovec> = bufs
            .iter()
            .map(|b| {
                let s = b.as_slice();
                libc::iovec {
                    iov_base: s.as_ptr() as *mut _,
                    iov_len: s.len(),
                }
            })
            .collect();
        // SAFETY: each iovec points into a buffer we own; the buffers
        // outlive the registration (we explicitly unregister in `Drop` before
        // the buffers are freed; field drop order ensures `ring` drops first).
        unsafe {
            ring.submitter()
                .register_buffers(&iovecs)
                .map_err(|e| io_err("register_buffers", &e))?;
        }

        let free = (0..qd).collect();
        Ok(Self {
            ring,
            files: Vec::new(),
            bufs,
            free,
            chunk_bytes,
            qd,
        })
    }

    pub fn chunk_bytes(&self) -> usize {
        self.chunk_bytes
    }

    pub fn queue_depth(&self) -> u32 {
        self.qd
    }

    /// Register a file with the kernel and return its `fd_idx` for later
    /// submissions. Re-registers the entire fd table on each call — this is
    /// `O(n)` in the file count but n is tiny (one per shard, typically
    /// single digits) and registration only happens during `prefetch()` setup.
    pub fn register_file(&mut self, path: &Path) -> Result<u32> {
        let file = File::open(path).map_err(|e| {
            Error::Other(format!("open {}: {e}", path.display()))
        })?;
        let mut fds: Vec<RawFd> = self.files.iter().map(|f| f.as_raw_fd()).collect();
        fds.push(file.as_raw_fd());
        // Unregister silently — first registration has nothing to unregister
        // (the kernel returns ENXIO), which is fine to swallow.
        let _ = self.ring.submitter().unregister_files();
        self.ring
            .submitter()
            .register_files(&fds)
            .map_err(|e| io_err("register_files", &e))?;
        let idx = self.files.len() as u32;
        self.files.push(file);
        Ok(idx)
    }

    // ── Lower-level primitives for the async CUDA pipeline ─────────────
    //
    // The `fetch` convenience below uses these internally for the
    // synchronous slot-recycle loop. The CUDA pipeline calls them directly
    // so it can interleave io_uring polling with `cuEventQuery`-based slot
    // recycling — slot release waits on the H2D event, not on a stream sync.

    /// Pop a free slot index from the pool, or `None` if all slots are
    /// currently in flight.
    pub fn acquire_slot(&mut self) -> Option<u32> {
        self.free.pop_front()
    }

    /// Push an `IORING_OP_READ_FIXED` SQE for `chunk` into the slot's
    /// staging buffer. Doesn't submit to the kernel — caller batches via
    /// `flush_submissions`.
    ///
    /// The SQE's user_data is set to the slot index, so completions
    /// returned by `peek_completions` carry the slot back.
    pub fn submit_read(&mut self, slot: u32, chunk: &ChunkRead) -> Result<()> {
        let buf_ptr = self.bufs[slot as usize].as_mut_slice().as_mut_ptr();
        let entry = opcode::ReadFixed::new(
            types::Fixed(chunk.fd_idx),
            buf_ptr,
            chunk.len as u32,
            slot as u16,
        )
        .offset(chunk.offset)
        .build()
        .user_data(slot as u64);
        // SAFETY: pointers refer to memory we own (`bufs[slot]`) and a
        // registered fd (`files`); both outlive the in-flight op (caller
        // tracks completion before recycling slot or dropping engine).
        unsafe {
            self.ring.submission().push(&entry).map_err(|_| {
                Error::Other("submission queue push failed (queue full?)".into())
            })?;
        }
        Ok(())
    }

    /// Submit any pending SQEs to the kernel. Returns the number submitted.
    pub fn flush_submissions(&mut self) -> Result<usize> {
        self.ring
            .submit()
            .map_err(|e| io_err("submit", &e))
    }

    /// Block until at least one CQE is available, submitting any pending
    /// SQEs in the same syscall. Use when the slot pool is exhausted and
    /// no more progress is possible without a completion.
    pub fn submit_and_wait_one(&mut self) -> Result<()> {
        self.ring
            .submit_and_wait(1)
            .map_err(|e| io_err("submit_and_wait", &e))?;
        Ok(())
    }

    /// Drain ready CQEs without blocking. Returns one entry per completed
    /// read carrying the slot index and the byte count actually read
    /// (which may be less than `chunk.len` near EOF).
    pub fn peek_completions(&mut self) -> Result<Vec<Completion>> {
        let cqes: Vec<io_uring::cqueue::Entry> = self.ring.completion().collect();
        let mut out = Vec::with_capacity(cqes.len());
        for cqe in cqes {
            let slot = cqe.user_data() as u32;
            let result = cqe.result();
            if result < 0 {
                return Err(Error::IoUring {
                    op: "read_fixed",
                    errno: -result,
                });
            }
            out.push(Completion {
                slot,
                bytes_read: result as usize,
            });
        }
        Ok(out)
    }

    /// Borrow a slot's staging buffer up to `bytes` bytes — caller passes
    /// the byte count from the matching `Completion`. Use after a CQE is
    /// returned and before the slot is released.
    pub fn slot_buffer(&self, slot: u32, bytes: usize) -> &[u8] {
        &self.bufs[slot as usize].as_slice()[..bytes]
    }

    /// Return a slot to the free pool. Caller is responsible for ensuring
    /// no pending consumer is still reading from the slot's buffer (e.g.
    /// pending H2D copy must be observed complete via `cuEventQuery`).
    pub fn release_slot(&mut self, slot: u32) {
        self.free.push_back(slot);
    }

    /// Submit `chunks` to io_uring and synchronously process completions via
    /// `on_chunk`. The callback receives the slot's staging buffer (read
    /// truncated to the actual byte count returned by the kernel) and the
    /// scatter entries the coalescer attached to this chunk. The slot
    /// returns to the free pool immediately after `on_chunk` returns.
    ///
    /// Used by the standalone iouring tests; the CUDA pipeline drives the
    /// lower-level primitives directly to overlap reads with H2D.
    pub fn fetch<F>(&mut self, chunks: &[ChunkRead], mut on_chunk: F) -> Result<()>
    where
        F: FnMut(&[u8], &[ScatterEntry]) -> Result<()>,
    {
        if chunks.is_empty() {
            return Ok(());
        }

        let total = chunks.len();
        let mut next_chunk = 0usize;
        let mut completed = 0usize;
        // slot_to_chunk[slot] = Some(idx) while slot has an in-flight chunk.
        let mut slot_to_chunk: Vec<Option<usize>> = vec![None; self.qd as usize];

        while completed < total {
            // Phase 1: submit until QD exhausted or we run out of chunks.
            let mut pushed_any = false;
            while next_chunk < total {
                let slot = match self.free.pop_front() {
                    Some(s) => s,
                    None => break, // QD full; need to drain CQEs first.
                };
                let chunk = &chunks[next_chunk];
                let buf_ptr = self.bufs[slot as usize].as_mut_slice().as_mut_ptr();

                let entry = opcode::ReadFixed::new(
                    types::Fixed(chunk.fd_idx),
                    buf_ptr,
                    chunk.len as u32,
                    slot as u16,
                )
                .offset(chunk.offset)
                .build()
                .user_data(slot as u64);

                // SAFETY: `entry`'s pointers (buf, fd index, buf index) all
                // refer to memory we own (`bufs[slot]`) and resources we
                // registered (`files`); both live longer than the in-flight
                // op (we synchronously wait for completion below).
                unsafe {
                    self.ring.submission().push(&entry).map_err(|_| {
                        Error::Other(
                            "submission queue push failed (queue full?)".into(),
                        )
                    })?;
                }
                slot_to_chunk[slot as usize] = Some(next_chunk);
                next_chunk += 1;
                pushed_any = true;
            }

            // Phase 2: submit + wait for at least one CQE if anything is
            // outstanding. If we pushed nothing this iteration but still have
            // outstanding ops, `submit_and_wait(1)` blocks for one of them.
            let outstanding = next_chunk - completed;
            if pushed_any || outstanding > 0 {
                let want = if outstanding > 0 { 1 } else { 0 };
                self.ring
                    .submit_and_wait(want)
                    .map_err(|e| io_err("submit_and_wait", &e))?;
            }

            // Phase 3: drain ready CQEs. Collect into a Vec first so the
            // mutable borrow of `self.ring.completion()` doesn't conflict
            // with the immutable borrow of `self.bufs` we need afterwards.
            let cqes: Vec<io_uring::cqueue::Entry> = self.ring.completion().collect();
            for cqe in cqes {
                let slot = cqe.user_data() as u32;
                let chunk_idx = slot_to_chunk[slot as usize].take().ok_or_else(|| {
                    Error::Other(format!(
                        "spurious CQE for slot {slot} with no in-flight chunk"
                    ))
                })?;
                let result = cqe.result();
                if result < 0 {
                    return Err(Error::IoUring {
                        op: "read_fixed",
                        errno: -result,
                    });
                }
                let bytes_read = result as usize;
                let chunk = &chunks[chunk_idx];
                let buf = &self.bufs[slot as usize].as_slice()[..bytes_read];

                // Short reads happen at EOF: kernel returns fewer bytes than
                // we asked for. We promise callers that every scatter entry
                // they receive fits inside `buf`, so trim entries that fall
                // entirely past `bytes_read` and clamp ones that straddle it.
                // Fast path (no allocation) when the read was full.
                if chunk
                    .scatter
                    .iter()
                    .all(|s| s.staging_offset + s.len <= bytes_read)
                {
                    on_chunk(buf, &chunk.scatter)?;
                } else {
                    let trimmed: Vec<ScatterEntry> = chunk
                        .scatter
                        .iter()
                        .filter_map(|s| {
                            if s.staging_offset >= bytes_read {
                                None
                            } else {
                                let avail = bytes_read - s.staging_offset;
                                Some(ScatterEntry {
                                    staging_offset: s.staging_offset,
                                    len: s.len.min(avail),
                                    dst_offset: s.dst_offset,
                                    user_data: s.user_data,
                                })
                            }
                        })
                        .collect();
                    on_chunk(buf, &trimmed)?;
                }
                self.free.push_back(slot);
                completed += 1;
            }
        }
        Ok(())
    }
}

impl<B: StagingBuffer> Drop for ReadEngine<B> {
    fn drop(&mut self) {
        // Unregister explicitly so the kernel releases its references to our
        // pinned buffers and fds before the backing memory / files are freed
        // by the field-drop sequence (ring → files → bufs).
        let _ = self.ring.submitter().unregister_buffers();
        let _ = self.ring.submitter().unregister_files();
    }
}

fn io_err(op: &'static str, e: &std::io::Error) -> Error {
    Error::IoUring {
        op,
        errno: e.raw_os_error().unwrap_or(0),
    }
}
