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

use std::collections::VecDeque;
use std::fs::File;
use std::os::fd::{AsRawFd, RawFd};
use std::path::Path;

use io_uring::{opcode, types, IoUring};

use crate::error::{Error, Result};

pub const DEFAULT_QUEUE_DEPTH: u32 = 64;
pub const DEFAULT_CHUNK_BYTES: usize = 128 * 1024;

pub struct ReadEngine {
    ring: IoUring,
    /// Owned files. Index in this Vec equals the index into the kernel's
    /// registered-files table (`Fixed(idx)` in opcodes).
    files: Vec<File>,
    /// One staging buffer per slot, length == `qd`. Index equals
    /// `buf_index` in the registered-buffers table.
    bufs: Vec<Box<[u8]>>,
    /// Free slot indices. `pop_front()` to claim, `push_back()` to release.
    free: VecDeque<u32>,
    chunk_bytes: usize,
    qd: u32,
}

impl ReadEngine {
    /// Open a ring with `qd` SQEs/CQEs and pre-allocate `qd` chunk-sized
    /// staging buffers, registered with the kernel. Files are added later
    /// via `register_file`.
    pub fn new(qd: u32, chunk_bytes: usize) -> Result<Self> {
        if qd == 0 || chunk_bytes == 0 {
            return Err(Error::Other(format!(
                "invalid ReadEngine config: qd={qd} chunk_bytes={chunk_bytes}"
            )));
        }
        let ring = IoUring::new(qd).map_err(|e| io_err("IoUring::new", &e))?;

        // Allocate one buffer per slot. Initialized to zero — the kernel
        // overwrites on read; we never observe the initial bytes.
        let bufs: Vec<Box<[u8]>> = (0..qd)
            .map(|_| vec![0u8; chunk_bytes].into_boxed_slice())
            .collect();

        // Build iovecs pointing at each buffer. The kernel doesn't move the
        // memory; we just hand it the address + length so it can do the read.
        let iovecs: Vec<libc::iovec> = bufs
            .iter()
            .map(|b| libc::iovec {
                iov_base: b.as_ptr() as *mut _,
                iov_len: b.len(),
            })
            .collect();
        // SAFETY: each iovec points into a `Box<[u8]>` we own; the boxes
        // outlive the registration (we explicitly unregister in `Drop` before
        // the boxes are freed; field drop order ensures `ring` drops first).
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

    /// Submit `chunks` to io_uring and synchronously process completions via
    /// `on_chunk`. The callback receives the slot's staging buffer (read
    /// truncated to the actual byte count returned by the kernel) and the
    /// scatter entries the coalescer attached to this chunk. The slot
    /// returns to the free pool immediately after `on_chunk` returns.
    ///
    /// In P4 this is replaced by a `submit` + `peek_completed` pair, with
    /// the DMA worker owning slot release.
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
                let buf_ptr = self.bufs[slot as usize].as_mut_ptr();

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
                let buf = &self.bufs[slot as usize][..bytes_read];

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

impl Drop for ReadEngine {
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
