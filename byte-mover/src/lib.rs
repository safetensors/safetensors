//! Reads byte ranges of files into memory the caller owns.
//!
//! Reads are queued on a [`Mover`], run by a pool of threads or by an io_uring,
//! and reaped one by one, each with the id it was queued under. Any thread may
//! queue and reap. A mover knows nothing of what the bytes are: file formats,
//! caches and devices are the caller's.
//!
//! ```no_run
//! use byte_mover::{Mover, Read};
//! use std::os::fd::AsRawFd;
//!
//! let file = std::fs::File::open("model.safetensors")?;
//! let mut record = vec![0u8; 4096];
//! let mover = Mover::threads(8)?;
//! let read = Read {
//!     id: 7,
//!     fd: file.as_raw_fd(),
//!     offset: 1 << 20,
//!     dst: record.as_mut_ptr(),
//!     len: record.len(),
//! };
//! // SAFETY: `record` and `file` outlive the read, which is reaped below
//! unsafe { mover.submit(read)? };
//! mover.flush()?;
//! let done = mover.wait_done().expect("one read is pending");
//! assert_eq!(done.id, 7);
//! done.result?;
//! # Ok::<(), std::io::Error>(())
//! ```
//!
//! For direct I/O (`O_DIRECT`, or `F_NOCACHE` on macOS) the caller opens the
//! file that way and keeps offsets, lengths and destinations multiples of
//! [`DIRECT_ALIGN`].
#![cfg(unix)]

mod threads;
#[cfg(all(target_os = "linux", feature = "io-uring"))]
mod uring;

use std::{
    io,
    os::fd::RawFd,
    sync::{
        atomic::{AtomicUsize, Ordering},
        Mutex,
    },
};

/// The alignment direct I/O needs of file offsets, lengths and destinations
/// on the filesystems and disks in common use
pub const DIRECT_ALIGN: usize = 4096;

/// A read of `len` bytes of `fd` at `offset`, to `dst`
#[derive(Debug, Clone, Copy)]
pub struct Read {
    /// Handed back by the [`Done`] of this read
    pub id: u64,
    pub fd: RawFd,
    pub offset: u64,
    pub dst: *mut u8,
    pub len: usize,
}

// SAFETY: `Mover::submit` makes the caller vouch for `dst` until the read is
// reaped, wherever it runs
unsafe impl Send for Read {}

/// A finished read
#[derive(Debug)]
pub struct Done {
    /// The [`Read::id`] it was queued under
    pub id: u64,
    /// `Ok` once all the bytes landed. A file that ends before them is an
    /// [`io::ErrorKind::UnexpectedEof`].
    pub result: io::Result<()>,
}

/// Runs queued reads and hands them back as they finish, in any order.
/// Dropping a mover waits for the reads still queued or running.
pub struct Mover {
    backend: Backend,
    /// reads queued and not reaped yet
    pending: AtomicUsize,
    /// held by the thread that reaps
    reaper: Mutex<()>,
}

enum Backend {
    Threads(threads::Threads),
    #[cfg(all(target_os = "linux", feature = "io-uring"))]
    Uring(Box<uring::Uring>),
}

impl Mover {
    /// A mover whose reads are `pread`s on `threads` threads: as many reads
    /// run at once.
    pub fn threads(threads: usize) -> io::Result<Self> {
        Ok(Self::new(Backend::Threads(threads::Threads::new(threads)?)))
    }

    /// A mover whose reads go through one io_uring holding up to `depth` reads
    /// in the kernel at once. Fails where io_uring is absent or forbidden, as
    /// under the default seccomp profile of most container runtimes: fall back
    /// to [`Self::threads`].
    #[cfg(all(target_os = "linux", feature = "io-uring"))]
    pub fn uring(depth: u32) -> io::Result<Self> {
        Ok(Self::new(Backend::Uring(Box::new(uring::Uring::new(
            depth,
        )?))))
    }

    fn new(backend: Backend) -> Self {
        Self {
            backend,
            pending: AtomicUsize::new(0),
            reaper: Mutex::new(()),
        }
    }

    /// Registers `len` bytes at `base` with the kernel: reads into them skip the
    /// page pinning each read otherwise pays. The memory counts against the
    /// process's locked-memory limit, and the call fails past it. No-op for
    /// [`Self::threads`].
    ///
    /// # Safety
    ///
    /// The memory stays valid for writes until the mover is dropped.
    pub unsafe fn register(&mut self, base: *mut u8, len: usize) -> io::Result<()> {
        match &mut self.backend {
            Backend::Threads(_) => {
                let _ = (base, len);
                Ok(())
            }
            #[cfg(all(target_os = "linux", feature = "io-uring"))]
            // SAFETY: the caller vouches for the memory
            Backend::Uring(uring) => unsafe { uring.register(base, len) },
        }
    }

    /// Queues `read`. It may start at once or at the next [`Self::flush`].
    ///
    /// # Safety
    ///
    /// Until the read is reaped ([`Self::try_done`], [`Self::wait_done`]) or
    /// the mover is dropped: `read.dst` is valid for writes of `read.len`
    /// bytes, nothing else reads or writes them, and `read.fd` stays open.
    pub unsafe fn submit(&self, read: Read) -> io::Result<()> {
        self.pending.fetch_add(1, Ordering::AcqRel);
        let queued = match &self.backend {
            Backend::Threads(threads) => {
                threads.submit(read);
                Ok(())
            }
            #[cfg(all(target_os = "linux", feature = "io-uring"))]
            Backend::Uring(uring) => uring.submit(read),
        };
        if queued.is_err() {
            self.pending.fetch_sub(1, Ordering::AcqRel);
        }
        queued
    }

    /// Starts the reads queued so far.
    pub fn flush(&self) -> io::Result<()> {
        match &self.backend {
            Backend::Threads(_) => Ok(()),
            #[cfg(all(target_os = "linux", feature = "io-uring"))]
            Backend::Uring(uring) => uring.flush(),
        }
    }

    /// How many reads are queued or running, and not reaped
    pub fn pending(&self) -> usize {
        self.pending.load(Ordering::Acquire)
    }

    /// A finished read, if one is. `None` too while another thread waits in
    /// [`Self::wait_done`]: the reads that finish are its own to reap.
    pub fn try_done(&self) -> Option<Done> {
        let _reaper = self.reaper.try_lock().ok()?;
        let done = match &self.backend {
            Backend::Threads(threads) => threads.try_done(),
            #[cfg(all(target_os = "linux", feature = "io-uring"))]
            Backend::Uring(uring) => uring.try_done(),
        }?;
        self.pending.fetch_sub(1, Ordering::AcqRel);
        Some(done)
    }

    /// The next finished read, waiting for it; `None` when none is pending.
    /// Starts the reads queued so far. One thread waits at a time: another
    /// one waits for its turn.
    ///
    /// # Panics
    ///
    /// If an io_uring fails with reads in the kernel: their destinations could
    /// not be handed back safely.
    pub fn wait_done(&self) -> Option<Done> {
        let _reaper = self.reaper.lock().unwrap();
        // a read pending now stays so until this thread reaps it
        if self.pending() == 0 {
            return None;
        }
        let done = match &self.backend {
            Backend::Threads(threads) => threads.wait_done(),
            #[cfg(all(target_os = "linux", feature = "io-uring"))]
            Backend::Uring(uring) => uring.wait_done(),
        };
        self.pending.fetch_sub(1, Ordering::AcqRel);
        Some(done)
    }
}

impl Drop for Mover {
    fn drop(&mut self) {
        // the kernel or a thread may still write to a destination
        while self.wait_done().is_some() {}
    }
}

/// `pread`s until `read` is whole.
fn pread_all(read: &Read) -> io::Result<()> {
    let mut got = 0;
    while got < read.len {
        let offset = libc::off_t::try_from(read.offset + got as u64)
            .map_err(|_| io::Error::from(io::ErrorKind::InvalidInput))?;
        // SAFETY: `Mover::submit` requires the destination valid for
        // `read.len` bytes and the descriptor open
        let n = unsafe { libc::pread(read.fd, read.dst.add(got).cast(), read.len - got, offset) };
        match n {
            0 => return Err(io::ErrorKind::UnexpectedEof.into()),
            n if n > 0 => got += n as usize,
            _ => {
                let err = io::Error::last_os_error();
                if err.kind() != io::ErrorKind::Interrupted {
                    return Err(err);
                }
            }
        }
    }
    Ok(())
}
