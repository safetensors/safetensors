//! Reads byte ranges of files into memory the caller owns.
//!
//! Reads are queued on a [`Mover`], run by a pool of threads that `pread` or
//! drive an io_uring each, or by an io_uring the calling threads drive, and
//! reaped one by one, each with the id it was queued under. Any thread may
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
//! unsafe { mover.submit(read) };
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

#[cfg(all(target_os = "linux", feature = "io-uring"))]
mod driven;
mod pool;
#[cfg(all(target_os = "linux", feature = "io-uring"))]
mod ring;

use std::{
    io,
    os::fd::RawFd,
    sync::{
        atomic::{AtomicUsize, Ordering},
        Mutex,
    },
    time::Duration,
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
    Pool(pool::Pool),
    #[cfg(all(target_os = "linux", feature = "io-uring"))]
    Driven(Box<driven::Driven>),
}

/// Calls the same method on the pool or the driven ring.
macro_rules! backend {
    ($backend:expr, $name:ident => $call:expr) => {
        match $backend {
            Backend::Pool($name) => $call,
            #[cfg(all(target_os = "linux", feature = "io-uring"))]
            Backend::Driven($name) => $call,
        }
    };
}

impl Mover {
    /// A mover whose reads are `pread`s on `threads` threads: as many reads
    /// run at once.
    pub fn threads(threads: usize) -> io::Result<Self> {
        Ok(Self::new(Backend::Pool(pool::Pool::threads(threads)?)))
    }

    /// A mover whose reads go through `rings` io_urings, a thread each, holding
    /// up to `depth` reads in the kernel each. Fails where io_uring is absent
    /// or forbidden, as under the default seccomp profile of most container
    /// runtimes: fall back to [`Self::threads`].
    #[cfg(all(target_os = "linux", feature = "io-uring"))]
    pub fn rings(rings: usize, depth: u32) -> io::Result<Self> {
        Ok(Self::new(Backend::Pool(pool::Pool::rings(rings, depth)?)))
    }

    /// A mover whose reads go through one io_uring that the calling threads
    /// drive, holding up to `depth` reads in the kernel. A read starts in
    /// [`Self::submit`] and the thread that waits for it sleeps in the kernel
    /// until it finished: no thread stands between the two, which is what a
    /// read costs most when few are in flight. Reads past `depth` start as
    /// finished ones are reaped. Needs Linux 5.11; fails as [`Self::rings`] does.
    #[cfg(all(target_os = "linux", feature = "io-uring"))]
    pub fn ring(depth: u32) -> io::Result<Self> {
        Ok(Self::new(Backend::Driven(Box::new(driven::Driven::new(
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

    /// Registers `len` bytes at `base` with the kernel, once per ring: reads
    /// into them skip the page pinning each read otherwise pays. The memory
    /// counts against the process's locked-memory limit once per ring, and
    /// the call fails past it: the rings that could not register read into
    /// the memory as into any other. Replaces the memory registered before.
    /// Waits for the reads running to finish. No-op for [`Self::threads`].
    ///
    /// # Safety
    ///
    /// The memory stays valid for writes until the mover is dropped.
    pub unsafe fn register(&mut self, base: *mut u8, len: usize) -> io::Result<()> {
        // SAFETY: the caller vouches for the memory
        backend!(&mut self.backend, backend => unsafe { backend.register(base, len) })
    }

    /// Queues `read`, which starts as soon as a thread or the ring is free.
    ///
    /// # Safety
    ///
    /// Until the read is reaped ([`Self::try_done`], [`Self::wait_done`],
    /// [`Self::wait`]) or the mover is dropped: `read.dst` is valid for writes
    /// of `read.len` bytes, nothing else reads or writes them, and `read.fd`
    /// stays open.
    pub unsafe fn submit(&self, read: Read) {
        self.pending.fetch_add(1, Ordering::AcqRel);
        backend!(&self.backend, backend => backend.submit(read));
    }

    /// How many reads are queued or running, and not reaped
    pub fn pending(&self) -> usize {
        self.pending.load(Ordering::Acquire)
    }

    /// A finished read, if one is. `None` too while another thread waits in
    /// [`Self::wait`] or [`Self::wait_done`]: the reads that finish are its own
    /// to reap.
    pub fn try_done(&self) -> Option<Done> {
        let _reaper = self.reaper.try_lock().ok()?;
        let done = backend!(&self.backend, backend => backend.try_done())?;
        self.pending.fetch_sub(1, Ordering::AcqRel);
        Some(done)
    }

    /// The next finished read, waiting for it; `None` when none is pending.
    /// One thread waits at a time: another one waits for its turn.
    pub fn wait_done(&self) -> Option<Done> {
        let _reaper = self.reaper.lock().unwrap();
        // a read pending now stays so until this thread reaps it
        if self.pending() == 0 {
            return None;
        }
        let done = backend!(&self.backend, backend => backend.wait_done());
        self.pending.fetch_sub(1, Ordering::AcqRel);
        Some(done)
    }

    /// Waits until a read finished, [`Self::wake`] was called or `timeout`
    /// elapsed, then moves up to `max` finished reads to `out` and returns how
    /// many. Unlike [`Self::wait_done`] it waits with no read pending too: a
    /// thread that only reaps can sleep here until another one queues a read
    /// or wakes it. One thread waits at a time.
    pub fn wait(&self, out: &mut Vec<Done>, max: usize, timeout: Option<Duration>) -> usize {
        let _reaper = self.reaper.lock().unwrap();
        let before = out.len();
        backend!(&self.backend, backend => backend.wait(out, max, timeout));
        let reaped = out.len() - before;
        self.pending.fetch_sub(reaped, Ordering::AcqRel);
        reaped
    }

    /// Makes the thread in [`Self::wait`], or the next one to call it, return
    /// even with no read finished.
    pub fn wake(&self) {
        backend!(&self.backend, backend => backend.wake());
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
