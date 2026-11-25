//! Every transfer between a file and memory is one pipeline: a worker takes a
//! slab from a [`SlabPool`], fills it with a chunk of the file from a
//! [`Source`], and drains it to a [`Sink`], until the [`Plan`]'s chunks run out.

use std::{
    fmt::Display,
    num::NonZeroUsize,
    ops::Range,
    sync::{
        atomic::{AtomicBool, AtomicUsize, Ordering},
        Condvar, Mutex, OnceLock,
    },
};

use super::pool::{Slab, SlabPool};

/// The most bytes a worker moves at once
pub const CHUNK_SIZE: NonZeroUsize = NonZeroUsize::new(16 * 1024 * 1024).unwrap();

/// Why a transfer failed
#[derive(Debug)]
pub enum Error {
    Io(std::io::Error),
    /// A source or sink failed
    Other(Box<dyn std::error::Error + Send + Sync>),
    /// The transfer was stopped before it reached this chunk
    Stopped,
}

impl Display for Error {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Io(e) => write!(f, "io error: {e}"),
            Self::Other(e) => write!(f, "{e}"),
            Self::Stopped => f.write_str("transfer stopped"),
        }
    }
}

impl std::error::Error for Error {}

impl From<std::io::Error> for Error {
    fn from(value: std::io::Error) -> Self {
        Self::Io(value)
    }
}

/// A chunk's bytes in a slab: the bytes of `file` start `head` bytes into it.
pub struct Staged {
    slab: Slab,
    head: usize,
    file: Range<u64>,
}

impl Staged {
    pub fn new(slab: Slab, head: usize, file: Range<u64>) -> Self {
        assert!(head + (file.end - file.start) as usize <= slab.len());
        Self { slab, head, file }
    }

    /// The range of the file these bytes come from
    pub fn file(&self) -> &Range<u64> {
        &self.file
    }

    pub fn bytes(&self) -> &[u8] {
        let len = (self.file.end - self.file.start) as usize;
        &self.slab[self.head..self.head + len]
    }

    /// The slab, to keep until the bytes are no longer read
    pub fn into_slab(self) -> Slab {
        self.slab
    }
}

/// Where a transfer reads chunks from
pub trait Source: Send + Sync {
    /// Fills `slab` with the bytes of `file`.
    fn fill(&self, file: Range<u64>, slab: Slab) -> Result<Staged, Error>;
}

/// Where a transfer writes chunks to
pub trait Sink: Send + Sync {
    /// Waits until the bytes of `file` can be drained. A worker calls it before
    /// taking a slab for them.
    fn wait_ready(&self, _file: &Range<u64>) -> Result<(), Error> {
        Ok(())
    }

    /// Writes a staged chunk, keeping its slab as long as it reads from it.
    fn drain(&self, staged: Staged) -> Result<(), Error>;

    /// The transfer stopped: a worker in [`Self::wait_ready`] returns
    /// [`Error::Stopped`].
    fn stop(&self) {}
}

/// The chunks of one transfer, claimed in order by its workers, and its
/// first error.
pub struct Plan {
    chunks: Box<[Range<u64>]>,
    next: AtomicUsize,
    stopped: AtomicBool,
    /// guards `next` advancing and `stopped` turning on, so neither is missed by
    /// a waiter on `progress`
    progress_lock: Mutex<()>,
    progress: Condvar,
    error: OnceLock<Error>,
}

impl Plan {
    pub fn new(chunks: Box<[Range<u64>]>) -> Self {
        Self {
            chunks,
            next: AtomicUsize::new(0),
            stopped: AtomicBool::new(false),
            progress_lock: Mutex::new(()),
            progress: Condvar::new(),
            error: OnceLock::new(),
        }
    }

    /// How many chunks the workers claimed: at least the ones being read or
    /// waiting to be
    pub fn claimed(&self) -> usize {
        self.next.load(Ordering::Relaxed)
    }

    /// Workers return before their next chunk.
    pub fn stop(&self) {
        self.stopped.store(true, Ordering::Release);
        self.notify();
    }

    fn notify(&self) {
        drop(self.progress_lock.lock().unwrap());
        self.progress.notify_all();
    }

    /// Waits until `done` holds or the plan stopped; `done` is checked again
    /// each time a chunk is claimed.
    pub fn wait_until(&self, done: impl Fn() -> bool) {
        let mut g = self.progress_lock.lock().unwrap();
        while !done() && !self.stopped() {
            g = self.progress.wait(g).unwrap();
        }
    }

    pub fn stopped(&self) -> bool {
        self.stopped.load(Ordering::Acquire)
    }

    /// Records `error` unless one is already, then stops.
    pub fn fail(&self, error: Error) {
        let _ = self.error.set(error);
        self.stop();
    }

    pub fn error(&self) -> Option<&Error> {
        self.error.get()
    }
}

/// The worker loop: moves the next chunk of `plan` from `from` to `to`, until
/// none is left or the plan stops. A failure fails the plan and stops `to`.
pub fn work(plan: &Plan, from: &impl Source, to: &impl Sink, pool: &'static SlabPool) {
    loop {
        if plan.stopped() {
            return;
        }
        let c = plan.next.fetch_add(1, Ordering::Relaxed);
        plan.notify();
        let Some(file) = plan.chunks.get(c) else {
            return;
        };
        let moved = to
            .wait_ready(file)
            .and_then(|()| pool.acquire())
            .and_then(|slab| from.fill(file.clone(), slab))
            .and_then(|staged| to.drain(staged));
        match moved {
            Ok(()) => {}
            Err(Error::Stopped) => return,
            Err(e) => {
                plan.fail(e);
                to.stop();
                return;
            }
        }
    }
}
