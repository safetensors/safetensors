//! The threads that run the reads, and the queues they share

use std::{
    collections::VecDeque,
    fs::File,
    io,
    sync::{Arc, Condvar, Mutex, MutexGuard},
    thread::JoinHandle,
    time::{Duration, Instant},
};

use crate::{pread_all, Done, Read};

pub(crate) struct Pool {
    shared: Arc<Shared>,
    threads: Vec<JoinHandle<()>>,
    /// how many of the threads drive an io_uring: all of them or none
    #[cfg(all(target_os = "linux", feature = "io-uring"))]
    rings: usize,
}

pub(crate) struct Shared {
    pub(crate) state: Mutex<State>,
    /// a read was queued, memory is to be registered, or the pool closed
    pub(crate) queued: Condvar,
    /// a read finished, a ring answered a registration, or the waiter was woken
    pub(crate) finished: Condvar,
}

#[derive(Default)]
pub(crate) struct State {
    pub(crate) queue: VecDeque<Read>,
    pub(crate) done: VecDeque<Done>,
    /// the threads end once the queue is empty
    pub(crate) closed: bool,
    /// set by `wake`, cleared by the `wait` it ends
    woken: bool,
    /// how many threads sleep until a read is queued
    idle: usize,
    /// the rings that wait in the kernel with room for more reads, by the
    /// eventfd that wakes each
    pub(crate) parked: Vec<Arc<File>>,
    #[cfg(all(target_os = "linux", feature = "io-uring"))]
    pub(crate) registration: crate::ring::Registration,
}

impl Pool {
    fn new() -> Self {
        Self {
            shared: Arc::new(Shared {
                state: Mutex::default(),
                queued: Condvar::new(),
                finished: Condvar::new(),
            }),
            threads: Vec::new(),
            #[cfg(all(target_os = "linux", feature = "io-uring"))]
            rings: 0,
        }
    }

    /// On failure, dropping the pool ends the threads already started.
    fn spawn(&mut self, work: impl FnOnce(&Shared) + Send + 'static) -> io::Result<()> {
        let shared = self.shared.clone();
        let thread = std::thread::Builder::new()
            .name("byte-mover".into())
            .spawn(move || work(&shared))?;
        self.threads.push(thread);
        Ok(())
    }

    pub(crate) fn threads(threads: usize) -> io::Result<Self> {
        let mut pool = Self::new();
        for _ in 0..threads.max(1) {
            pool.spawn(Shared::pread)?;
        }
        Ok(pool)
    }

    #[cfg(all(target_os = "linux", feature = "io-uring"))]
    pub(crate) fn rings(rings: usize, depth: u32) -> io::Result<Self> {
        let rings = (0..rings.max(1))
            .map(|_| crate::ring::Ring::new(depth))
            .collect::<io::Result<Vec<_>>>()?;
        let mut pool = Self::new();
        pool.rings = rings.len();
        let count = rings.len();
        for ring in rings {
            pool.spawn(move |shared| ring.work(shared, count))?;
        }
        Ok(pool)
    }

    /// # Safety
    ///
    /// `len` bytes at `base` stay valid for writes until the pool is dropped.
    pub(crate) unsafe fn register(&mut self, base: *mut u8, len: usize) -> io::Result<()> {
        #[cfg(all(target_os = "linux", feature = "io-uring"))]
        if self.rings > 0 {
            let mut state = self.shared.state.lock().unwrap();
            state.registration.ask(base, len);
            self.shared.queued.notify_all();
            while state.registration.answers < self.rings {
                state = self.shared.finished.wait(state).unwrap();
            }
            return state.registration.refused.take().map_or(Ok(()), Err);
        }
        let _ = (base, len);
        Ok(())
    }

    pub(crate) fn submit(&self, read: Read) {
        let mut state = self.shared.state.lock().unwrap();
        state.queue.push_back(read);
        let parked = self.shared.rouse(&mut state);
        drop(state);
        wake(parked);
    }

    pub(crate) fn try_done(&self) -> Option<Done> {
        self.shared.state.lock().unwrap().done.pop_front()
    }

    /// Call with a read pending, from the one thread that reaps.
    pub(crate) fn wait_done(&self) -> Done {
        let mut state = self.shared.state.lock().unwrap();
        loop {
            if let Some(done) = state.done.pop_front() {
                return done;
            }
            state = self.shared.finished.wait(state).unwrap();
        }
    }

    pub(crate) fn wait(&self, out: &mut Vec<Done>, max: usize, timeout: Option<Duration>) {
        let deadline = timeout.map(|timeout| Instant::now() + timeout);
        let mut state = self.shared.state.lock().unwrap();
        loop {
            let take = state.done.len().min(max);
            if take > 0 || std::mem::take(&mut state.woken) {
                out.extend(state.done.drain(..take));
                return;
            }
            state = match deadline {
                None => self.shared.finished.wait(state).unwrap(),
                Some(deadline) => {
                    let Some(left) = deadline.checked_duration_since(Instant::now()) else {
                        return;
                    };
                    self.shared.finished.wait_timeout(state, left).unwrap().0
                }
            };
        }
    }

    pub(crate) fn wake(&self) {
        self.shared.state.lock().unwrap().woken = true;
        self.shared.finished.notify_all();
    }
}

impl Shared {
    /// Runs queued reads as `pread`s, one at a time.
    fn pread(&self) {
        let mut state = self.state.lock().unwrap();
        loop {
            let Some(read) = state.queue.pop_front() else {
                if state.closed {
                    return;
                }
                state = self.sleep(state);
                continue;
            };
            let parked = match state.queue.is_empty() {
                true => None,
                false => self.rouse(&mut state),
            };
            drop(state);
            wake(parked);
            let done = Done {
                id: read.id,
                result: pread_all(&read),
            };
            state = self.state.lock().unwrap();
            state.done.push_back(done);
            self.finished.notify_one();
        }
    }

    /// Gets a thread that waits for reads to take one: a thread asleep, or a
    /// ring that waits in the kernel for the reads it runs. The ring is for
    /// the caller to [`wake`] once the state is unlocked.
    #[must_use]
    pub(crate) fn rouse(&self, state: &mut State) -> Option<Arc<File>> {
        if state.idle > 0 {
            self.queued.notify_one();
            return None;
        }
        state.parked.pop()
    }

    /// Sleeps until a read is queued, memory is to be registered, or the pool
    /// closed.
    pub(crate) fn sleep<'a>(&'a self, mut state: MutexGuard<'a, State>) -> MutexGuard<'a, State> {
        state.idle += 1;
        state = self.queued.wait(state).unwrap();
        state.idle -= 1;
        state
    }
}

/// Ends the wait in the kernel of the ring [`Shared::rouse`] picked.
pub(crate) fn wake(parked: Option<Arc<File>>) {
    if let Some(parked) = parked {
        let _ = io::Write::write(&mut &*parked, &1u64.to_ne_bytes());
    }
}

impl Drop for Pool {
    fn drop(&mut self) {
        self.shared.state.lock().unwrap().closed = true;
        self.shared.queued.notify_all();
        for thread in self.threads.drain(..) {
            let _ = thread.join();
        }
    }
}
