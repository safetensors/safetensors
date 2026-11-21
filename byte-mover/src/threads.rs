//! Reads as `pread`s on a pool of threads

use std::{
    collections::VecDeque,
    io,
    sync::{Arc, Condvar, Mutex},
    thread::JoinHandle,
    time::{Duration, Instant},
};

use crate::{pread_all, Done, Read};

pub(crate) struct Threads {
    shared: Arc<Shared>,
    threads: Vec<JoinHandle<()>>,
}

struct Shared {
    state: Mutex<State>,
    /// a read was queued, or the pool closed
    queued: Condvar,
    /// a read finished, or the waiter was woken
    finished: Condvar,
}

#[derive(Default)]
struct State {
    queue: VecDeque<Read>,
    done: VecDeque<Done>,
    /// the threads end once the queue is empty
    closed: bool,
    /// set by `wake`, cleared by the `wait` it ends
    woken: bool,
}

impl Threads {
    pub(crate) fn new(threads: usize) -> io::Result<Self> {
        let shared = Arc::new(Shared {
            state: Mutex::default(),
            queued: Condvar::new(),
            finished: Condvar::new(),
        });
        let mut pool = Self {
            shared,
            threads: Vec::with_capacity(threads.max(1)),
        };
        for _ in 0..threads.max(1) {
            let shared = pool.shared.clone();
            // on failure, dropping the pool ends the threads already started
            let thread = std::thread::Builder::new()
                .name("byte-mover".into())
                .spawn(move || shared.work())?;
            pool.threads.push(thread);
        }
        Ok(pool)
    }

    pub(crate) fn submit(&self, read: Read) {
        self.shared.state.lock().unwrap().queue.push_back(read);
        self.shared.queued.notify_one();
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
}

impl Threads {
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
    fn work(&self) {
        let mut state = self.state.lock().unwrap();
        loop {
            let Some(read) = state.queue.pop_front() else {
                if state.closed {
                    return;
                }
                state = self.queued.wait(state).unwrap();
                continue;
            };
            drop(state);
            let done = Done {
                id: read.id,
                result: pread_all(&read),
            };
            state = self.state.lock().unwrap();
            state.done.push_back(done);
            self.finished.notify_one();
        }
    }
}

impl Drop for Threads {
    fn drop(&mut self) {
        self.shared.state.lock().unwrap().closed = true;
        self.shared.queued.notify_all();
        for thread in self.threads.drain(..) {
            let _ = thread.join();
        }
    }
}
