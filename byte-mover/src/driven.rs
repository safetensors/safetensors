//! An io_uring driven by the threads that queue and reap. A read starts in the
//! kernel when it is queued and its completion wakes the thread that waits
//! for it: no thread of the mover's stands in between.

use std::{
    collections::VecDeque,
    io,
    sync::{Mutex, MutexGuard},
    time::{Duration, Instant},
};

use io_uring::{opcode, squeue, types, IoUring};

use crate::{
    ring::{AbortOnPanic, Slots, WAKE},
    Done, Read,
};

pub(crate) struct Driven {
    ring: IoUring,
    /// held to touch the ring's queues, never across a wait in the kernel
    state: Mutex<State>,
}

struct State {
    slots: Slots,
    /// reads queued while the ring is full
    queue: VecDeque<Read>,
    done: VecDeque<Done>,
    /// a completion that ends a wait is on its way
    waking: bool,
    /// set by that completion, cleared by the `wait` it ends
    woken: bool,
}

impl Driven {
    pub(crate) fn new(depth: u32) -> io::Result<Self> {
        let depth = depth.max(1);
        Ok(Self {
            ring: IoUring::new(depth)?,
            state: Mutex::new(State {
                slots: Slots::new(depth),
                queue: VecDeque::new(),
                done: VecDeque::new(),
                waking: false,
                woken: false,
            }),
        })
    }

    /// # Safety
    ///
    /// `len` bytes at `base` stay valid for writes until the ring is dropped.
    pub(crate) unsafe fn register(&mut self, base: *mut u8, len: usize) -> io::Result<()> {
        let mut state = self.reaped();
        while !state.slots.idle() {
            drop(state);
            self.enter(None);
            state = self.reaped();
        }
        let memory = Some((base as usize, len));
        state.slots.register(&self.ring.submitter(), memory)
    }

    pub(crate) fn submit(&self, read: Read) {
        let mut state = self.state.lock().unwrap();
        if state.slots.free() == 0 {
            state.queue.push_back(read);
            return;
        }
        let slot = state.slots.take(read);
        self.push(&state.slots.entry(slot));
        self.flush();
    }

    pub(crate) fn try_done(&self) -> Option<Done> {
        self.reaped().done.pop_front()
    }

    /// Call with a read pending, from the one thread that reaps.
    pub(crate) fn wait_done(&self) -> Done {
        loop {
            if let Some(done) = self.reaped().done.pop_front() {
                return done;
            }
            self.enter(None);
        }
    }

    pub(crate) fn wait(&self, out: &mut Vec<Done>, max: usize, timeout: Option<Duration>) {
        let deadline = timeout.map(|timeout| Instant::now() + timeout);
        loop {
            let mut state = self.reaped();
            let take = state.done.len().min(max);
            if take > 0 || std::mem::take(&mut state.woken) {
                out.extend(state.done.drain(..take));
                return;
            }
            drop(state);
            let left = match deadline {
                None => None,
                Some(deadline) => match deadline.checked_duration_since(Instant::now()) {
                    Some(left) if !left.is_zero() => Some(left),
                    _ => return,
                },
            };
            self.enter(left);
        }
    }

    pub(crate) fn wake(&self) {
        let mut state = self.state.lock().unwrap();
        if !std::mem::replace(&mut state.waking, true) {
            self.push(&opcode::Nop::new().build().user_data(WAKE));
            self.flush();
        }
    }

    /// The state, once the completions in the ring went through it: reads
    /// that ended are in `done`, short ones and queued ones are in the ring.
    fn reaped(&self) -> MutexGuard<'_, State> {
        let mut state = self.state.lock().unwrap();
        let mut pushed = false;
        loop {
            // SAFETY: the lock on `state` is held, as by every user of the queue
            let Some(completion) = unsafe { self.ring.completion_shared() }.next() else {
                break;
            };
            if completion.user_data() == WAKE {
                state.waking = false;
                state.woken = true;
                continue;
            }
            let mut slot = completion.user_data() as usize;
            if let Some(done) = state.slots.land(slot, completion.result()) {
                state.done.push_back(done);
                let Some(read) = state.queue.pop_front() else {
                    continue;
                };
                slot = state.slots.take(read);
            }
            self.push(&state.slots.entry(slot));
            pushed = true;
        }
        if pushed {
            self.flush();
        }
        state
    }

    /// Queues `entry` for the next [`Self::flush`]. Call with `state` locked.
    fn push(&self, entry: &squeue::Entry) {
        // SAFETY: the lock on `state` is held, as by every user of the queue.
        // `Mover::submit` requires the destination and the descriptor of a
        // read valid until it is reaped, and dropping the mover waits for it.
        while unsafe { self.ring.submission_shared().push(entry) }.is_err() {
            self.flush();
        }
    }

    /// Hands the queued requests to the kernel, without waiting.
    fn flush(&self) {
        self.entered(self.ring.submit());
    }

    /// Waits in the kernel for a completion, at most `timeout`. Call with
    /// `state` unlocked: other threads queue reads meanwhile.
    fn enter(&self, timeout: Option<Duration>) {
        let submitter = self.ring.submitter();
        self.entered(match timeout {
            None => submitter.submit_and_wait(1),
            Some(timeout) => {
                let timeout = types::Timespec::new()
                    .sec(timeout.as_secs())
                    .nsec(timeout.subsec_nanos());
                submitter.submit_with_args(1, &types::SubmitArgs::new().timespec(&timeout))
            }
        });
    }

    fn entered(&self, entered: io::Result<usize>) {
        match entered {
            Ok(_) => {}
            // `EBUSY`: completions to reap first. `ETIME`: the timeout.
            Err(err)
                if matches!(
                    err.raw_os_error(),
                    Some(libc::EINTR | libc::EBUSY | libc::ETIME)
                ) => {}
            Err(err) => {
                let _abort = AbortOnPanic;
                panic!("io_uring failed with reads in flight: {err}");
            }
        }
    }
}
