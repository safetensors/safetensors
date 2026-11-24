//! A thread of the pool that runs its reads through an io_uring of its own

use std::{
    fs::File,
    io,
    os::fd::{AsRawFd, FromRawFd},
    sync::Arc,
};

use io_uring::{opcode, squeue, types, IoUring, Submitter};

use crate::{
    pool::{Shared, State},
    Done, Read,
};

/// The kernel takes registered buffers of at most 1 GiB each
const PIECE: usize = 1 << 30;

/// What a completion that only ends a wait carries, in place of a slot
pub(crate) const WAKE: u64 = u64::MAX;

/// How many times a ring waits for the reads it submits before it checks
/// again whether they finish as they are submitted
const RECHECK: u32 = 32;

/// Memory the rings are asked to register, each for itself
#[derive(Default)]
pub(crate) struct Registration {
    /// its start and length
    memory: Option<(usize, usize)>,
    /// counts the requests: a ring answers the ones it has not seen
    asked: u64,
    /// how many rings answered the last request
    pub(crate) answers: usize,
    /// why one of them could not
    pub(crate) refused: Option<io::Error>,
}

impl Registration {
    pub(crate) fn ask(&mut self, base: *mut u8, len: usize) {
        self.memory = Some((base as usize, len));
        self.asked += 1;
        self.answers = 0;
        self.refused = None;
    }
}

/// The reads in a ring and the memory registered with it
pub(crate) struct Slots {
    /// the reads, by the index the ring knows them under
    reads: Vec<Option<Inflight>>,
    free: Vec<usize>,
    /// the memory registered, in pieces of [`PIECE`]
    registered: Option<(usize, usize)>,
}

pub(crate) struct Ring {
    ring: IoUring,
    slots: Slots,
    /// the last [`Registration::asked`] answered
    answered: u64,
    /// an eventfd: a write to it ends the ring's wait in the kernel, for a
    /// read queued while the ring has room for it
    wake: Arc<File>,
    /// whether the ring polls `wake`
    armed: bool,
    /// whether `wake` is in [`State::parked`]
    parked: bool,
}

struct Inflight {
    read: Read,
    /// bytes landed so far: a read may come back short
    got: usize,
}

/// Ends the process if the thread unwinds: the kernel may still write to the
/// destinations of its reads, which nothing can hand back any more.
pub(crate) struct AbortOnPanic;

impl Drop for AbortOnPanic {
    fn drop(&mut self) {
        if std::thread::panicking() {
            std::process::abort();
        }
    }
}

impl Ring {
    pub(crate) fn new(depth: u32) -> io::Result<Self> {
        let depth = depth.max(1);
        // SAFETY: no pointer is involved
        let wake = unsafe { libc::eventfd(0, libc::EFD_CLOEXEC | libc::EFD_NONBLOCK) };
        if wake < 0 {
            return Err(io::Error::last_os_error());
        }
        // SAFETY: a descriptor just created, that nothing else owns
        let wake = Arc::new(unsafe { File::from_raw_fd(wake) });
        Ok(Self {
            wake,
            armed: false,
            parked: false,
            ring: IoUring::new(depth)?,
            slots: Slots::new(depth),
            answered: 0,
        })
    }

    /// Runs queued reads until the pool closes, as many at once as the ring is
    /// deep. With few reads queued it leaves some to the other `rings - 1`:
    /// reads spread over the rings finish sooner than behind one another.
    pub(crate) fn work(mut self, shared: &Shared, rings: usize) {
        let _abort = AbortOnPanic;
        let mut finished = Vec::new();
        let mut taken = Vec::new();
        // Whether reads finish as they are submitted, as reads from the page
        // cache do. Nothing is left to wait for in the kernel then: listed
        // as parked, the ring would be woken for nothing by each read queued
        // while it copies.
        let mut inline = true;
        let mut waited = 0;
        loop {
            let mut state = shared.state.lock().unwrap();
            if std::mem::take(&mut self.parked) {
                state.parked.retain(|wake| !Arc::ptr_eq(wake, &self.wake));
            }
            if !finished.is_empty() {
                state.done.extend(finished.drain(..));
                shared.finished.notify_one();
            }
            let mut roused = None;
            loop {
                if state.registration.asked != self.answered {
                    let (memory, asked) = (state.registration.memory, state.registration.asked);
                    drop(state);
                    let refused = self.slots.register(&self.ring.submitter(), memory).err();
                    self.answered = asked;
                    state = shared.state.lock().unwrap();
                    state.registration.answers += 1;
                    if state.registration.refused.is_none() {
                        state.registration.refused = refused;
                    }
                    shared.finished.notify_all();
                    continue;
                }
                let take = state.queue.len().div_ceil(rings).min(self.slots.free());
                taken.extend(state.queue.drain(..take).map(|read| self.slots.take(read)));
                if take > 0 && !state.queue.is_empty() {
                    roused = shared.rouse(&mut state);
                }
                if !self.slots.idle() {
                    break;
                }
                if state.closed {
                    return;
                }
                state = shared.sleep(state);
            }
            let wait = taken.is_empty() || !inline;
            if wait && self.slots.free() > 0 {
                self.park(&mut state);
            }
            drop(state);
            shared.wake(roused);
            for &slot in &taken {
                self.push(slot);
            }
            match self.ring.submit_and_wait(wait as usize) {
                Ok(_) => {}
                // `EBUSY`: completions to reap first
                Err(err) if matches!(err.raw_os_error(), Some(libc::EINTR | libc::EBUSY)) => {}
                Err(err) => panic!("io_uring failed with reads in flight: {err}"),
            }
            self.reap(&mut finished);
            if !taken.is_empty() {
                if wait {
                    waited = (waited + 1) % RECHECK;
                    inline = waited == 0;
                } else {
                    inline = taken.iter().all(|&slot| self.slots.reads[slot].is_none());
                }
                taken.clear();
            }
        }
    }

    /// Lists the ring among those a queued read wakes from the kernel.
    fn park(&mut self, state: &mut State) {
        state.parked.push(self.wake.clone());
        self.parked = true;
        if !std::mem::replace(&mut self.armed, true) {
            let poll = opcode::PollAdd::new(types::Fd(self.wake.as_raw_fd()), libc::POLLIN as u32)
                .build()
                .user_data(WAKE);
            self.queue(&poll);
        }
    }

    fn queue(&mut self, entry: &squeue::Entry) {
        // SAFETY: `Mover::submit` requires the destination and the descriptor
        // of a read valid until it is reaped, and dropping the mover waits
        // for it; the poll is on a descriptor the ring owns
        while unsafe { self.ring.submission().push(entry) }.is_err() {
            if let Err(err) = self.ring.submit() {
                panic!("io_uring failed with reads in flight: {err}");
            }
        }
    }

    /// Queues what is left of the read in `slot`.
    fn push(&mut self, slot: usize) {
        let entry = self.slots.entry(slot);
        self.queue(&entry);
    }

    /// Moves the reads that ended to `finished`; short ones go back in the ring.
    fn reap(&mut self, finished: &mut Vec<Done>) {
        loop {
            let Some(completion) = self.ring.completion().next() else {
                return;
            };
            if completion.user_data() == WAKE {
                self.armed = false;
                let _ = io::Read::read(&mut &*self.wake, &mut [0; 8]);
                continue;
            }
            let slot = completion.user_data() as usize;
            match self.slots.land(slot, completion.result()) {
                Some(done) => finished.push(done),
                None => self.push(slot),
            }
        }
    }
}

impl Slots {
    pub(crate) fn new(depth: u32) -> Self {
        Self {
            reads: (0..depth).map(|_| None).collect(),
            free: (0..depth as usize).collect(),
            registered: None,
        }
    }

    /// How many more reads the ring takes
    pub(crate) fn free(&self) -> usize {
        self.free.len()
    }

    /// Whether the ring holds no read
    pub(crate) fn idle(&self) -> bool {
        self.free.len() == self.reads.len()
    }

    /// Gives `read` a slot, which must be [`Self::free`].
    pub(crate) fn take(&mut self, read: Read) -> usize {
        let slot = self.free.pop().expect("a free slot");
        self.reads[slot] = Some(Inflight { read, got: 0 });
        slot
    }

    /// Registers `memory` in place of what was registered. A ring that cannot
    /// reads into it as into any other memory.
    pub(crate) fn register(
        &mut self,
        submitter: &Submitter<'_>,
        memory: Option<(usize, usize)>,
    ) -> io::Result<()> {
        if self.registered.take().is_some() {
            submitter.unregister_buffers()?;
        }
        let Some((base, len)) = memory else {
            return Ok(());
        };
        let iovecs: Vec<libc::iovec> = (0..len.div_ceil(PIECE))
            .map(|i| libc::iovec {
                iov_base: (base + i * PIECE) as *mut libc::c_void,
                iov_len: PIECE.min(len - i * PIECE),
            })
            .collect();
        // SAFETY: `Mover::register` requires the memory valid until the mover
        // is dropped, which ends the ring
        unsafe { submitter.register_buffers(&iovecs) }?;
        self.registered = Some((base, len));
        Ok(())
    }

    /// The registered piece that holds all of `len` bytes at `dst`
    fn piece(&self, dst: *mut u8, len: usize) -> Option<u16> {
        let (base, registered) = self.registered?;
        let at = (dst as usize).checked_sub(base)?;
        let piece = at / PIECE;
        let within = at + len <= registered && (at + len - 1) / PIECE == piece;
        (len > 0 && within).then_some(piece as u16)
    }

    /// The request for what is left of the read in `slot`
    pub(crate) fn entry(&self, slot: usize) -> squeue::Entry {
        let Inflight { read, got } = self.reads[slot].as_ref().expect("in flight");
        let (dst, len) = (read.dst.wrapping_add(*got), read.len - got);
        // one request reads at most `u32::MAX` bytes: the rest follows as a short read
        let chunk = len.min(u32::MAX as usize) as u32;
        let (fd, offset) = (types::Fd(read.fd), read.offset + *got as u64);
        match self.piece(dst, chunk as usize) {
            Some(piece) => opcode::ReadFixed::new(fd, dst, chunk, piece)
                .offset(offset)
                .build(),
            None => opcode::Read::new(fd, dst, chunk).offset(offset).build(),
        }
        .user_data(slot as u64)
    }

    /// Takes what the request of `slot` completed with: the read once it
    /// ended, or `None` when what is left of it is to go back in the ring.
    pub(crate) fn land(&mut self, slot: usize, result: i32) -> Option<Done> {
        let inflight = self.reads[slot].as_mut().expect("in flight");
        let result = match result {
            0 if inflight.read.len > 0 => Err(io::ErrorKind::UnexpectedEof.into()),
            n if n >= 0 => {
                inflight.got += n as usize;
                if inflight.got < inflight.read.len {
                    return None;
                }
                Ok(())
            }
            errno if errno == -libc::EINTR || errno == -libc::EAGAIN => return None,
            errno => Err(io::Error::from_raw_os_error(-errno)),
        };
        let id = self.reads[slot].take().expect("in flight").read.id;
        self.free.push(slot);
        Some(Done { id, result })
    }
}
