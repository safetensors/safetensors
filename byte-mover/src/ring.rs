//! A thread of the pool that runs its reads through an io_uring of its own

use std::io;

use io_uring::{opcode, types, IoUring};

use crate::{pool::Shared, Done, Read};

/// The kernel takes registered buffers of at most 1 GiB each
const PIECE: usize = 1 << 30;

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

pub(crate) struct Ring {
    ring: IoUring,
    /// the reads in the ring, by the index the ring knows them under
    slots: Vec<Option<Inflight>>,
    free: Vec<usize>,
    /// the memory this ring registered, in pieces of [`PIECE`]
    registered: Option<(usize, usize)>,
    /// the last [`Registration::asked`] answered
    answered: u64,
}

struct Inflight {
    read: Read,
    /// bytes landed so far: a read may come back short
    got: usize,
}

/// Ends the process if the thread unwinds: the kernel may still write to the
/// destinations of its reads, which nothing can hand back any more.
struct AbortOnPanic;

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
        Ok(Self {
            ring: IoUring::new(depth)?,
            slots: (0..depth).map(|_| None).collect(),
            free: (0..depth as usize).collect(),
            registered: None,
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
        loop {
            let mut state = shared.state.lock().unwrap();
            if !finished.is_empty() {
                state.done.extend(finished.drain(..));
                shared.finished.notify_one();
            }
            loop {
                if state.registration.asked != self.answered {
                    let (memory, asked) = (state.registration.memory, state.registration.asked);
                    drop(state);
                    let refused = self.register(memory).err();
                    self.answered = asked;
                    state = shared.state.lock().unwrap();
                    state.registration.answers += 1;
                    if state.registration.refused.is_none() {
                        state.registration.refused = refused;
                    }
                    shared.finished.notify_all();
                    continue;
                }
                let take = state.queue.len().div_ceil(rings).min(self.free.len());
                for read in state.queue.drain(..take) {
                    let slot = self.free.pop().expect("a free slot");
                    self.slots[slot] = Some(Inflight { read, got: 0 });
                    taken.push(slot);
                }
                if self.free.len() < self.slots.len() {
                    break;
                }
                if state.closed {
                    return;
                }
                state = shared.queued.wait(state).unwrap();
            }
            drop(state);
            for slot in taken.drain(..) {
                self.push(slot);
            }
            match self.ring.submit_and_wait(1) {
                Ok(_) => {}
                // `EBUSY`: completions to reap first
                Err(err) if matches!(err.raw_os_error(), Some(libc::EINTR | libc::EBUSY)) => {}
                Err(err) => panic!("io_uring failed with reads in flight: {err}"),
            }
            self.reap(&mut finished);
        }
    }

    /// Registers `memory` in place of what was registered. A ring that cannot
    /// reads into it as into any other memory.
    fn register(&mut self, memory: Option<(usize, usize)>) -> io::Result<()> {
        if self.registered.take().is_some() {
            self.ring.submitter().unregister_buffers()?;
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
        // is dropped, which ends this ring
        unsafe { self.ring.submitter().register_buffers(&iovecs) }?;
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

    /// Queues what is left of the read in `slot`.
    fn push(&mut self, slot: usize) {
        let Inflight { read, got } = self.slots[slot].as_ref().expect("in flight");
        let (dst, len) = (read.dst.wrapping_add(*got), read.len - got);
        // one request reads at most `u32::MAX` bytes: the rest follows as a short read
        let chunk = len.min(u32::MAX as usize) as u32;
        let (fd, offset) = (types::Fd(read.fd), read.offset + *got as u64);
        let entry = match self.piece(dst, chunk as usize) {
            Some(piece) => opcode::ReadFixed::new(fd, dst, chunk, piece)
                .offset(offset)
                .build(),
            None => opcode::Read::new(fd, dst, chunk).offset(offset).build(),
        }
        .user_data(slot as u64);
        // SAFETY: `Mover::submit` requires the destination and the descriptor
        // valid until the read is reaped, and dropping the mover waits for it
        while unsafe { self.ring.submission().push(&entry) }.is_err() {
            if let Err(err) = self.ring.submit() {
                panic!("io_uring failed with reads in flight: {err}");
            }
        }
    }

    /// Moves the reads that ended to `finished`; short ones go back in the ring.
    fn reap(&mut self, finished: &mut Vec<Done>) {
        loop {
            let Some(completion) = self.ring.completion().next() else {
                return;
            };
            let slot = completion.user_data() as usize;
            let inflight = self.slots[slot].as_mut().expect("in flight");
            let result = match completion.result() {
                0 if inflight.read.len > 0 => Err(io::ErrorKind::UnexpectedEof.into()),
                n if n >= 0 => {
                    inflight.got += n as usize;
                    if inflight.got < inflight.read.len {
                        self.push(slot);
                        continue;
                    }
                    Ok(())
                }
                errno if errno == -libc::EINTR || errno == -libc::EAGAIN => {
                    self.push(slot);
                    continue;
                }
                errno => Err(io::Error::from_raw_os_error(-errno)),
            };
            let id = self.slots[slot].take().expect("in flight").read.id;
            self.free.push(slot);
            finished.push(Done { id, result });
        }
    }
}
