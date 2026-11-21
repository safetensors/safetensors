//! Reads through one io_uring

use std::{
    io,
    sync::{
        atomic::{AtomicBool, Ordering},
        Mutex,
    },
    time::{Duration, Instant},
};

use io_uring::{opcode, types, IoUring};

use crate::{Done, Read};

/// The kernel takes registered buffers of at most 1 GiB each
const PIECE: usize = 1 << 30;

/// The `user_data` of the no-op `wake` queues to end a wait
const WAKE: u64 = u64::MAX;

pub(crate) struct Uring {
    ring: IoUring,
    /// guards the ring's submission queue too
    table: Mutex<Table>,
    /// guards the ring's completion queue; taken before `table`
    completions: Mutex<()>,
    /// registered memory: its start and length, in pieces of [`PIECE`]
    registered: Option<(usize, usize)>,
    /// set by `wake`, cleared by the `wait` it ends
    woken: AtomicBool,
}

/// The reads in the ring, by the index the ring knows them under
#[derive(Default)]
struct Table {
    inflight: Vec<Option<Inflight>>,
    free: Vec<usize>,
}

struct Inflight {
    read: Read,
    /// bytes landed so far: a read may come back short
    got: usize,
}

impl Uring {
    pub(crate) fn new(depth: u32) -> io::Result<Self> {
        Ok(Self {
            ring: IoUring::new(depth.max(1))?,
            table: Mutex::default(),
            completions: Mutex::new(()),
            registered: None,
            woken: AtomicBool::new(false),
        })
    }

    /// # Safety
    ///
    /// `len` bytes at `base` stay valid for writes until the ring is dropped.
    pub(crate) unsafe fn register(&mut self, base: *mut u8, len: usize) -> io::Result<()> {
        let iovecs: Vec<libc::iovec> = (0..len.div_ceil(PIECE))
            .map(|i| libc::iovec {
                iov_base: base.wrapping_add(i * PIECE).cast(),
                iov_len: PIECE.min(len - i * PIECE),
            })
            .collect();
        // SAFETY: the caller vouches for the memory
        unsafe { self.ring.submitter().register_buffers(&iovecs) }?;
        self.registered = Some((base as usize, len));
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

    pub(crate) fn submit(&self, read: Read) -> io::Result<()> {
        let mut table = self.table.lock().unwrap();
        let index = match table.free.pop() {
            Some(index) => index,
            None => {
                table.inflight.push(None);
                table.inflight.len() - 1
            }
        };
        table.inflight[index] = Some(Inflight { read, got: 0 });
        self.push(&table, index).inspect_err(|_| {
            table.inflight[index] = None;
            table.free.push(index);
        })
    }

    /// Queues what is left of the read at `index`. The caller holds `table`,
    /// and with it the submission queue.
    fn push(&self, table: &Table, index: usize) -> io::Result<()> {
        let Inflight { read, got } = table.inflight[index].as_ref().expect("in flight");
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
        .user_data(index as u64);
        loop {
            // SAFETY: `table` is locked, so no other submission queue exists;
            // `Mover::submit` requires the destination and the descriptor valid
            // until the read is reaped, and dropping waits for it
            if unsafe { self.ring.submission_shared().push(&entry) }.is_ok() {
                return Ok(());
            }
            self.ring.submit()?;
        }
    }

    pub(crate) fn flush(&self) -> io::Result<()> {
        self.ring.submit().map(drop)
    }

    /// Call from the one thread that reaps.
    pub(crate) fn try_done(&self) -> Option<Done> {
        let _completions = self.completions.lock().unwrap();
        loop {
            // SAFETY: `completions` is locked, so no other completion queue exists
            let completion = unsafe { self.ring.completion_shared() }.next()?;
            if completion.user_data() == WAKE {
                continue;
            }
            let index = completion.user_data() as usize;
            let mut table = self.table.lock().unwrap();
            let inflight = table.inflight[index].as_mut().expect("in flight");
            let again = match completion.result() {
                0 if inflight.read.len > 0 => Err(io::ErrorKind::UnexpectedEof.into()),
                n if n >= 0 => {
                    inflight.got += n as usize;
                    // short: the rest goes back in the ring
                    Ok(inflight.got < inflight.read.len)
                }
                errno if errno == -libc::EINTR || errno == -libc::EAGAIN => Ok(true),
                errno => Err(io::Error::from_raw_os_error(-errno)),
            };
            let result = match again {
                Ok(true) => match self.push(&table, index) {
                    Ok(()) => continue,
                    Err(err) => Err(err),
                },
                Ok(false) => Ok(()),
                Err(err) => Err(err),
            };
            let id = table.inflight[index].take().expect("in flight").read.id;
            table.free.push(index);
            return Some(Done { id, result });
        }
    }

    /// Call from the one thread that reaps.
    pub(crate) fn wait(&self, out: &mut Vec<Done>, max: usize, timeout: Option<Duration>) {
        let deadline = timeout.map(|timeout| Instant::now() + timeout);
        let before = out.len();
        loop {
            while out.len() - before < max {
                match self.try_done() {
                    Some(done) => out.push(done),
                    None => break,
                }
            }
            if out.len() > before || self.woken.swap(false, Ordering::AcqRel) {
                return;
            }
            let waited = match deadline {
                None => self.ring.submit_and_wait(1),
                Some(deadline) => {
                    let Some(left) = deadline.checked_duration_since(Instant::now()) else {
                        return;
                    };
                    let left = types::Timespec::from(left);
                    let args = types::SubmitArgs::new().timespec(&left);
                    self.ring.submitter().submit_with_args(1, &args)
                }
            };
            match waited {
                Ok(_) => {}
                // `ETIME`: the next turn sees the deadline; `EBUSY`: completions to reap first
                Err(err)
                    if matches!(
                        err.raw_os_error(),
                        Some(libc::EINTR | libc::EBUSY | libc::ETIME)
                    ) => {}
                // reads may be in the kernel: their destinations can't be handed back
                Err(err) => panic!("io_uring failed with reads in flight: {err}"),
            }
        }
    }

    /// A no-op through the ring ends the wait in the kernel.
    pub(crate) fn wake(&self) {
        self.woken.store(true, Ordering::Release);
        let entry = opcode::Nop::new().build().user_data(WAKE);
        let _table = self.table.lock().unwrap();
        // SAFETY: `table` is locked, so no other submission queue exists; a
        // no-op touches no memory. A full queue means a completion is on its
        // way, which ends the wait too.
        let _ = unsafe { self.ring.submission_shared().push(&entry) };
        let _ = self.ring.submit();
    }

    /// Call with a read pending, from the one thread that reaps.
    pub(crate) fn wait_done(&self) -> Done {
        loop {
            if let Some(done) = self.try_done() {
                return done;
            }
            match self.ring.submit_and_wait(1) {
                Ok(_) => {}
                // `EBUSY`: completions to reap first
                Err(err) if matches!(err.raw_os_error(), Some(libc::EINTR | libc::EBUSY)) => {}
                // reads may be in the kernel: their destinations can't be handed back
                Err(err) => panic!("io_uring failed with reads in flight: {err}"),
            }
        }
    }
}
