//! Reads of file ranges into memory the caller owns, each on its own: queued
//! on a [`Reader`], which lives across calls, and waited on by request.
//!
//! ```text
//! caller
//!   Reader::read(file, range, target)   one byte-mover read per piece: straight
//!                                       to host memory, or into a pinned slab
//!                                       for a device
//! reaper thread
//!   Mover::wait_done                    a piece landed
//!   CudaDevice::scatter                 device targets: an asynchronous copy
//!                                       out of the slab, which its fence keeps
//! caller
//!   Request::wait                       every piece landed, then their fences
//! ```

use std::{
    collections::HashMap,
    fs::File,
    ops::Range,
    os::fd::AsRawFd,
    sync::{Arc, Condvar, Mutex},
    thread::JoinHandle,
};

use byte_mover::{Mover, Read};
use safetensors::slice::StridedCopy;

use super::{
    cuda::api,
    cuda_memory::{pinned_pool, CudaDevice},
    memory::Scatter,
    pool::{Slab, SlabPool},
    transfer::{Error, Fence, Staged, CHUNK_SIZE},
};

/// Where a read's bytes go
#[derive(Clone, Copy, Debug)]
pub enum Target {
    /// host memory at this address
    Host(u64),
    /// memory of CUDA device `device` at `addr`. It may still be in use by
    /// work queued on `stream`: the copy waits for it.
    Cuda { device: i32, addr: u64, stream: u64 },
}

pub struct Reader {
    shared: Arc<Shared>,
    reaper: Option<JoinHandle<()>>,
}

struct Shared {
    mover: Mover,
    state: Mutex<State>,
    /// a piece was queued or landed, or the reader closed
    changed: Condvar,
    devices: Mutex<HashMap<i32, Arc<Device>>>,
}

struct Device {
    space: CudaDevice,
    pool: Arc<SlabPool>,
}

#[derive(Default)]
struct State {
    /// the pieces in the mover, by read id
    pieces: HashMap<u64, Piece>,
    requests: HashMap<u64, Progress>,
    next: u64,
    closed: bool,
}

struct Piece {
    request: u64,
    /// kept open until the piece landed
    _file: Arc<File>,
    staged: Option<StagedPiece>,
}

/// A piece on its way to a device
struct StagedPiece {
    device: Arc<Device>,
    slab: Slab,
    file: Range<u64>,
    addr: u64,
    ready: Option<Arc<dyn Fence>>,
}

struct Progress {
    /// pieces not landed yet
    left: usize,
    error: Option<String>,
    /// of the copies still in flight
    fences: Vec<Arc<dyn Fence>>,
    /// its [`Request`] was dropped: forgotten once the last piece lands
    dropped: bool,
}

/// One read of a [`Reader`]
pub struct Request {
    shared: Arc<Shared>,
    id: u64,
}

impl Reader {
    /// A reader whose reads run on `threads` threads.
    pub fn new(threads: usize) -> std::io::Result<Self> {
        let shared = Arc::new(Shared {
            mover: Mover::threads(threads)?,
            state: Mutex::default(),
            changed: Condvar::new(),
            devices: Mutex::default(),
        });
        let reaper = std::thread::Builder::new()
            .name("safetensors-reader".into())
            .spawn({
                let shared = shared.clone();
                move || shared.reap()
            })?;
        Ok(Self {
            shared,
            reaper: Some(reaper),
        })
    }

    /// Starts reading `range` of `file` to `target`.
    ///
    /// # Safety
    ///
    /// `target` is valid for writes of the range's length, and nothing else
    /// accesses those bytes, until the request is waited ([`Request::wait`]) or
    /// the reader is dropped.
    pub unsafe fn read(
        &self,
        file: &Arc<File>,
        range: Range<u64>,
        target: Target,
    ) -> Result<Request, Error> {
        let shared = &self.shared;
        // (offset in the file, length, destination of the read, staging)
        let mut reads = Vec::new();
        match target {
            Target::Host(addr) => {
                reads.push((range.start, (range.end - range.start) as usize, addr, None))
            }
            Target::Cuda {
                device,
                addr,
                stream,
            } => {
                let device = shared.device(device)?;
                let ready = device.space.queued(stream)?;
                let chunk = CHUNK_SIZE.get() as u64;
                let mut start = range.start;
                while start < range.end {
                    let end = (start + chunk).min(range.end);
                    let slab = device.pool.acquire()?;
                    let staged = StagedPiece {
                        device: device.clone(),
                        file: start..end,
                        addr: addr + (start - range.start),
                        ready: ready.clone(),
                        slab,
                    };
                    reads.push((start, (end - start) as usize, 0, Some(staged)));
                    start = end;
                }
            }
        }
        let mut state = shared.state.lock().unwrap();
        let id = state.next;
        state.next += reads.len().max(1) as u64;
        state.requests.insert(
            id,
            Progress {
                left: reads.len(),
                error: None,
                fences: Vec::new(),
                dropped: false,
            },
        );
        for (i, (offset, len, addr, mut staged)) in reads.into_iter().enumerate() {
            let dst = match &mut staged {
                Some(staged) => staged.slab.as_mut_ptr(),
                None => addr as *mut u8,
            };
            let read = Read {
                id: id + i as u64,
                fd: file.as_raw_fd(),
                offset,
                dst,
                len,
            };
            state.pieces.insert(
                read.id,
                Piece {
                    request: id,
                    _file: file.clone(),
                    staged,
                },
            );
            // SAFETY: the caller vouches for a host target; a slab is leased
            // until its piece landed; the piece keeps the file open
            if let Err(err) = unsafe { shared.mover.submit(read) } {
                state.pieces.remove(&read.id);
                let progress = state.requests.get_mut(&id).expect("just inserted");
                progress.left -= 1;
                progress.error.get_or_insert(err.to_string());
            }
        }
        drop(state);
        let _ = shared.mover.flush();
        shared.changed.notify_all();
        Ok(Request {
            shared: shared.clone(),
            id,
        })
    }
}

impl Drop for Reader {
    fn drop(&mut self) {
        self.shared.state.lock().unwrap().closed = true;
        self.shared.changed.notify_all();
        // returns once every piece landed
        if let Some(reaper) = self.reaper.take() {
            let _ = reaper.join();
        }
    }
}

impl Shared {
    fn device(&self, device: i32) -> Result<Arc<Device>, Error> {
        let mut devices = self.devices.lock().unwrap();
        if let Some(device) = devices.get(&device) {
            return Ok(device.clone());
        }
        let api = api().ok_or_else(|| Error::Other("no CUDA runtime is loaded".into()))?;
        let opened = Arc::new(Device {
            space: CudaDevice::new(api, device)?,
            pool: api.with_device(device, |_| pinned_pool(api))?,
        });
        devices.insert(device, opened.clone());
        Ok(opened)
    }

    /// The reaper thread: lands the pieces the mover finished, until the
    /// reader closed and none is left.
    fn reap(&self) {
        loop {
            {
                let mut state = self.state.lock().unwrap();
                while state.pieces.is_empty() && !state.closed {
                    state = self.changed.wait(state).unwrap();
                }
                if state.pieces.is_empty() {
                    return;
                }
            }
            match self.mover.wait_done() {
                Some(done) => self.land(done.id, done.result.map_err(Error::from)),
                // a piece is registered just before it is queued
                None => std::thread::yield_now(),
            }
        }
    }

    fn land(&self, id: u64, read: Result<(), Error>) {
        let piece = self.state.lock().unwrap().pieces.remove(&id);
        let piece = piece.expect("a piece the mover was given");
        let landed = read.and_then(|()| match piece.staged {
            None => Ok(None),
            Some(staged) => {
                if let Some(ready) = &staged.ready {
                    ready.wait()?;
                }
                let len = (staged.file.end - staged.file.start) as usize;
                let copy = StridedCopy {
                    src: 0,
                    dst: staged.addr as usize,
                    width: len,
                    height: 1,
                    src_pitch: len,
                    dst_pitch: len,
                };
                let bytes = Staged::new(staged.slab, 0, staged.file);
                staged.device.space.scatter(bytes, &[copy])
            }
        });
        let mut state = self.state.lock().unwrap();
        let progress = state.requests.get_mut(&piece.request).expect("its request");
        progress.left -= 1;
        match landed {
            Ok(fence) => progress.fences.extend(fence),
            Err(err) => {
                progress.error.get_or_insert(err.to_string());
            }
        }
        if progress.left == 0 && progress.dropped {
            state.requests.remove(&piece.request);
        }
        drop(state);
        self.changed.notify_all();
    }
}

impl Request {
    /// Returns once the bytes are at their target, or with what failed.
    pub fn wait(&self) -> Result<(), Error> {
        let mut state = self.shared.state.lock().unwrap();
        let fences = loop {
            let progress = &state.requests[&self.id];
            if progress.left == 0 {
                if let Some(err) = &progress.error {
                    return Err(Error::Other(err.clone().into()));
                }
                break progress.fences.clone();
            }
            state = self.shared.changed.wait(state).unwrap();
        };
        drop(state);
        fences.iter().try_for_each(|fence| fence.wait())
    }

    /// Whether [`Self::wait`] would return at once.
    pub fn done(&self) -> Result<bool, Error> {
        let state = self.shared.state.lock().unwrap();
        let progress = &state.requests[&self.id];
        if progress.left > 0 {
            return Ok(false);
        }
        if progress.error.is_some() {
            return Ok(true);
        }
        let fences = progress.fences.clone();
        drop(state);
        for fence in &fences {
            if !fence.done()? {
                return Ok(false);
            }
        }
        Ok(true)
    }
}

impl Drop for Request {
    fn drop(&mut self) {
        let mut state = self.shared.state.lock().unwrap();
        match state.requests.get_mut(&self.id) {
            Some(progress) if progress.left > 0 => progress.dropped = true,
            _ => {
                state.requests.remove(&self.id);
            }
        }
    }
}
