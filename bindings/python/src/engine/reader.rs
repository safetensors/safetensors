//! Reads of file ranges into memory the caller owns, each on its own: queued
//! on a [`Reader`], which lives across calls, and waited on by request.
//!
//! ```text
//! caller
//!   Reader::read(reads)                 one byte-mover read per piece: straight
//!                                       to host memory, or staged in a slab
//!                                       (pinned for a device), where a direct
//!                                       read lands its aligned superset
//! reaper thread
//!   Mover::wait                         pieces landed
//!   Scatter::scatter                    staged pieces: a copy out of the slab
//!                                       (asynchronous to a device, whose fence
//!                                       keeps the slab)
//! caller
//!   Request::wait                       every piece landed, then their fences
//! ```

use std::{
    collections::HashMap,
    fs::File,
    ops::Range,
    os::fd::AsRawFd,
    sync::{Arc, Condvar, Mutex, OnceLock},
    thread::JoinHandle,
};

use byte_mover::{Mover, Read};
use safetensors::slice::StridedCopy;

use super::{
    cuda::api,
    cuda_memory::{pinned_pool, CudaDevice},
    file::{reopen_direct, DIRECT_ALIGN},
    memory::Scatter,
    pool::{Slab, SlabPool},
    transfer::{Error, Fence, Staged, CHUNK_SIZE},
};
#[cfg(any(target_os = "linux", all(target_os = "macos", target_arch = "aarch64")))]
use super::{
    memory::Host,
    pool::{HostMemory, SLAB_SIZE},
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
    /// a piece landed
    landed: Condvar,
    devices: Mutex<HashMap<i32, Arc<Device>>>,
    /// whether reads skip the page cache where the filesystem allows it
    direct: bool,
    /// per file, by the address of its `File`: it reopened for direct reads with
    /// its size, `None` when it can't be
    reopened: Mutex<HashMap<usize, Reopened>>,
    /// slabs for direct reads to host memory, allocated at the first one
    host_slabs: OnceLock<Option<Arc<SlabPool>>>,
    /// reads at once: the size of the host slab pool
    depth: usize,
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
    file: Arc<File>,
    /// its direct twin, when the piece reads through it
    direct: Option<Arc<File>>,
    staged: Option<StagedPiece>,
}

/// A file reopened for direct reads, with its size
type Reopened = Option<(Arc<File>, u64)>;

/// What a copy to a device waits for
type Ready = Option<Arc<dyn Fence>>;

/// A piece read into a slab, copied to its target once landed
struct StagedPiece {
    /// the device of the target, `None` for host memory
    device: Option<Arc<Device>>,
    slab: Slab,
    file: Range<u64>,
    /// where the piece starts in the slab: past the head of a direct read
    head: usize,
    addr: u64,
    ready: Ready,
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

/// How a range is staged: in slabs of `slabs`, then copied to `addr`
struct Staging {
    /// the device of the target, `None` for host memory
    device: Option<Arc<Device>>,
    slabs: Arc<SlabPool>,
    addr: u64,
    ready: Ready,
}

impl State {
    /// Counts `pieces` of request `id` as failed with `error`.
    fn fail(&mut self, id: u64, pieces: usize, error: String) {
        let progress = self.requests.get_mut(&id).expect("its request");
        progress.left -= pieces;
        progress.error.get_or_insert(error);
        if progress.left == 0 && progress.dropped {
            self.requests.remove(&id);
        }
    }
}

/// One read of a [`Reader`]
pub struct Request {
    shared: Arc<Shared>,
    id: u64,
}

impl Reader {
    /// A reader that runs `threads` reads at once. With `direct`, reads skip
    /// the page cache on the filesystems that allow it.
    pub fn new(threads: usize, direct: bool) -> std::io::Result<Self> {
        let depth = threads.max(1);
        let shared = Arc::new(Shared {
            mover: Mover::threads(depth)?,
            state: Mutex::default(),
            landed: Condvar::new(),
            devices: Mutex::default(),
            direct,
            reopened: Mutex::default(),
            host_slabs: OnceLock::new(),
            depth,
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

    /// Starts reading each range of a file to its target, as one request.
    ///
    /// # Safety
    ///
    /// Each target is valid for writes of its range's length, and nothing else
    /// accesses those bytes, until the request is waited ([`Request::wait`]) or
    /// the reader is dropped.
    pub unsafe fn read(
        &self,
        ranges: Vec<(Arc<File>, Range<u64>, Target)>,
    ) -> Result<Request, Error> {
        let shared = &self.shared;
        let chunk = CHUNK_SIZE.get() as u64;
        // where each range is staged, if it is, and the work queued on a
        // device's stream, looked up once per request
        let mut queued: HashMap<(i32, u64), Ready> = HashMap::new();
        let mut routes = Vec::with_capacity(ranges.len());
        let mut pieces = 0;
        for (file, range, target) in &ranges {
            let direct = shared.reopen(file);
            let route = match *target {
                Target::Host(addr) => match direct.as_ref().and_then(|_| shared.host_slabs()) {
                    Some(slabs) => Ok(Staging {
                        device: None,
                        slabs,
                        addr,
                        ready: None,
                    }),
                    // through the page cache, straight to the target
                    None => Err(addr),
                },
                Target::Cuda {
                    device: index,
                    addr,
                    stream,
                } => {
                    let device = shared.device(index)?;
                    let ready = match queued.get(&(index, stream)) {
                        Some(ready) => ready.clone(),
                        None => {
                            let ready = device.space.queued(stream)?;
                            queued.insert((index, stream), ready.clone());
                            ready
                        }
                    };
                    Ok(Staging {
                        slabs: device.pool.clone(),
                        device: Some(device),
                        addr,
                        ready,
                    })
                }
            };
            pieces += match route {
                Ok(_) => (range.end - range.start).div_ceil(chunk) as usize,
                Err(_) => 1,
            };
            routes.push((direct, route));
        }

        let id = {
            let mut state = shared.state.lock().unwrap();
            let id = state.next;
            state.next += pieces.max(1) as u64;
            state.requests.insert(
                id,
                Progress {
                    left: pieces,
                    error: None,
                    fences: Vec::new(),
                    dropped: false,
                },
            );
            id
        };
        let request = Request {
            shared: shared.clone(),
            id,
        };
        // Each piece is queued as soon as it has its slab: a request of more
        // pieces than there are slabs gets them back as its first ones land.
        let mut next = id;
        let queue = |read: Read, piece: Piece| {
            let mut state = shared.state.lock().unwrap();
            state.pieces.insert(read.id, piece);
            // SAFETY: the caller vouches for a host target; a slab is leased
            // until its piece landed; the piece keeps its files open
            unsafe { shared.mover.submit(read) };
        };
        for ((file, range, _), (direct, route)) in ranges.into_iter().zip(routes) {
            let staging = match route {
                Ok(staging) => staging,
                Err(addr) => {
                    let read = Read {
                        id: next,
                        fd: file.as_raw_fd(),
                        offset: range.start,
                        dst: addr as *mut u8,
                        len: (range.end - range.start) as usize,
                    };
                    let piece = Piece {
                        request: id,
                        file,
                        direct: None,
                        staged: None,
                    };
                    queue(read, piece);
                    next += 1;
                    continue;
                }
            };
            let mut start = range.start;
            while start < range.end {
                let end = (start + chunk).min(range.end);
                let len = (end - start) as usize;
                let mut slab = match staging.slabs.acquire() {
                    Ok(slab) => slab,
                    Err(err) => {
                        // what is left of the request is not read
                        let left = (id + pieces as u64 - next) as usize;
                        shared.state.lock().unwrap().fail(id, left, err.to_string());
                        shared.landed.notify_all();
                        return Ok(request);
                    }
                };
                // a direct read takes the aligned superset of the piece, unless
                // it would run past the end of the file
                let head = (start % DIRECT_ALIGN as u64) as usize;
                let aligned = (head + len).next_multiple_of(DIRECT_ALIGN);
                let direct = direct
                    .as_ref()
                    .filter(|(_, size)| start - head as u64 + aligned as u64 <= *size)
                    .map(|(direct, _)| direct.clone());
                let (head, offset, len) = match direct {
                    Some(_) => (head, start - head as u64, aligned),
                    None => (0, start, len),
                };
                let read = Read {
                    id: next,
                    fd: direct.as_ref().unwrap_or(&file).as_raw_fd(),
                    offset,
                    dst: slab.as_mut_ptr(),
                    len,
                };
                let staged = StagedPiece {
                    device: staging.device.clone(),
                    slab,
                    file: start..end,
                    head,
                    addr: staging.addr + (start - range.start),
                    ready: staging.ready.clone(),
                };
                let piece = Piece {
                    request: id,
                    file: file.clone(),
                    direct,
                    staged: Some(staged),
                };
                queue(read, piece);
                next += 1;
                start = end;
            }
        }
        Ok(request)
    }
}

impl Drop for Reader {
    fn drop(&mut self) {
        self.shared.state.lock().unwrap().closed = true;
        self.shared.mover.wake();
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

    /// `file` reopened for direct reads, and its size, if this reader reads
    /// directly and the filesystem takes it.
    fn reopen(&self, file: &Arc<File>) -> Reopened {
        if !self.direct {
            return None;
        }
        let mut reopened = self.reopened.lock().unwrap();
        reopened
            .entry(Arc::as_ptr(file) as usize)
            .or_insert_with(|| {
                let size = file.metadata().ok()?.len();
                Some((Arc::new(reopen_direct(file)?), size))
            })
            .clone()
    }

    /// The slabs direct reads to host memory are staged in
    fn host_slabs(&self) -> Option<Arc<SlabPool>> {
        self.host_slabs
            .get_or_init(|| {
                #[cfg(any(target_os = "linux", all(target_os = "macos", target_arch = "aarch64")))]
                {
                    let backing = HostMemory::new(self.depth * SLAB_SIZE);
                    Some(SlabPool::new(Box::new(backing), SLAB_SIZE))
                }
                #[cfg(not(any(
                    target_os = "linux",
                    all(target_os = "macos", target_arch = "aarch64")
                )))]
                None
            })
            .clone()
    }

    /// The reaper thread: lands the pieces the mover finished, until the
    /// reader closed and none is left.
    fn reap(&self) {
        let mut done = Vec::new();
        loop {
            // until a piece landed, or the reader woke it to close
            self.mover.wait(&mut done, 64, None);
            for done in done.drain(..) {
                self.land(done.id, done.result);
            }
            let state = self.state.lock().unwrap();
            if state.closed && state.pieces.is_empty() {
                return;
            }
        }
    }

    fn land(&self, id: u64, read: std::io::Result<()>) {
        let piece = self.state.lock().unwrap().pieces.remove(&id);
        let mut piece = piece.expect("a piece the mover was given");
        // a filesystem that refuses a direct read after all: through the page cache
        if let (Err(err), Some(_), Some(staged)) = (&read, &piece.direct, &mut piece.staged) {
            if err.raw_os_error() == Some(libc::EINVAL) {
                let again = Read {
                    id,
                    fd: piece.file.as_raw_fd(),
                    offset: staged.file.start,
                    dst: staged.slab.as_mut_ptr(),
                    len: (staged.file.end - staged.file.start) as usize,
                };
                staged.head = 0;
                piece.direct = None;
                let mut state = self.state.lock().unwrap();
                state.pieces.insert(id, piece);
                // SAFETY: as when the piece was first queued
                unsafe { self.mover.submit(again) };
                return;
            }
        }
        let landed = read.map_err(Error::from).and_then(|()| match piece.staged {
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
                let bytes = Staged::new(staged.slab, staged.head, staged.file);
                match staged.device {
                    Some(device) => device.space.scatter(bytes, &[copy]),
                    #[cfg(any(
                        target_os = "linux",
                        all(target_os = "macos", target_arch = "aarch64")
                    ))]
                    None => Host.scatter(bytes, &[copy]),
                    #[cfg(not(any(
                        target_os = "linux",
                        all(target_os = "macos", target_arch = "aarch64")
                    )))]
                    None => unreachable!("no host slabs on this platform"),
                }
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
        self.landed.notify_all();
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
            state = self.shared.landed.wait(state).unwrap();
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
