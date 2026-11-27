//! CUDA for the transfer engine: a device's memory to [`Scatter`] into, pinned host
//! memory as a pool [`Backing`], events as [`Fence`]s.

use std::sync::{
    atomic::{AtomicU64, Ordering},
    Arc, Mutex, OnceLock,
};

use super::{
    cuda::{api, CudaApi, CudaError, DeviceGuard, Event, Stream},
    memory::{Gather, Scatter},
    pool::{Backing, SlabPool, SLAB_SIZE},
    transfer::{Error, Fence, Staged},
};
use safetensors::slice::StridedCopy;

/// Whether a CUDA runtime is loaded in this process
pub fn available() -> bool {
    api().is_some()
}

/// This is a generous limit on the maximum number of devices that could be connected to a single host
pub const MAX_DEVICES: usize = 64;
static STREAMS: [AtomicU64; MAX_DEVICES] = [const { AtomicU64::new(0) }; MAX_DEVICES];

fn get_stream(stream_slots: &[AtomicU64], api: &CudaApi, device: i32) -> Result<Stream, CudaError> {
    let slot = stream_slots
        .get(device as usize)
        .unwrap_or_else(|| panic!("device index {device} exceeds MAX_DEVICES ({MAX_DEVICES})"));
    let stream = slot.load(Ordering::Acquire);
    if stream != 0 {
        return Ok(stream as Stream);
    }

    let new = api.with_device(device, |d| d.stream_create())?;
    match slot.compare_exchange(0, new as u64, Ordering::AcqRel, Ordering::Acquire) {
        Ok(_) => Ok(new),
        Err(existing) => {
            let _ = api.stream_destroy(new);
            Ok(existing as Stream)
        }
    }
}

fn device_stream(api: &CudaApi, device: i32) -> Result<Stream, CudaError> {
    get_stream(&STREAMS, api, device)
}

/// An event recorded on a stream: the work queued on it before
pub struct CudaEvent {
    api: &'static CudaApi,
    event: Event,
}

// SAFETY: an event handle is used from any thread through the runtime API
unsafe impl Send for CudaEvent {}
// SAFETY: as for `Send`: the runtime API is thread safe
unsafe impl Sync for CudaEvent {}

impl CudaEvent {
    /// Records an event on `stream` of `device`.
    pub fn record(api: &'static CudaApi, device: i32, stream: Stream) -> Result<Self, CudaError> {
        api.with_device(device, |d| Self::record_on(api, d, stream))
    }

    fn record_on(
        api: &'static CudaApi,
        d: &DeviceGuard<'_>,
        stream: Stream,
    ) -> Result<Self, CudaError> {
        let event = d.event_create()?;
        if let Err(err) = api.event_record(event, stream) {
            let _ = api.event_destroy(event);
            return Err(err);
        }
        Ok(Self { api, event })
    }
}

impl Fence for CudaEvent {
    fn wait(&self) -> Result<(), Error> {
        Ok(self.api.event_sync(self.event)?)
    }
}

impl Drop for CudaEvent {
    fn drop(&mut self) {
        let _ = self.api.event_destroy(self.event);
    }
}

/// A CUDA device's memory, copied to on the device's stream
pub struct CudaDevice {
    api: &'static CudaApi,
    device: i32,
    stream: Stream,
}

// SAFETY: the stream handle is used from any thread through the runtime API
unsafe impl Send for CudaDevice {}
// SAFETY: as for `Send`: the runtime API is thread safe
unsafe impl Sync for CudaDevice {}

impl CudaDevice {
    pub fn new(api: &'static CudaApi, device: i32) -> Result<Self, CudaError> {
        Ok(Self {
            api,
            device,
            stream: device_stream(api, device)?,
        })
    }
}

impl Scatter for CudaDevice {
    fn scatter(
        &self,
        staged: Staged,
        copies: &[StridedCopy],
    ) -> Result<Option<Arc<dyn Fence>>, Error> {
        let src = staged.bytes().as_ptr();
        let api = self.api;
        let fence = api.with_device(self.device, |d| {
            let issued = copies.iter().try_for_each(|c| {
                let src = src.wrapping_add(c.src);
                match c.height {
                    1 => api.memcpy_h2d_async(c.dst as u64, src, c.width, self.stream),
                    _ => api.memcpy2d_h2d_async(
                        c.dst as u64,
                        c.dst_pitch,
                        src,
                        c.src_pitch,
                        c.width,
                        c.height,
                        self.stream,
                    ),
                }
            });
            let fence = issued.and_then(|()| CudaEvent::record_on(api, d, self.stream));
            if fence.is_err() {
                // the slab returns to the pool with `staged`: nothing may read it
                let _ = api.stream_sync(self.stream);
            }
            fence
        })?;
        let fence: Arc<dyn Fence> = Arc::new(fence);
        staged.into_slab().release_after(fence.clone());
        Ok(Some(fence))
    }

    fn allocation_end(&self, addr: u64) -> Option<u64> {
        self.api.allocation_end(addr)
    }
}

impl Gather for CudaDevice {
    fn gather(&self, dst: &mut [u8], copies: &[StridedCopy]) -> Result<(), Error> {
        let api = self.api;
        let done = api.with_device(self.device, |d| {
            let issued = copies.iter().try_for_each(|c| {
                let to = dst[c.dst..c.dst + c.width].as_mut_ptr();
                api.memcpy_d2h_async(to, c.src as u64, c.width, self.stream)
            });
            let done = issued.and_then(|()| CudaEvent::record_on(api, d, self.stream));
            if done.is_err() {
                // `dst` is the caller's again on return: nothing may write it
                let _ = api.stream_sync(self.stream);
            }
            done
        })?;
        done.wait()
    }
}

/// Page-locked host memory, which devices copy from and to asynchronously
struct Pinned {
    api: &'static CudaApi,
    ptr: *mut u8,
    size: usize,
}

// SAFETY: `ptr` is an allocation owned by `Pinned`, whose methods only read it
unsafe impl Send for Pinned {}
// SAFETY: as for `Send`
unsafe impl Sync for Pinned {}

// SAFETY: `cudaHostAlloc` returned `size` bytes at `ptr`, freed on drop only
unsafe impl Backing for Pinned {
    fn ptr(&self) -> *mut u8 {
        self.ptr
    }

    fn size(&self) -> usize {
        self.size
    }
}

impl Drop for Pinned {
    fn drop(&mut self) {
        let _ = self.api.free_host(self.ptr);
    }
}

/// The pool of pinned slabs every CUDA transfer in the process shares
pub fn pinned_pool(api: &'static CudaApi) -> Result<Arc<SlabPool>, CudaError> {
    static POOL: OnceLock<Arc<SlabPool>> = OnceLock::new();
    static INIT: Mutex<()> = Mutex::new(());
    if POOL.get().is_none() {
        let _g = INIT.lock().unwrap();
        if POOL.get().is_none() {
            let size = 32 * SLAB_SIZE;
            let ptr = api.host_alloc(size)?;
            let _ = POOL.set(SlabPool::new(
                Box::new(Pinned { api, ptr, size }),
                SLAB_SIZE,
            ));
        }
    }
    Ok(POOL.get().unwrap().clone())
}
