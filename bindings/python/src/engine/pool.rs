use std::{
    collections::VecDeque,
    sync::{Condvar, Mutex, OnceLock},
};

use super::{file::DIRECT_ALIGN, unix::CHUNK_SIZE};
use crate::engine::cuda::{CudaApi, CudaError, DeviceGuard, Event, Stream};

/// A chunk's superset aligned to [`DIRECT_ALIGN`] fits in a slab
pub(super) const SLAB_SIZE: usize = CHUNK_SIZE.get() + 2 * DIRECT_ALIGN;
const _: () = assert!(SLAB_SIZE.is_multiple_of(DIRECT_ALIGN));

#[derive(Clone, Copy)]
struct Slab {
    offset: usize,
    /// (completion event of the last host2device copy, device id)
    last_copy: Option<(Event, i32)>,
}

pub(super) struct SlabLease {
    pool: &'static SlabPool,
    slab: Slab,
}

impl SlabLease {
    pub(super) fn ptr(&self) -> *mut u8 {
        unsafe { self.pool.buffer_ptr.add(self.slab.offset) }
    }

    pub(super) fn slab(&mut self) -> &mut [u8] {
        // SAFETY: the lease holds the slab, `slab_size` bytes of the pool's buffer
        unsafe { std::slice::from_raw_parts_mut(self.ptr(), self.pool.slab_size) }
    }

    pub(super) fn release(mut self, d: &DeviceGuard<'_>, stream: Stream) -> Result<(), CudaError> {
        let event = match self.slab.last_copy.take() {
            Some((event, device)) if device == d.id() => event,
            Some((event, _)) => {
                let _ = d.api().event_destroy(event);
                d.event_create()?
            }
            None => d.event_create()?,
        };
        self.slab.last_copy = Some((event, d.id()));
        d.api().event_record(event, stream)
    }
}

impl Drop for SlabLease {
    fn drop(&mut self) {
        self.pool.put_back(self.slab);
    }
}

pub(super) struct SlabPool {
    buffer_ptr: *mut u8,
    slab_size: usize,
    /// whether every slab is aligned to [`DIRECT_ALIGN`]
    pub(super) aligned: bool,
    free: Mutex<VecDeque<Slab>>,
    available: Condvar,
}

unsafe impl Send for SlabPool {}
unsafe impl Sync for SlabPool {}

impl SlabPool {
    fn new(cuda: &CudaApi, n_slabs: usize, slab_size: usize) -> Result<Self, CudaError> {
        let buffer_ptr = cuda.host_alloc(n_slabs * slab_size)?;
        let free = (0..n_slabs)
            .map(|i| Slab {
                offset: i * slab_size,
                last_copy: None,
            })
            .collect();
        Ok(Self {
            buffer_ptr,
            slab_size,
            aligned: buffer_ptr.addr().is_multiple_of(DIRECT_ALIGN)
                && slab_size.is_multiple_of(DIRECT_ALIGN),
            free: Mutex::new(free),
            available: Condvar::new(),
        })
    }

    pub(super) fn acquire(&'static self, cuda: &CudaApi) -> Result<SlabLease, CudaError> {
        // TODO: handle poisoning
        let mut free = self.free.lock().unwrap();
        let slab = loop {
            if let Some(s) = free.pop_front() {
                break s;
            }
            free = self.available.wait(free).unwrap();
        };
        drop(free);
        let lease = SlabLease { pool: self, slab };
        if let Some((event, _)) = lease.slab.last_copy {
            cuda.event_sync(event)?;
        }
        Ok(lease)
    }

    fn put_back(&self, slab: Slab) {
        self.free.lock().unwrap().push_back(slab);
        self.available.notify_one();
    }
}

pub(super) fn pool(cuda_api: &'static CudaApi) -> Result<&'static SlabPool, CudaError> {
    static POOL: OnceLock<SlabPool> = OnceLock::new();
    static INIT: Mutex<()> = Mutex::new(());
    if POOL.get().is_none() {
        let _g = INIT.lock().unwrap();
        if POOL.get().is_none() {
            let _ = POOL.set(SlabPool::new(cuda_api, 32, SLAB_SIZE)?);
        }
    }
    Ok(POOL.get().unwrap())
}
