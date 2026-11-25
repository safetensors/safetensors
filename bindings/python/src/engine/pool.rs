use std::{
    collections::VecDeque,
    ops::{Deref, DerefMut},
    sync::{Arc, Condvar, Mutex, OnceLock},
};

use super::{
    file::DIRECT_ALIGN,
    transfer::{Error, Fence, CHUNK_SIZE},
};
use crate::engine::cuda::{CudaApi, CudaError};

/// A chunk's superset aligned to [`DIRECT_ALIGN`] fits in a slab
pub(super) const SLAB_SIZE: usize = CHUNK_SIZE.get() + 2 * DIRECT_ALIGN;
const _: () = assert!(SLAB_SIZE.is_multiple_of(DIRECT_ALIGN));

struct Slot {
    offset: usize,
    /// what may still read the slab
    pending: Option<Arc<dyn Fence>>,
}

/// A slab leased from a [`SlabPool`]; returns to it on drop.
pub struct Slab {
    pool: &'static SlabPool,
    offset: usize,
    pending: Option<Arc<dyn Fence>>,
}

impl Slab {
    fn ptr(&self) -> *mut u8 {
        // SAFETY: `offset` is a slab's start in the pool's buffer
        unsafe { self.pool.buffer_ptr.add(self.offset) }
    }

    /// Returns the slab to its pool, to be leased again once `fence` completed.
    pub fn release_after(mut self, fence: Arc<dyn Fence>) {
        self.pending = Some(fence);
    }
}

impl Deref for Slab {
    type Target = [u8];

    fn deref(&self) -> &[u8] {
        // SAFETY: the lease holds the slab, `slab_size` bytes of the pool's buffer
        unsafe { std::slice::from_raw_parts(self.ptr(), self.pool.slab_size) }
    }
}

impl DerefMut for Slab {
    fn deref_mut(&mut self) -> &mut [u8] {
        // SAFETY: the lease holds the slab, `slab_size` bytes of the pool's buffer
        unsafe { std::slice::from_raw_parts_mut(self.ptr(), self.pool.slab_size) }
    }
}

impl Drop for Slab {
    fn drop(&mut self) {
        let slot = Slot {
            offset: self.offset,
            pending: self.pending.take(),
        };
        self.pool.free.lock().unwrap().push_back(slot);
        self.pool.available.notify_one();
    }
}

pub struct SlabPool {
    buffer_ptr: *mut u8,
    slab_size: usize,
    /// whether every slab is aligned to [`DIRECT_ALIGN`]
    pub(super) aligned: bool,
    free: Mutex<VecDeque<Slot>>,
    available: Condvar,
}

unsafe impl Send for SlabPool {}
unsafe impl Sync for SlabPool {}

impl SlabPool {
    fn new(cuda: &'static CudaApi, n_slabs: usize, slab_size: usize) -> Result<Self, CudaError> {
        let buffer_ptr = cuda.host_alloc(n_slabs * slab_size)?;
        let free = (0..n_slabs)
            .map(|i| Slot {
                offset: i * slab_size,
                pending: None,
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

    /// Leases a slab, waiting for one to be free and for what still reads it.
    pub fn acquire(&'static self) -> Result<Slab, Error> {
        // TODO: handle poisoning
        let mut free = self.free.lock().unwrap();
        let slot = loop {
            if let Some(s) = free.pop_front() {
                break s;
            }
            free = self.available.wait(free).unwrap();
        };
        drop(free);
        let mut slab = Slab {
            pool: self,
            offset: slot.offset,
            pending: slot.pending,
        };
        if let Some(pending) = &slab.pending {
            pending.wait()?;
        }
        slab.pending = None;
        Ok(slab)
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
