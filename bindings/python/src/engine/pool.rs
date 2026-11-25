use std::{
    collections::VecDeque,
    ops::{Deref, DerefMut},
    sync::{Condvar, Mutex, OnceLock},
};

use super::{
    file::DIRECT_ALIGN,
    transfer::{Error, CHUNK_SIZE},
};
use crate::engine::cuda::{CudaApi, CudaError, DeviceGuard, Event, Stream};

/// A chunk's superset aligned to [`DIRECT_ALIGN`] fits in a slab
pub(super) const SLAB_SIZE: usize = CHUNK_SIZE.get() + 2 * DIRECT_ALIGN;
const _: () = assert!(SLAB_SIZE.is_multiple_of(DIRECT_ALIGN));

#[derive(Clone, Copy)]
struct Slot {
    offset: usize,
    /// (completion event of the last host2device copy, device id)
    last_copy: Option<(Event, i32)>,
}

/// A slab leased from a [`SlabPool`]; returns to it on drop.
pub struct Slab {
    pool: &'static SlabPool,
    slot: Slot,
}

impl Slab {
    fn ptr(&self) -> *mut u8 {
        // SAFETY: `offset` is a slab's start in the pool's buffer
        unsafe { self.pool.buffer_ptr.add(self.slot.offset) }
    }

    /// Records on `stream` the copies reading the slab: [`SlabPool::acquire`]
    /// waits for them before leasing it again.
    pub(super) fn release(mut self, d: &DeviceGuard<'_>, stream: Stream) -> Result<(), CudaError> {
        let event = match self.slot.last_copy.take() {
            Some((event, device)) if device == d.id() => event,
            Some((event, _)) => {
                let _ = d.api().event_destroy(event);
                d.event_create()?
            }
            None => d.event_create()?,
        };
        self.slot.last_copy = Some((event, d.id()));
        d.api().event_record(event, stream)
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
        self.pool.put_back(self.slot);
    }
}

pub struct SlabPool {
    cuda: &'static CudaApi,
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
                last_copy: None,
            })
            .collect();
        Ok(Self {
            cuda,
            buffer_ptr,
            slab_size,
            aligned: buffer_ptr.addr().is_multiple_of(DIRECT_ALIGN)
                && slab_size.is_multiple_of(DIRECT_ALIGN),
            free: Mutex::new(free),
            available: Condvar::new(),
        })
    }

    /// Leases a slab, waiting for one to be free and for the copies that read it.
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
        let slab = Slab { pool: self, slot };
        if let Some((event, _)) = slab.slot.last_copy {
            self.cuda.event_sync(event)?;
        }
        Ok(slab)
    }

    fn put_back(&self, slot: Slot) {
        self.free.lock().unwrap().push_back(slot);
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
