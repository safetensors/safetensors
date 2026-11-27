//! Slabs: buffers a worker stages a chunk in, leased from a [`SlabPool`] cut
//! from one [`Backing`] allocation.

use std::{
    collections::VecDeque,
    ops::{Deref, DerefMut},
    sync::{Arc, Condvar, Mutex},
};

use super::{
    file::DIRECT_ALIGN,
    transfer::{Error, Fence, CHUNK_SIZE},
};

/// A chunk's superset aligned to [`DIRECT_ALIGN`] fits in a slab
pub const SLAB_SIZE: usize = CHUNK_SIZE.get() + 2 * DIRECT_ALIGN;
const _: () = assert!(SLAB_SIZE.is_multiple_of(DIRECT_ALIGN));

/// The memory a [`SlabPool`] cuts its slabs from.
///
/// # Safety
///
/// [`Self::ptr`] points to [`Self::size`] bytes, valid for reads and writes
/// for as long as the backing lives, and used by nothing else.
pub unsafe trait Backing: Send + Sync {
    fn ptr(&self) -> *mut u8;
    fn size(&self) -> usize;
}

/// Zeroed host memory, aligned to [`DIRECT_ALIGN`]
#[cfg(any(test, target_os = "linux"))]
pub struct HostMemory {
    _bytes: Vec<u8>,
    /// of the aligned start in `_bytes`, taken with `as_mut_ptr`
    addr: usize,
    size: usize,
}

#[cfg(any(test, target_os = "linux"))]
impl HostMemory {
    pub fn new(size: usize) -> Self {
        let mut bytes = vec![0; size + DIRECT_ALIGN];
        let ptr = bytes.as_mut_ptr();
        let addr = ptr.wrapping_add(ptr.align_offset(DIRECT_ALIGN)) as usize;
        Self {
            _bytes: bytes,
            addr,
            size,
        }
    }
}

// SAFETY: `size` bytes at `addr` in `_bytes`, which is neither resized nor
// accessed otherwise until dropped
#[cfg(any(test, target_os = "linux"))]
unsafe impl Backing for HostMemory {
    fn ptr(&self) -> *mut u8 {
        self.addr as *mut u8
    }

    fn size(&self) -> usize {
        self.size
    }
}

struct Slot {
    offset: usize,
    /// what may still read the slab
    pending: Option<Arc<dyn Fence>>,
}

/// A slab leased from a [`SlabPool`]; returns to it on drop.
pub struct Slab {
    pool: Arc<SlabPool>,
    offset: usize,
    pending: Option<Arc<dyn Fence>>,
}

impl Slab {
    /// Returns the slab to its pool, to be leased again once `fence` completed.
    pub fn release_after(mut self, fence: Arc<dyn Fence>) {
        self.pending = Some(fence);
    }
}

impl Deref for Slab {
    type Target = [u8];

    fn deref(&self) -> &[u8] {
        // SAFETY: the lease holds `slab_size` bytes at `offset` of the backing
        unsafe {
            std::slice::from_raw_parts(
                self.pool.backing.ptr().add(self.offset),
                self.pool.slab_size,
            )
        }
    }
}

impl DerefMut for Slab {
    fn deref_mut(&mut self) -> &mut [u8] {
        // SAFETY: the lease holds `slab_size` bytes at `offset` of the backing
        unsafe {
            std::slice::from_raw_parts_mut(
                self.pool.backing.ptr().add(self.offset),
                self.pool.slab_size,
            )
        }
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
    backing: Box<dyn Backing>,
    slab_size: usize,
    /// whether every slab is aligned to [`DIRECT_ALIGN`]
    aligned: bool,
    free: Mutex<VecDeque<Slot>>,
    available: Condvar,
}

impl SlabPool {
    /// A pool of the slabs of `slab_size` bytes that fit in `backing`
    pub fn new(backing: Box<dyn Backing>, slab_size: usize) -> Arc<Self> {
        let free = (0..backing.size() / slab_size)
            .map(|i| Slot {
                offset: i * slab_size,
                pending: None,
            })
            .collect();
        Arc::new(Self {
            aligned: backing.ptr().addr().is_multiple_of(DIRECT_ALIGN)
                && slab_size.is_multiple_of(DIRECT_ALIGN),
            backing,
            slab_size,
            free: Mutex::new(free),
            available: Condvar::new(),
        })
    }

    /// Whether every slab is aligned to [`DIRECT_ALIGN`]
    pub fn aligned(&self) -> bool {
        self.aligned
    }

    /// Leases a slab, waiting for one to be free and for what still reads it.
    pub fn acquire(self: &Arc<Self>) -> Result<Slab, Error> {
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
            pool: self.clone(),
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

#[cfg(test)]
pub(crate) mod tests {
    use std::sync::{
        atomic::{AtomicBool, Ordering},
        Arc,
    };

    use super::{HostMemory, SlabPool};
    use crate::engine::transfer::{Error, Fence};

    /// `slabs` slabs of `slab_size` bytes in host memory
    pub(crate) fn heap_pool(slabs: usize, slab_size: usize) -> Arc<SlabPool> {
        SlabPool::new(Box::new(HostMemory::new(slabs * slab_size)), slab_size)
    }

    struct Flag(AtomicBool);

    impl Fence for Flag {
        fn wait(&self) -> Result<(), Error> {
            self.0.store(true, Ordering::Relaxed);
            Ok(())
        }
    }

    #[test]
    fn test_slab_is_leased_again_once_its_fence_completed() {
        let pool = heap_pool(1, 64);
        let fence = Arc::new(Flag(AtomicBool::new(false)));
        let mut slab = pool.acquire().unwrap();
        slab[..3].copy_from_slice(b"abc");
        slab.release_after(fence.clone());
        assert!(!fence.0.load(Ordering::Relaxed));
        let slab = pool.acquire().unwrap();
        assert!(fence.0.load(Ordering::Relaxed));
        assert_eq!(&slab[..3], b"abc");
    }
}
