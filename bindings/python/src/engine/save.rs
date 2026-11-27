//! `save_file`: a safetensors file written from tensors in host memory or on
//! CUDA devices.

use std::{ops::Range, path::Path};

use super::{
    cuda::api,
    cuda_memory::{pinned_pool, CudaDevice, MAX_DEVICES},
    file::write_file,
    memory::{Host, LoadPlan, Memory, Span},
    pool::{HostMemory, Slab, SlabPool, SLAB_SIZE},
    transfer::{Error, Source, Staged, CHUNK_SIZE},
};

/// The memory a tensor's address is in
#[derive(Clone, Copy, PartialEq, Eq)]
pub enum Location {
    Host,
    Cuda(i32),
}

/// The file's bytes from host memory and from CUDA devices, each holding some
/// of its spans
struct Memories {
    host: Memory<Host>,
    devices: Vec<Memory<CudaDevice>>,
}

impl Source for Memories {
    fn fill(&self, file: Range<u64>, mut slab: Slab) -> Result<Staged, Error> {
        let dst = &mut slab[..(file.end - file.start) as usize];
        self.host.gather_into(&file, dst)?;
        for device in &self.devices {
            device.gather_into(&file, dst)?;
        }
        Ok(Staged::new(slab, 0, file))
    }
}

/// The spans of `extents` in `at`, a [`Memory`] of them in `space`
///
/// # Safety
///
/// As [`save`] requires.
unsafe fn memory<S>(
    extents: &[(Range<usize>, u64, Location)],
    at: Location,
    space: S,
) -> Memory<S> {
    let (spans, addrs): (Vec<Span>, Vec<u64>) = extents
        .iter()
        .filter(|(_, _, location)| *location == at)
        .map(|(r, addr, _)| {
            (
                Span {
                    start: r.start,
                    end: r.end,
                },
                *addr,
            )
        })
        .unzip();
    let plan = LoadPlan::new(spans, vec![None; addrs.len()], 0, CHUNK_SIZE);
    let memory = Memory::new(plan, space);
    // SAFETY: the caller vouches for the addresses
    unsafe { memory.provide(addrs.into_iter(), None) };
    memory
}

/// Writes a file of `size` bytes: `header`, then each tensor's bytes at its
/// range of the data section, on `threads` threads. The file replaces `path`
/// once complete. `tensors` are `(range, address, location)` in file order;
/// work queued on CUDA tensors must be complete.
///
/// # Safety
///
/// Each address is valid for reads of its range's length during the call.
pub unsafe fn save(
    path: &Path,
    header: &[u8],
    tensors: &[(Range<usize>, u64, Location)],
    size: usize,
    threads: usize,
) -> Result<(), Error> {
    let at = header.len();
    let extents: Vec<_> = std::iter::once((0..at, header.as_ptr() as u64, Location::Host))
        .chain(
            tensors
                .iter()
                .map(|(r, addr, location)| (at + r.start..at + r.end, *addr, *location)),
        )
        .collect();
    let mut devices: Vec<i32> = tensors
        .iter()
        .filter_map(|(_, _, location)| match location {
            Location::Cuda(device) => Some(*device),
            Location::Host => None,
        })
        .collect();
    devices.sort_unstable();
    devices.dedup();
    if let Some(device) = devices
        .iter()
        .find(|d| !(0..MAX_DEVICES as i32).contains(d))
    {
        return Err(Error::Other(format!("invalid CUDA device {device}").into()));
    }
    let mut memories = Memories {
        // SAFETY: `header` is borrowed for the call, the caller vouches for the rest
        host: unsafe { memory(&extents, Location::Host, Host) },
        devices: Vec::new(),
    };
    let pool = match devices.first() {
        None => {
            let slabs = threads.min(size.div_ceil(CHUNK_SIZE.get())).max(1);
            SlabPool::new(Box::new(HostMemory::new(slabs * SLAB_SIZE)), SLAB_SIZE)
        }
        Some(&first) => {
            let cuda = api().ok_or_else(|| Error::Other("no CUDA runtime is loaded".into()))?;
            for device in devices {
                let space = CudaDevice::new(cuda, device)?;
                // SAFETY: the caller vouches for the addresses
                let memory = unsafe { memory(&extents, Location::Cuda(device), space) };
                memories.devices.push(memory);
            }
            // device-to-host copies run asynchronously into pinned slabs only
            cuda.with_device(first, |_| pinned_pool(cuda))?
        }
    };
    write_file(path, size as u64, &memories, &pool, threads)
}
