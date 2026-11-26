//! `save_file`: a safetensors file written from tensors in memory.

use std::{ops::Range, path::Path};

use super::{
    file::write_file,
    memory::{Host, LoadPlan, Memory, Span},
    pool::{HostMemory, SlabPool, SLAB_SIZE},
    transfer::{Error, CHUNK_SIZE},
};

/// Writes a file of `size` bytes: `header`, then each tensor's bytes at its
/// range of the data section, on `threads` threads. The file replaces `path`
/// once complete. `tensors` are `(range, address)` in file order.
///
/// # Safety
///
/// Each address is valid for reads of its range's length during the call.
pub unsafe fn save(
    path: &Path,
    header: &[u8],
    tensors: &[(Range<usize>, u64)],
    size: usize,
    threads: usize,
) -> Result<(), Error> {
    let at = header.len();
    let spans: Vec<Span> = std::iter::once(0..at)
        .chain(tensors.iter().map(|(r, _)| at + r.start..at + r.end))
        .map(|r| Span {
            start: r.start,
            end: r.end,
        })
        .collect();
    let addrs: Vec<u64> = std::iter::once(header.as_ptr() as u64)
        .chain(tensors.iter().map(|&(_, addr)| addr))
        .collect();
    let plan = LoadPlan::new(spans, vec![None; addrs.len()], 0, CHUNK_SIZE);
    let memory = Memory::new(plan, Host);
    // SAFETY: `header` is borrowed for the call, the caller vouches for the rest
    unsafe { memory.provide(addrs.into_iter(), None) };
    let slabs = threads.min(size.div_ceil(CHUNK_SIZE.get())).max(1);
    let pool = SlabPool::new(Box::new(HostMemory::new(slabs * SLAB_SIZE)), SLAB_SIZE);
    write_file(path, size as u64, &memory, &pool, threads)
}
