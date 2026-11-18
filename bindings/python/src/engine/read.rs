//! `load_file` on the CPU: tensors read from the file in parallel.

use std::{fs::File, ops::Range, sync::Arc};

use super::{
    file::DirectFile,
    memory::{Host, LoadPlan, Memory, Span},
    pool::{HostMemory, SlabPool, SLAB_SIZE},
    transfer::{transfer, Error, Plan, CHUNK_SIZE},
};

/// Reads each tensor's bytes, at its range of the data section that starts
/// at `offset` in `file`, to its address, on `threads` threads. `tensors` are
/// `(range, address)` in file order. Chunks the page cache holds are read
/// from it, the others with `O_DIRECT` where the filesystem allows it.
///
/// # Safety
///
/// Each address is valid for writes of its range's length during the call,
/// and nothing else accesses those bytes meanwhile.
pub unsafe fn read(
    file: Arc<File>,
    offset: usize,
    tensors: &[(Range<usize>, u64)],
    threads: usize,
) -> Result<(), Error> {
    let (spans, addrs): (Vec<Span>, Vec<u64>) = tensors
        .iter()
        .map(|(r, addr)| {
            let span = Span {
                start: r.start,
                end: r.end,
            };
            (span, *addr)
        })
        .unzip();
    let plan = LoadPlan::new(spans, vec![None; addrs.len()], offset, CHUNK_SIZE);
    let memory = Memory::new(plan, Host);
    // SAFETY: the caller vouches for the addresses
    unsafe { memory.provide(addrs.into_iter(), None) };
    let chunks = memory.chunks();
    let slabs = threads.min(chunks.len()).max(1);
    let pool = SlabPool::new(Box::new(HostMemory::new(slabs * SLAB_SIZE)), SLAB_SIZE);
    let source = DirectFile::new(file, None, pool.aligned());
    transfer(Plan::new(chunks), &source, &memory, &pool, threads)
}
