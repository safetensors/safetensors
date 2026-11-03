#[cfg(unix)]
mod cuda;

#[cfg(unix)]
mod unix;
#[cfg(unix)]
pub use unix::*;

#[cfg(not(unix))]
mod windows;
#[cfg(not(unix))]
pub use windows::*;

/// Device allocator of the framework the tensors are handed to (torch's
/// caching allocator), so its accounting (`memory_allocated`, `empty_cache`,
/// OOM handling) covers prefetched tensors. `alloc(device, bytes, stream)` and
/// `free(device, ptr, stream)` are stream-ordered on `stream`.
pub struct ExternalAllocator {
    pub alloc: Box<dyn Fn(i32, usize, u64) -> Result<u64, String> + Send + Sync>,
    pub free: Box<dyn Fn(i32, u64, u64) + Send + Sync>,
}
