#[cfg(unix)]
mod cuda;
#[cfg(unix)]
mod cuda_memory;
#[cfg(unix)]
mod file;
#[cfg(unix)]
mod loader;
#[cfg(unix)]
mod memory;
#[cfg(unix)]
mod pool;
#[cfg(all(unix, target_endian = "little"))]
mod read;
#[cfg(unix)]
mod reader;
#[cfg(target_os = "linux")]
mod save;
#[cfg(unix)]
mod transfer;
#[cfg(all(unix, target_endian = "little"))]
pub use read::read;
#[cfg(target_os = "linux")]
pub use save::{save, Location};
#[cfg(unix)]
pub use {
    cuda_memory::available,
    loader::{Feed, Loader, LoaderError, TensorIter},
    memory::Span,
    reader::{Reader, Request, Target},
};

#[cfg(not(unix))]
mod windows;
#[cfg(not(unix))]
pub use windows::*;
