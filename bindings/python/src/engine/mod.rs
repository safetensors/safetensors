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
#[cfg(unix)]
mod transfer;
#[cfg(unix)]
pub use {
    cuda_memory::available,
    loader::{Feed, Loader, LoaderError, TensorIter},
    memory::Span,
};

#[cfg(not(unix))]
mod windows;
#[cfg(not(unix))]
pub use windows::*;
