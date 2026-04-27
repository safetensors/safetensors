//! Errors produced by the ionic-rs engine.
//!
//! Pyo3-free by design — the bindings crate provides a `From<Error> for PyErr`
//! adapter to surface failures to Python.

#[derive(Debug)]
pub enum Error {
    /// Tensor name not present in the open safetensors file.
    UnknownTensor(String),
    /// CUDA driver call failed. Carries the driver's `cuResult` code and the
    /// symbol that produced it.
    Cuda { symbol: &'static str, code: i32 },
    /// CUDA driver was not loadable at runtime (`libcuda.so.1` absent).
    CudaUnavailable(String),
    /// io_uring submit/poll/register call failed.
    IoUring { op: &'static str, errno: i32 },
    /// Failure while parsing sysfs for NUMA topology. Non-fatal — callers
    /// fall back to not pinning. Surfaced for diagnostics only.
    NumaProbe(String),
    /// Generic catch-all with context.
    Other(String),
}

impl std::fmt::Display for Error {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Error::UnknownTensor(name) => write!(f, "unknown tensor: {name}"),
            Error::Cuda { symbol, code } => write!(f, "CUDA {symbol} failed with code {code}"),
            Error::CudaUnavailable(msg) => write!(f, "CUDA driver unavailable: {msg}"),
            Error::IoUring { op, errno } => write!(f, "io_uring {op} failed with errno {errno}"),
            Error::NumaProbe(msg) => write!(f, "NUMA probe failed: {msg}"),
            Error::Other(msg) => f.write_str(msg),
        }
    }
}

impl std::error::Error for Error {}

pub type Result<T> = std::result::Result<T, Error>;
