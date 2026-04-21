//! Errors produced by the prefetch pipeline.
//!
//! Kept as a lightweight enum here; converted into `SafetensorError` at the
//! Python boundary so callers see a single exception type regardless of
//! whether the failure came from io_uring, CUDA, or metadata lookup.

use pyo3::PyErr;

use crate::SafetensorError;

#[derive(Debug)]
pub enum PipelineError {
    /// Tensor name not present in the open safetensors file.
    UnknownTensor(String),
    /// CUDA driver call failed. Carries the driver's `cuResult` code and the
    /// symbol that produced it.
    Cuda {
        symbol: &'static str,
        code: i32,
    },
    /// CUDA driver was not loadable at runtime (`libcuda.so.1` absent).
    CudaUnavailable(String),
    /// io_uring submit/poll/register call failed.
    IoUring {
        op: &'static str,
        errno: i32,
    },
    /// Failure while parsing sysfs for NUMA topology. Non-fatal — callers
    /// fall back to not pinning. Surfaced for diagnostics only.
    NumaProbe(String),
    /// Generic catch-all with context.
    Other(String),
}

impl std::fmt::Display for PipelineError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            PipelineError::UnknownTensor(name) => {
                write!(f, "unknown tensor: {name}")
            }
            PipelineError::Cuda { symbol, code } => {
                write!(f, "CUDA {symbol} failed with code {code}")
            }
            PipelineError::CudaUnavailable(msg) => {
                write!(f, "CUDA driver unavailable: {msg}")
            }
            PipelineError::IoUring { op, errno } => {
                write!(f, "io_uring {op} failed with errno {errno}")
            }
            PipelineError::NumaProbe(msg) => {
                write!(f, "NUMA probe failed: {msg}")
            }
            PipelineError::Other(msg) => f.write_str(msg),
        }
    }
}

impl std::error::Error for PipelineError {}

impl From<PipelineError> for PyErr {
    fn from(e: PipelineError) -> Self {
        SafetensorError::new_err(e.to_string())
    }
}

pub type PipelineResult<T> = Result<T, PipelineError>;
