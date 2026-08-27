use std::fmt::Display;
use std::fs::File;
use std::sync::Arc;

use safetensors::tensor::Metadata;

pub enum CudaBuffer {}

impl CudaBuffer {
    pub fn ptr(&self) -> u64 {
        match *self {}
    }

    pub fn len(&self) -> usize {
        match *self {}
    }

    pub fn device(&self) -> i32 {
        match *self {}
    }
}

pub enum DeviceBuffer {
    #[allow(dead_code)]
    Cuda(CudaBuffer),
}

pub struct TensorIter;

impl Iterator for TensorIter {
    type Item = Result<(usize, DeviceBuffer), LoaderError>;
    fn next(&mut self) -> Option<Self::Item> {
        None
    }
}

#[derive(Debug)]
pub struct LoaderError;

impl Display for LoaderError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "prefetch loading is not supported on this platform")
    }
}

pub struct Loader;

impl Loader {
    pub fn load(
        _file: Arc<File>,
        _metadata: &Metadata,
        _buffer_start_pos: usize,
        _device: i32,
        _threads: usize,
    ) -> Result<Self, LoaderError> {
        Err(LoaderError)
    }

    pub fn take_tensor(&self, _tensor: usize) -> Result<DeviceBuffer, LoaderError> {
        Err(LoaderError)
    }

    pub fn iter(&self) -> TensorIter {
        TensorIter
    }
}
