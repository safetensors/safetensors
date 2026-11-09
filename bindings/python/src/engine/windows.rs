use std::fmt::Display;
use std::fs::File;
use std::marker::PhantomData;
use std::sync::Arc;

pub struct TensorIter<D>(PhantomData<D>);

impl<D> Iterator for TensorIter<D> {
    type Item = Result<(usize, D), LoaderError>;
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

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Span {
    pub start: usize,
    pub end: usize,
}

pub struct Loader<D>(PhantomData<D>);

impl<D> Loader<D> {
    pub fn load(
        _file: Arc<File>,
        _in_file_offset: usize,
        _device: i32,
        _threads: usize,
        _spans: Vec<Span>,
        _dsts: Vec<(u64, D)>,
        _alloc_stream: u64,
    ) -> Result<Self, LoaderError> {
        Err(LoaderError)
    }

    pub fn take_tensor(&self, _idx: usize) -> Result<D, LoaderError> {
        Err(LoaderError)
    }

    pub fn close(&mut self) -> Vec<D> {
        Vec::new()
    }

    pub fn iter(&self) -> TensorIter<D> {
        TensorIter(PhantomData)
    }
}
