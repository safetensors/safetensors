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

pub struct Feed<D>(PhantomData<D>);

impl<D> Feed<D> {
    #[allow(clippy::type_complexity)]
    pub fn provide(
        &self,
        dsts: Vec<(u64, D)>,
        _stream: u64,
    ) -> Result<(), (LoaderError, Vec<(u64, D)>)> {
        Err((LoaderError, dsts))
    }
}

impl<D> Loader<D> {
    pub fn load(
        _file: Arc<File>,
        _in_file_offset: usize,
        _device: i32,
        _threads: usize,
        _spans: Vec<Span>,
    ) -> Result<Self, LoaderError> {
        Err(LoaderError)
    }

    pub fn feed(&self) -> Feed<D> {
        Feed(PhantomData)
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
