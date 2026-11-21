//! `Reader`: reads parts of tensors into tensors the caller gives.

use std::{
    collections::HashMap,
    fs::File,
    path::PathBuf,
    sync::{Arc, Mutex},
};

use pyo3::{exceptions::PyFileNotFoundError, intern, prelude::*, Bound as PyBound};
use safetensors::{
    slice::slice_layout,
    tensor::{Metadata, SafeTensors},
};

use crate::{
    current_stream, engine, get_module, get_pydtype, parse_indexers, torch_storage_shape,
    SafetensorError, TORCH_MODULE,
};

struct Checkpoint {
    file: Arc<File>,
    metadata: Metadata,
    /// where the tensors' bytes start in the file
    data: u64,
}

/// Reads parts of the tensors of one or several safetensors files into tensors
/// the caller gives, in the background.
///
/// A reader is meant to live as long as its files are read from: it keeps them
/// open and keeps its threads. Each `read` returns at once with a
/// `ReadRequest`, and the bytes are in `out` once it was waited.
///
/// Args:
///     filenames (`str`, `os.PathLike` or a list of those):
///         The files to read from. A tensor name appears in one of them only.
///     threads (`int`, *keyword-only*, defaults to 8):
///         How many reads run at once.
#[pyclass(frozen)]
pub struct Reader {
    checkpoints: Vec<Checkpoint>,
    /// tensor name -> its checkpoint
    names: HashMap<String, usize>,
    /// `None` once closed
    reader: Mutex<Option<Arc<engine::Reader>>>,
}

#[pymethods]
impl Reader {
    #[new]
    #[pyo3(signature = (filenames, *, threads=8))]
    fn new(py: Python<'_>, filenames: &PyBound<'_, PyAny>, threads: usize) -> PyResult<Self> {
        let paths: Vec<PathBuf> = match filenames.extract::<PathBuf>() {
            Ok(path) => vec![path],
            Err(_) => filenames
                .try_iter()?
                .map(|path| path?.extract())
                .collect::<PyResult<_>>()?,
        };
        if cfg!(target_endian = "big") {
            return Err(SafetensorError::new_err(
                "Reader needs a little-endian host",
            ));
        }
        let torch = PyModule::import(py, intern!(py, "torch"))?;
        TORCH_MODULE.get_or_init(|| torch.unbind());
        let (mut checkpoints, mut names) = (Vec::new(), HashMap::new());
        for path in paths {
            let file = File::open(&path).map_err(|_| {
                PyFileNotFoundError::new_err(format!(
                    "No such file or directory: {}",
                    path.display()
                ))
            })?;
            // SAFETY: the mapping is only read for the header, here
            let mapped = unsafe { memmap2::MmapOptions::new().map_copy_read_only(&file)? };
            let (n, metadata) = SafeTensors::read_metadata(&mapped).map_err(|e| {
                SafetensorError::new_err(format!("Error while deserializing header: {e}"))
            })?;
            for name in metadata.offset_keys() {
                if names.insert(name.clone(), checkpoints.len()).is_some() {
                    return Err(SafetensorError::new_err(format!(
                        "tensor {name} is in more than one file"
                    )));
                }
            }
            checkpoints.push(Checkpoint {
                file: Arc::new(file),
                metadata,
                data: (n + 8) as u64,
            });
        }
        let reader = engine::Reader::new(threads)
            .map_err(|e| SafetensorError::new_err(format!("cannot start the reader: {e}")))?;
        Ok(Self {
            checkpoints,
            names,
            reader: Mutex::new(Some(Arc::new(reader))),
        })
    }

    /// Starts reading `name[index]` into `out`, and returns at once.
    ///
    /// Args:
    ///     name (`str`):
    ///         The tensor to read from.
    ///     index (*optional*):
    ///         Which part: any index `get_slice` takes whose bytes are contiguous
    ///         in the file, such as a row or a range of rows. `None` reads the
    ///         whole tensor.
    ///     out (`torch.Tensor`):
    ///         Where the bytes go: a contiguous tensor with the dtype and shape
    ///         of `name[index]`, on the CPU, on a CUDA device or on MPS (torch
    ///         2.12 or later). Nothing may read or write it, on its device
    ///         either, until the request was waited.
    ///
    /// Returns:
    ///     (`ReadRequest`):
    ///         The read, to wait on.
    #[pyo3(signature = (name, index, out))]
    fn read(
        &self,
        py: Python<'_>,
        name: &str,
        index: &PyBound<'_, PyAny>,
        out: PyBound<'_, PyAny>,
    ) -> PyResult<ReadRequest> {
        let reader = self.reader.lock().unwrap().clone();
        let reader = reader.ok_or_else(|| SafetensorError::new_err("the reader is closed"))?;
        let checkpoint = self.names.get(name).map(|&i| &self.checkpoints[i]);
        let info = checkpoint
            .and_then(|c| c.metadata.info(name))
            .ok_or_else(|| SafetensorError::new_err(format!("no tensor {name} in the files")))?;
        let checkpoint = checkpoint.expect("found with its tensor");
        let (start, _) = info.data_offsets;
        let (read, shape) = if index.is_none() {
            (0..info.data_offsets.1 - start, info.shape.clone())
        } else {
            let slices: Vec<_> = parse_indexers(index, &info.shape)?
                .into_iter()
                .map(|ix| vec![ix])
                .collect();
            let layout = slice_layout(info.dtype, &info.shape, &slices)
                .map_err(|e| SafetensorError::new_err(format!("invalid index for {name}: {e}")))?;
            if layout.gather.is_some() {
                return Err(SafetensorError::new_err(format!(
                    "the bytes of this index of {name} are not contiguous in the file"
                )));
            }
            (layout.read, layout.shape)
        };

        let torch = get_module(py, &TORCH_MODULE)?;
        let dtype = get_pydtype(torch, info.dtype, false)?;
        let shape = torch_storage_shape(info.dtype, &shape)?;
        let out_shape: Vec<usize> = out.getattr(intern!(py, "shape"))?.extract()?;
        let nbytes: usize = out.getattr(intern!(py, "nbytes"))?.extract()?;
        let contiguous: bool = out.call_method0(intern!(py, "is_contiguous"))?.extract()?;
        if !out.getattr(intern!(py, "dtype"))?.eq(dtype.bind(py))?
            || out_shape != shape
            || nbytes != read.len()
            || !contiguous
        {
            return Err(SafetensorError::new_err(format!(
                "out must be a contiguous {:?} tensor of shape {shape:?} for this index of {name}",
                info.dtype
            )));
        }
        let target = match nbytes {
            // no memory behind an empty tensor
            0 => engine::Target::Host(0),
            _ => target(py, &out, nbytes)?,
        };
        let at = checkpoint.data + (start + read.start) as u64;
        let file = checkpoint.file.clone();
        // SAFETY: the request keeps `out`, checked to hold the bytes read, and
        // waits when dropped
        let request = py
            .detach(|| unsafe { reader.read(&file, at..at + read.len() as u64, target) })
            .map_err(|e| SafetensorError::new_err(format!("cannot read {name}: {e}")))?;
        Ok(ReadRequest {
            request: Some(request),
            _reader: reader,
            _out: out.unbind(),
        })
    }

    /// Waits for the reads in flight and stops the reader's threads.
    fn close(&self, py: Python<'_>) {
        let reader = self.reader.lock().unwrap().take();
        // the last reference, a request's included, joins the threads
        py.detach(|| drop(reader));
    }

    fn __enter__(slf: Py<Self>) -> Py<Self> {
        slf
    }

    fn __exit__(
        &self,
        py: Python<'_>,
        _exc_type: Py<PyAny>,
        _exc_value: Py<PyAny>,
        _traceback: Py<PyAny>,
    ) {
        self.close(py);
    }
}

/// Where the bytes of `out` are, for its device.
#[cfg_attr(
    not(all(target_os = "macos", target_arch = "aarch64")),
    allow(unused_variables)
)]
fn target(py: Python<'_>, out: &PyBound<'_, PyAny>, nbytes: usize) -> PyResult<engine::Target> {
    let device = out.getattr(intern!(py, "device"))?;
    let kind: String = device.getattr(intern!(py, "type"))?.extract()?;
    let addr = || -> PyResult<u64> { out.call_method0(intern!(py, "data_ptr"))?.extract() };
    match kind.as_str() {
        "cpu" => Ok(engine::Target::Host(addr()?)),
        "cuda" => {
            let device: i32 = device.getattr(intern!(py, "index"))?.extract()?;
            let (_, stream) = current_stream(py, device)?;
            Ok(engine::Target::Cuda {
                device,
                addr: addr()?,
                stream,
            })
        }
        #[cfg(all(target_os = "macos", target_arch = "aarch64"))]
        "mps" => {
            let buffer: u64 = out
                .call_method0(intern!(py, "untyped_storage"))?
                .call_method0(intern!(py, "data_ptr"))?
                .extract()?;
            let offset: usize = out.call_method0(intern!(py, "storage_offset"))?.extract()?;
            let item: usize = out.call_method0(intern!(py, "element_size"))?.extract()?;
            // SAFETY: the data pointer of a torch storage on MPS is its Metal buffer
            unsafe { crate::metal::host_address(buffer as _, offset * item, nbytes) }
                .map(engine::Target::Host)
                .ok_or_else(|| {
                    SafetensorError::new_err(
                        "the host cannot write this MPS tensor: torch allocates MPS \
                         memory the host can write from 2.12 on",
                    )
                })
        }
        kind => Err(SafetensorError::new_err(format!(
            "cannot read into a tensor on {kind}"
        ))),
    }
}

/// One read of a `Reader`.
#[pyclass(frozen)]
pub struct ReadRequest {
    /// `None` once dropped
    request: Option<engine::Request>,
    _reader: Arc<engine::Reader>,
    _out: Py<PyAny>,
}

#[pymethods]
impl ReadRequest {
    /// Returns once the bytes are in `out`. Raises if the read failed.
    fn wait(&self, py: Python<'_>) -> PyResult<()> {
        let request = self.request.as_ref().expect("until dropped");
        py.detach(|| request.wait())
            .map_err(|e| SafetensorError::new_err(format!("read failed: {e}")))
    }

    /// Whether `wait` would return at once.
    fn done(&self) -> PyResult<bool> {
        let request = self.request.as_ref().expect("until dropped");
        request
            .done()
            .map_err(|e| SafetensorError::new_err(format!("read failed: {e}")))
    }
}

impl Drop for ReadRequest {
    fn drop(&mut self) {
        // `out` is written until the read landed
        if let Some(request) = self.request.take() {
            Python::attach(|py| {
                py.detach(|| {
                    let _ = request.wait();
                })
            });
        }
    }
}
