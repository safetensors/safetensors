#![deny(missing_docs)]
//! Dummy doc
#[cfg(all(target_os = "macos", target_arch = "aarch64"))]
mod dlpack;
#[cfg(all(target_os = "macos", target_arch = "aarch64"))]
mod metal;

mod engine;
#[cfg(unix)]
mod reader;

use core::slice;
use memmap2::{Mmap, MmapOptions};
use pyo3::buffer::PyUntypedBuffer;
use pyo3::exceptions::{PyException, PyFileNotFoundError};
use pyo3::prelude::*;
use pyo3::sync::{OnceLockExt, PyOnceLock};
use pyo3::types::IntoPyDict;
use pyo3::types::{PyBool, PyByteArray, PyBytes, PyDict, PyEllipsis, PyList, PySlice, PyTuple};
use pyo3::Bound as PyBound;
use pyo3::{intern, PyErr};
use safetensors::slice::{slice_layout, InvalidSlice, TensorIndexer};
use safetensors::tensor::{Dtype, Metadata, SafeTensors, TensorInfo, TensorView};
use safetensors::View;
use std::borrow::Cow;
use std::collections::{BTreeMap, HashMap};
use std::fs::File;
use std::num::NonZeroUsize;
use std::ops::Bound;
use std::path::PathBuf;
use std::sync::atomic::{AtomicBool, AtomicU32, Ordering};
use std::sync::OnceLock;
use std::sync::{Arc, Condvar, Mutex, Weak};
use std::thread::JoinHandle;

static TORCH_MODULE: OnceLock<Py<PyModule>> = OnceLock::new();
static NUMPY_MODULE: OnceLock<Py<PyModule>> = OnceLock::new();
static TENSORFLOW_MODULE: OnceLock<Py<PyModule>> = OnceLock::new();
static FLAX_MODULE: OnceLock<Py<PyModule>> = OnceLock::new();
static MLX_MODULE: OnceLock<Py<PyModule>> = OnceLock::new();
static PADDLE_MODULE: OnceLock<Py<PyModule>> = OnceLock::new();

#[cfg(all(target_os = "macos", target_arch = "aarch64"))]
static TORCH_MPS_DLPACK: OnceLock<bool> = OnceLock::new();

/// Describes a single tensor passed to [`serialize`] / [`serialize_file`].
///
/// Constructed from Python as `TensorSpec(dtype, shape, data)`.
/// The dtype string is validated at construction; an unknown dtype raises
/// immediately rather than failing further inside the serializer.
///
/// `shape` is the logical (header) shape: the number of elements along each
/// axis as recorded in the safetensors header. For packed dtypes like
/// `float4_e2m1fn_x2` (two F4 values per byte), callers may pass the storage
/// shape reported by their framework (e.g. `torch.Size`); the constructor
/// transparently doubles the last dimension so `spec.shape` always reflects
/// the logical element count.
///
/// `data` is an object exposing the tensor's bytes through the buffer protocol
/// (e.g. a numpy array, `bytes` or a `memoryview`): C-contiguous, in
/// little-endian order, as many bytes as `dtype` and `shape` take. The spec
/// holds the buffer, which keeps the memory alive until the spec is dropped.
///
/// `TensorSpec.cuda` describes a tensor on a CUDA device: only
/// `serialize_file`, on Linux, reads such a spec.
#[pyclass(frozen)]
struct TensorSpec {
    dtype: Dtype,
    shape: Vec<usize>,
    data: Data,
}

/// Where a spec's bytes are
enum Data {
    Host(PyUntypedBuffer),
    /// `len` bytes at `ptr` on a CUDA device, which `_owner` keeps alive
    Cuda {
        #[cfg_attr(not(target_os = "linux"), allow(dead_code))]
        ptr: u64,
        len: usize,
        device: i32,
        _owner: Py<PyAny>,
    },
}

/// The header's dtype and shape for a framework's, and the bytes they take
fn spec_layout(dtype: &str, mut shape: Vec<usize>) -> PyResult<(Dtype, Vec<usize>, usize)> {
    let dtype = parse_dtype_str(dtype)?;
    // F4 packs two elements per byte; the safetensors header records the
    // logical element count, so double the last dim.
    if dtype == Dtype::F4 && !shape.is_empty() {
        let n = shape.len();
        shape[n - 1] = shape[n - 1].checked_mul(2).ok_or_else(|| {
            SafetensorError::new_err(format!(
                "F4 last-dim {} doubled to logical shape overflows usize",
                shape[n - 1]
            ))
        })?;
    }
    let len = shape
        .iter()
        .try_fold(dtype.bitsize(), |bits, dim| bits.checked_mul(*dim))
        .ok_or_else(|| {
            SafetensorError::new_err(format!("{dtype} of shape {shape:?} overflows usize"))
        })?
        .div_ceil(8);
    Ok((dtype, shape, len))
}

#[pymethods]
impl TensorSpec {
    #[new]
    #[pyo3(signature = (*, dtype, shape, data))]
    fn new(dtype: &str, shape: Vec<usize>, data: &PyBound<'_, PyAny>) -> PyResult<Self> {
        let (dtype, shape, len) = spec_layout(dtype, shape)?;
        let data = PyUntypedBuffer::get(data)?;
        if !data.is_c_contiguous() {
            return Err(SafetensorError::new_err("data is not C-contiguous"));
        }
        if data.len_bytes() != len {
            return Err(SafetensorError::new_err(format!(
                "data is {} bytes, which {dtype} of shape {shape:?} is not",
                data.len_bytes()
            )));
        }
        Ok(Self {
            dtype,
            shape,
            data: Data::Host(data),
        })
    }

    /// Describes a tensor on CUDA device `device`: `data_ptr` is the device
    /// address of its bytes, as many as `dtype` and `shape` take, and the spec
    /// holds `owner` to keep them alive.
    ///
    /// SAFETY: `data_ptr` is a raw device address. The caller must ensure it
    /// is valid for as long as `owner` is alive.
    #[staticmethod]
    #[pyo3(signature = (*, dtype, shape, owner, data_ptr, device))]
    fn cuda(
        dtype: &str,
        shape: Vec<usize>,
        owner: Py<PyAny>,
        data_ptr: u64,
        device: i32,
    ) -> PyResult<Self> {
        let (dtype, shape, len) = spec_layout(dtype, shape)?;
        Ok(Self {
            dtype,
            shape,
            data: Data::Cuda {
                ptr: data_ptr,
                len,
                device,
                _owner: owner,
            },
        })
    }

    /// The tensor's dtype as its safetensors format code (e.g. `"F32"`, `"BF16"`,
    /// `"F8_E5M2FNUZ"`). This is the identifier written into the safetensors
    /// header, not the Python constructor-style name (`"float32"` etc.).
    #[getter]
    fn dtype(&self) -> String {
        format!("{}", self.dtype)
    }

    /// The tensor's logical shape: the element-count shape recorded in the
    /// safetensors header. For packed dtypes like `float4_e2m1fn_x2`, this is
    /// the last-dim-doubled version of whatever was passed to the constructor.
    #[getter]
    fn shape(&self) -> Vec<usize> {
        self.shape.clone()
    }

    /// The length of the tensor's buffer in bytes.
    #[getter]
    fn data_len(&self) -> usize {
        match &self.data {
            Data::Host(buffer) => buffer.len_bytes(),
            Data::Cuda { len, .. } => *len,
        }
    }

    /// The index of the CUDA device the tensor is on, `None` for host memory.
    #[getter]
    fn device(&self) -> Option<i32> {
        match self.data {
            Data::Host(_) => None,
            Data::Cuda { device, .. } => Some(device),
        }
    }

    fn __repr__(&self) -> String {
        let device = match self.device() {
            Some(d) => format!(", device={d}"),
            None => String::new(),
        };
        format!(
            "TensorSpec(dtype='{}', shape={:?}, data_len={}{device})",
            self.dtype(),
            self.shape,
            self.data_len()
        )
    }
}

/// Fails on a spec in device memory, which only `serialize_file` on Linux reads.
fn host_specs(tensor_dict: &HashMap<String, Py<TensorSpec>>) -> PyResult<()> {
    match tensor_dict.iter().find(|(_, v)| v.get().device().is_some()) {
        Some((name, _)) => Err(SafetensorError::new_err(format!(
            "Error while serializing: {name} is in device memory, which only \
             serialize_file on Linux reads"
        ))),
        None => Ok(()),
    }
}

impl View for &TensorSpec {
    fn dtype(&self) -> Dtype {
        self.dtype
    }

    fn shape(&self) -> &[usize] {
        &self.shape
    }

    fn data(&self) -> Cow<'_, [u8]> {
        let Data::Host(buffer) = &self.data else {
            panic!("device memory read as host memory")
        };
        let len = buffer.len_bytes();
        if len == 0 {
            return Cow::Borrowed(&[]);
        }
        // SAFETY: `buffer` keeps `len` contiguous bytes alive at this address
        Cow::Borrowed(unsafe { slice::from_raw_parts(buffer.buf_ptr() as *const u8, len) })
    }

    fn data_len(&self) -> usize {
        TensorSpec::data_len(self)
    }
}

fn parse_dtype_str(dtype: &str) -> PyResult<Dtype> {
    Ok(match dtype {
        "bool" => Dtype::BOOL,
        "int8" => Dtype::I8,
        "uint8" => Dtype::U8,
        "int16" => Dtype::I16,
        "uint16" => Dtype::U16,
        "int32" => Dtype::I32,
        "uint32" => Dtype::U32,
        "int64" => Dtype::I64,
        "uint64" => Dtype::U64,
        "float16" => Dtype::F16,
        "float32" => Dtype::F32,
        "float64" => Dtype::F64,
        "bfloat16" => Dtype::BF16,
        "float8_e4m3fn" => Dtype::F8_E4M3,
        "float8_e4m3fnuz" => Dtype::F8_E4M3FNUZ,
        "float8_e5m2" => Dtype::F8_E5M2,
        "float8_e5m2fnuz" => Dtype::F8_E5M2FNUZ,
        "float8_e8m0fnu" => Dtype::F8_E8M0,
        "float4_e2m1fn_x2" => Dtype::F4,
        "complex64" => Dtype::C64,
        other => {
            return Err(SafetensorError::new_err(format!(
                "Unknown dtype {other:?}. Supported dtypes: bool, int8, uint8, int16, uint16, \
                 int32, uint32, int64, uint64, float16, float32, float64, bfloat16, \
                 float8_e4m3fn, float8_e4m3fnuz, float8_e5m2, float8_e5m2fnuz, float8_e8m0fnu, \
                 float4_e2m1fn_x2, complex64",
            )));
        }
    })
}

/// Serializes raw data.
///
/// Args:
///     tensor_dict (`Dict[str, TensorSpec]`):
///         Mapping of tensor name to its `TensorSpec`, e.g.:
///             {"tensor_name": TensorSpec(dtype="float32", shape=[2, 3], data=array)}
///     metadata (`Dict[str, str]`, *optional*):
///         The optional purely text annotations
///
/// Returns:
///     (`bytes`):
///         The serialized content.
#[pyfunction]
#[pyo3(signature = (tensor_dict, metadata=None))]
fn serialize<'b>(
    py: Python<'b>,
    tensor_dict: HashMap<String, Py<TensorSpec>>,
    metadata: Option<HashMap<String, String>>,
) -> PyResult<PyBound<'b, PyBytes>> {
    host_specs(&tensor_dict)?;
    let out = py
        .detach(|| {
            safetensors::tensor::serialize(
                tensor_dict.iter().map(|(k, v)| (k.as_str(), v.get())),
                metadata,
            )
        })
        .map_err(|e| SafetensorError::new_err(format!("Error while serializing: {e}")))?;
    let pybytes = PyBytes::new(py, &out);
    Ok(pybytes)
}

/// Threads that write a file in `serialize_file`
#[cfg(target_os = "linux")]
const SAVE_THREADS: usize = 16;

/// Writes the file `serialize_to_file` writes, its header and tensor offsets
/// from `file_layout`, the tensors' bytes from their buffers.
#[cfg(target_os = "linux")]
fn write_tensors(
    tensor_dict: &HashMap<String, Py<TensorSpec>>,
    metadata: Option<HashMap<String, String>>,
    filename: &std::path::Path,
) -> Result<(), String> {
    let layout = safetensors::tensor::file_layout(
        tensor_dict.iter().map(|(k, v)| (k.as_str(), v.get())),
        metadata,
    )
    .map_err(|e| e.to_string())?;
    let tensors: Vec<_> = layout
        .tensors
        .iter()
        .map(|(name, range)| match &tensor_dict[name].get().data {
            Data::Host(buffer) => (
                range.clone(),
                buffer.buf_ptr() as u64,
                engine::Location::Host,
            ),
            Data::Cuda { ptr, device, .. } => {
                (range.clone(), *ptr, engine::Location::Cuda(*device))
            }
        })
        .collect();
    let size = layout.header.len() + layout.data_len;
    // SAFETY: each spec holds a buffer of `data_len` bytes, the length of its
    // range, or vouches for as many at its device address (see `TensorSpec::cuda`)
    unsafe { engine::save(filename, &layout.header, &tensors, size, SAVE_THREADS) }
        .map_err(|e| e.to_string())
}

/// Serializes raw data into file.
///
/// On Linux, a tensor on a CUDA device (`TensorSpec.cuda`) is read from the
/// device: work that writes it must be complete before the call.
///
/// Args:
///     tensor_dict (`Dict[str, TensorSpec]`):
///         Mapping of tensor name to its `TensorSpec`, e.g.:
///             {"tensor_name": TensorSpec(dtype="float32", shape=[2, 3], data=array)}
///     filename (`str`, or `os.PathLike`):
///         The name of the file to write into.
///     metadata (`Dict[str, str]`, *optional*):
///         The optional purely text annotations
///
/// Returns:
///     (`NoneType`):
///         On success return None
#[pyfunction]
#[pyo3(signature = (tensor_dict, filename, metadata=None))]
fn serialize_file(
    py: Python<'_>,
    tensor_dict: HashMap<String, Py<TensorSpec>>,
    filename: PathBuf,
    metadata: Option<HashMap<String, String>>,
) -> PyResult<()> {
    #[cfg(not(target_os = "linux"))]
    host_specs(&tensor_dict)?;
    py.detach(|| {
        #[cfg(target_os = "linux")]
        let written = write_tensors(&tensor_dict, metadata, &filename);
        #[cfg(not(target_os = "linux"))]
        let written = safetensors::tensor::serialize_to_file(
            tensor_dict.iter().map(|(k, v)| (k.as_str(), v.get())),
            metadata,
            filename.as_path(),
        )
        .map_err(|e| e.to_string());
        written.map_err(|e| SafetensorError::new_err(format!("Error while serializing: {e}")))
    })
}

/// Opens a safetensors lazily and returns tensors as asked
///
/// Args:
///     data (`bytes`):
///         The byte content of a file
///
/// Returns:
///     (`List[str, Dict[str, Dict[str, any]]]`):
///         The deserialized content is like:
///             [("tensor_name", {"shape": [2, 3], "dtype": "F32", "data":
/// b"\0\0.." }), (...)]
#[pyfunction]
#[pyo3(signature = (bytes))]
#[allow(clippy::type_complexity)]
fn deserialize(py: Python, bytes: &[u8]) -> PyResult<Vec<(String, HashMap<String, Py<PyAny>>)>> {
    let safetensor = SafeTensors::deserialize(bytes)
        .map_err(|e| SafetensorError::new_err(format!("Error while deserializing: {e}")))?;

    let tensors = safetensor.tensors();
    let mut items = Vec::with_capacity(tensors.len());

    for (tensor_name, tensor) in tensors {
        let pyshape: Py<PyAny> = PyList::new(py, tensor.shape().iter())?.into();
        let pydtype: Py<PyAny> = tensor.dtype().to_string().into_pyobject(py)?.into();

        let pydata: Py<PyAny> = PyByteArray::new(py, tensor.data()).into();

        let map = HashMap::from([
            ("shape".to_string(), pyshape),
            ("dtype".to_string(), pydtype),
            ("data".to_string(), pydata),
        ]);
        items.push((tensor_name, map));
    }
    Ok(items)
}

/// Parses a Python index (`int`, `slice`, `...`, or a tuple of those) into
/// byte-range [`TensorIndexer`]s. The slice step is carried in
/// [`TensorIndexer::Narrow`]; step <= 0 is rejected.
fn parse_indexers(slices: &PyBound<'_, PyAny>, shape: &[usize]) -> PyResult<Vec<TensorIndexer>> {
    let py = slices.py();

    let items: Vec<PyBound<'_, PyAny>> = if let Ok(tup) = slices.cast::<PyTuple>() {
        tup.iter().collect()
    } else if let Ok(lst) = slices.cast::<PyList>() {
        if lst.is_empty() {
            // Empty list = empty selection, matching the prior behavior.
            return Ok(vec![TensorIndexer::Narrow(
                Bound::Included(0),
                Bound::Excluded(0),
                NonZeroUsize::MIN,
            )]);
        }
        return Err(SafetensorError::new_err(
            "Non empty lists are not implemented",
        ));
    } else {
        vec![slices.clone()]
    };

    let ellipsis_count = items
        .iter()
        .filter(|it| it.is_instance_of::<PyEllipsis>())
        .count();
    if ellipsis_count > 1 {
        return Err(SafetensorError::new_err(
            "Only one ellipsis (...) is allowed in slice index",
        ));
    }
    let n_explicit = items.len() - ellipsis_count;
    let n_expansion = shape.len().saturating_sub(n_explicit);

    let mut indexers: Vec<TensorIndexer> = Vec::with_capacity(shape.len());
    let mut dim_idx = 0usize;
    for it in items {
        if it.is_instance_of::<PyEllipsis>() {
            for _ in 0..n_expansion {
                indexers.push(TensorIndexer::Narrow(
                    Bound::Unbounded,
                    Bound::Unbounded,
                    NonZeroUsize::MIN,
                ));
                dim_idx += 1;
            }
        } else if let Ok(slice) = it.cast::<PySlice>() {
            let dim = shape.get(dim_idx).copied().unwrap_or(0);
            // Negative bounds count from the end, clamped at 0 as in Python.
            let resolve = |bound: Option<isize>| {
                bound.map(|b| usize::try_from(b).unwrap_or_else(|_| dim.saturating_add_signed(b)))
            };
            let start = resolve(slice.getattr(intern!(py, "start"))?.extract()?);
            let stop = resolve(slice.getattr(intern!(py, "stop"))?.extract()?);
            let step_raw: i64 = slice
                .getattr(intern!(py, "step"))?
                .extract::<Option<i64>>()?
                .unwrap_or(1);
            let step = usize::try_from(step_raw)
                .ok()
                .and_then(NonZeroUsize::new)
                .ok_or_else(|| {
                    SafetensorError::new_err(format!(
                        "Slice step must be a positive integer, got {step_raw}"
                    ))
                })?;
            let start_b = match start {
                Some(s) => Bound::Included(s),
                None => Bound::Unbounded,
            };
            let stop_b = match stop {
                Some(s) => Bound::Excluded(s),
                None => Bound::Unbounded,
            };
            indexers.push(TensorIndexer::Narrow(start_b, stop_b, step));
            dim_idx += 1;
        } else if let Ok(idx) = it.extract::<i32>() {
            let dim = shape.get(dim_idx).copied().unwrap_or(0);
            let resolved = if idx < 0 {
                dim.checked_add_signed(idx as isize).ok_or_else(|| {
                    SafetensorError::new_err(format!(
                        "Invalid index {idx} for dimension {dim_idx} of size {dim}"
                    ))
                })?
            } else {
                idx as usize
            };
            indexers.push(TensorIndexer::Select(resolved));
            dim_idx += 1;
        } else {
            return Err(SafetensorError::new_err(format!(
                "Unsupported slice index at position {dim_idx}: expected slice, int, or ellipsis"
            )));
        }
    }

    Ok(indexers)
}

/// Storage backend used to serve tensor bytes.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Backend {
    Mmap,
    /// Keeps the file handle open and serves each `get_tensor` /
    /// `get_slice` via `pread(2)` (or its Windows equivalent),
    /// dispatching on `(framework, device)` to write directly into a
    /// destination buffer chosen for performance.
    Pread,
}

impl fmt::Display for Backend {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(match *self {
            Backend::Mmap => "mmap",
            Backend::Pread => "pread",
        })
    }
}

impl<'a, 'py> FromPyObject<'a, 'py> for Backend {
    type Error = PyErr;

    fn extract(ob: Borrowed<'a, 'py, PyAny>) -> Result<Self, Self::Error> {
        let name: String = ob.extract()?;
        match &name[..] {
            "mmap" => Ok(Backend::Mmap),
            "pread" => Ok(Backend::Pread),
            name => Err(SafetensorError::new_err(format!(
                "backend {name:?} is invalid (expected one of: \"mmap\", \"pread\")"
            ))),
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
enum Framework {
    Pytorch,
    Numpy,
    Tensorflow,
    Flax,
    Mlx,
    Paddle,
}

impl fmt::Display for Framework {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(match *self {
            Framework::Pytorch => "pytorch",
            Framework::Numpy => "numpy",
            Framework::Tensorflow => "tensorflow",
            Framework::Flax => "flax",
            Framework::Mlx => "mlx",
            Framework::Paddle => "paddle",
        })
    }
}

impl<'a, 'py> FromPyObject<'a, 'py> for Framework {
    type Error = PyErr;

    fn extract(ob: Borrowed<'a, 'py, PyAny>) -> Result<Self, Self::Error> {
        let name: String = ob.extract()?;
        match &name[..] {
            "pt" => Ok(Framework::Pytorch),
            "torch" => Ok(Framework::Pytorch),
            "pytorch" => Ok(Framework::Pytorch),

            "np" => Ok(Framework::Numpy),
            "numpy" => Ok(Framework::Numpy),

            "tf" => Ok(Framework::Tensorflow),
            "tensorflow" => Ok(Framework::Tensorflow),

            "jax" => Ok(Framework::Flax),
            "flax" => Ok(Framework::Flax),
            "mlx" => Ok(Framework::Mlx),

            "paddle" => Ok(Framework::Paddle),
            name => Err(SafetensorError::new_err(format!(
                "framework {name} is invalid"
            ))),
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
enum Device {
    Cpu,
    Cuda(usize),
    Mps,
    Npu(usize),
    Xpu(usize),
    Xla(usize),
    Mlu(usize),
    Musa(usize),
    Hpu(usize),
    /// User didn't specify accelerator, torch
    /// is responsible for choosing.
    Anonymous(usize),
}

impl fmt::Display for Device {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match *self {
            Device::Cpu => write!(f, "cpu"),
            Device::Mps => write!(f, "mps"),
            Device::Cuda(index) => write!(f, "cuda:{index}"),
            Device::Musa(index) => write!(f, "musa:{index}"),
            Device::Npu(index) => write!(f, "npu:{index}"),
            Device::Xpu(index) => write!(f, "xpu:{index}"),
            Device::Xla(index) => write!(f, "xla:{index}"),
            Device::Mlu(index) => write!(f, "mlu:{index}"),
            Device::Hpu(index) => write!(f, "hpu:{index}"),
            Device::Anonymous(index) => write!(f, "{index}"),
        }
    }
}

/// Parsing the device index.
fn parse_device(name: &str) -> PyResult<usize> {
    let tokens: Vec<_> = name.split(':').collect();
    if tokens.len() == 2 {
        Ok(tokens[1].parse()?)
    } else {
        Err(SafetensorError::new_err(format!(
            "device {name} is invalid"
        )))
    }
}

impl<'a, 'py> FromPyObject<'a, 'py> for Device {
    type Error = PyErr;

    fn extract(ob: Borrowed<'a, 'py, PyAny>) -> Result<Self, Self::Error> {
        if let Ok(name) = ob.extract::<String>() {
            match name.as_str() {
                "cpu" => Ok(Device::Cpu),
                "cuda" => Ok(Device::Cuda(0)),
                "musa" => Ok(Device::Musa(0)),
                "mps" => Ok(Device::Mps),
                "npu" => Ok(Device::Npu(0)),
                "xpu" => Ok(Device::Xpu(0)),
                "xla" => Ok(Device::Xla(0)),
                "mlu" => Ok(Device::Mlu(0)),
                "hpu" => Ok(Device::Hpu(0)),
                name if name.starts_with("cuda:") => parse_device(name).map(Device::Cuda),
                name if name.starts_with("musa:") => parse_device(name).map(Device::Musa),
                name if name.starts_with("mps:") => match parse_device(name)? {
                    0 => Ok(Device::Mps),
                    _ => Err(SafetensorError::new_err(format!(
                        "device {name} is invalid: only mps or mps:0 is supported"
                    ))),
                },
                name if name.starts_with("npu:") => parse_device(name).map(Device::Npu),
                name if name.starts_with("xpu:") => parse_device(name).map(Device::Xpu),
                name if name.starts_with("xla:") => parse_device(name).map(Device::Xla),
                name if name.starts_with("mlu:") => parse_device(name).map(Device::Mlu),
                name if name.starts_with("hpu:") => parse_device(name).map(Device::Hpu),
                name => Err(SafetensorError::new_err(format!(
                    "device {name} is invalid"
                ))),
            }
        } else if let Ok(number) = ob.extract::<usize>() {
            Ok(Device::Anonymous(number))
        } else {
            Err(SafetensorError::new_err(format!(
                "device {ob:?} is invalid"
            )))
        }
    }
}

impl<'py> IntoPyObject<'py> for Device {
    type Target = PyAny;
    type Output = pyo3::Bound<'py, Self::Target>;
    type Error = std::convert::Infallible;

    fn into_pyobject(self, py: Python<'py>) -> Result<Self::Output, Self::Error> {
        match self {
            Device::Cpu => "cpu".into_pyobject(py).map(|x| x.into_any()),
            Device::Cuda(n) => format!("cuda:{n}").into_pyobject(py).map(|x| x.into_any()),
            Device::Musa(n) => format!("musa:{n}").into_pyobject(py).map(|x| x.into_any()),
            Device::Mps => "mps".into_pyobject(py).map(|x| x.into_any()),
            Device::Npu(n) => format!("npu:{n}").into_pyobject(py).map(|x| x.into_any()),
            Device::Xpu(n) => format!("xpu:{n}").into_pyobject(py).map(|x| x.into_any()),
            Device::Xla(n) => format!("xla:{n}").into_pyobject(py).map(|x| x.into_any()),
            Device::Mlu(n) => format!("mlu:{n}").into_pyobject(py).map(|x| x.into_any()),
            Device::Hpu(n) => format!("hpu:{n}").into_pyobject(py).map(|x| x.into_any()),
            Device::Anonymous(n) => n.into_pyobject(py).map(|x| x.into_any()),
        }
    }
}

enum Storage {
    Mmap(Mmap),
    /// Torch specific mmap
    /// This allows us to not manage it
    /// so Pytorch can handle the whole lifecycle.
    /// https://pytorch.org/docs/stable/storage.html#torch.TypedStorage.from_file.
    Torch(OnceLock<Py<PyAny>>),
    // Paddle specific mmap
    // This allows us to not manage the lifecycle of the storage,
    // Paddle can handle the whole lifecycle.
    // https://www.paddlepaddle.org.cn/documentation/docs/en/develop/api/paddle/MmapStorage_en.html
    Paddle(OnceLock<Py<PyAny>>),
    /// Holds an open file handle and
    /// serves each tensor via `pread(2)` into a fresh per-tensor host
    /// buffer, with framework/device-specific buffer choices for performance.
    Pread(Arc<File>),
}

#[derive(Debug, PartialEq, Eq, PartialOrd)]
struct Version {
    major: u8,
    minor: u8,
    patch: u8,
}

impl Version {
    fn new(major: u8, minor: u8, patch: u8) -> Self {
        Self {
            major,
            minor,
            patch,
        }
    }

    fn from_string(string: &str) -> Result<Self, String> {
        let mut parts = string.split('.');
        let err = || format!("Could not parse torch package version {string}.");
        let major_str = parts.next().ok_or_else(err)?;
        let minor_str = parts.next().ok_or_else(err)?;
        let patch_str = parts.next().ok_or_else(err)?;
        // Patch is more complex and can be:
        // - `1` a number
        // - `1a0`, `1b0`, `1rc1` an alpha, beta, release candidate version
        // - `1a0+git2323` from source with commit number
        let patch_str: String = patch_str
            .chars()
            .take_while(|c| c.is_ascii_digit())
            .collect();

        let major = major_str.parse().map_err(|_| err())?;
        let minor = minor_str.parse().map_err(|_| err())?;
        let patch = patch_str.parse().map_err(|_| err())?;
        Ok(Version {
            major,
            minor,
            patch,
        })
    }
}

/// What one span is handed out as: its tensor's name and torch dtype, and the
/// shape torch allocates for its planned part.
struct SpanMeta {
    name: String,
    dtype: Py<PyAny>,
    storage_shape: Vec<usize>,
    /// the dtype in the file, for a tensor made from a Metal buffer
    #[cfg(all(target_os = "macos", target_arch = "aarch64"))]
    file_dtype: Dtype,
}

/// Where a loader's tensors live
#[derive(Clone, Copy)]
enum Place {
    Cuda(i32),
    #[cfg(all(target_os = "macos", target_arch = "aarch64"))]
    Mps,
}

impl Place {
    /// The CUDA device: its tensors are allocated and handed out on streams.
    fn cuda(self) -> Option<i32> {
        match self {
            Self::Cuda(device) => Some(device),
            #[cfg(all(target_os = "macos", target_arch = "aarch64"))]
            Self::Mps => None,
        }
    }
}

impl fmt::Display for Place {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Cuda(device) => write!(f, "cuda:{device}"),
            #[cfg(all(target_os = "macos", target_arch = "aarch64"))]
            Self::Mps => f.write_str("mps"),
        }
    }
}

/// The tensors of one file loading to a CUDA or MPS device in the background,
/// see [`safe_open::prefetch`]. Each tensor is handed out once, by [`Self::take`]
/// or by iterating; tensors not taken are released at `close`.
///
/// On CUDA, tensors are allocated by torch on the stream current at `prefetch`,
/// so torch accounts for them and frees each one when dropped. They are ready on
/// any stream when handed out. Their memory is kept for the stream current at
/// hand-out too; reading one on yet another stream needs `record_stream`, as for
/// any torch tensor.
///
/// On MPS, each tensor wraps a Metal buffer allocated here, as `get_tensor` does.
#[pyclass(frozen)]
pub struct PrefetchLoader {
    loader: Loader<Py<PyAny>>,
    /// per span, aligned with the spans given to `loader`
    names: Vec<String>,
    /// tensor name -> span index
    index_map: HashMap<String, usize>,
    place: Place,
    /// the stream torch allocated the tensors on, on CUDA
    alloc_stream: u64,
    background: Arc<Background>,
}

/// Reader threads of the loader `get_tensors` loads to a CUDA device with
const GET_TENSORS_THREADS: usize = 16;

/// Reader threads of `get_tensors` on the CPU with the `pread` backend
#[cfg(all(target_os = "linux", target_endian = "little"))]
const CPU_READ_THREADS: usize = 16;

/// Bytes of the plan torch allocates ahead of the read schedule
const ALLOC_AHEAD: usize = 1 << 30;

/// At most this many tensors, and about this many bytes, per batch: the workers
/// copy one batch while torch allocates the next, and the GIL is held per batch
const BATCH_TENSORS: usize = 256;
const BATCH_BYTES: usize = 16 << 20;

const ALLOCATOR_THREAD: &str = "safetensors-prefetch-alloc";

/// Largest block warmed in torch's cache ahead of the tensors that split it: one
/// cudaMalloc for all of them, and a tensor kept alive pins no more than this.
const WARM_PIECE: usize = 256 << 20;

/// The largest cached block torch still splits for smaller requests on `device`:
/// [`WARM_PIECE`], less when `max_split_size` is set, 0 when torch reports none.
/// Call once torch's CUDA is initialised: the stats are empty before.
/// The sizes here and in [`Allocation::new`] mirror torch's caching allocator.
fn warm_piece(torch: &PyBound<'_, PyModule>, device: i32) -> usize {
    let max_split = || -> PyResult<i64> {
        let py = torch.py();
        torch
            .getattr(intern!(py, "cuda"))?
            .call_method1(intern!(py, "memory_stats_as_nested_dict"), (device,))?
            .get_item(intern!(py, "max_split_size"))?
            .extract()
    };
    match max_split() {
        Ok(n) if n < 0 => WARM_PIECE, // unlimited
        // blocks of max_split_size or more are never split, and torch rounds segments up to 2 MiB
        Ok(n) => WARM_PIECE.min((n as usize).saturating_sub(2 << 20)),
        Err(_) => 0,
    }
}

/// The CUDA stream current on `device` in this thread, with its raw handle.
fn current_stream<'py>(py: Python<'py>, device: i32) -> PyResult<(PyBound<'py, PyAny>, u64)> {
    let stream = get_module(py, &TORCH_MODULE)?
        .getattr(intern!(py, "cuda"))?
        .call_method1(intern!(py, "current_stream"), (device,))?;
    let raw = stream.getattr(intern!(py, "cuda_stream"))?.extract()?;
    Ok((stream, raw))
}

/// [`current_stream`] where `place` has streams.
fn place_stream<'py>(
    py: Python<'py>,
    place: Place,
) -> PyResult<Option<(PyBound<'py, PyAny>, u64)>> {
    place
        .cuda()
        .map(|device| current_stream(py, device))
        .transpose()
}

/// A tensor torch could not allocate: `reason` names it, `cause` is torch's error.
struct AllocFailure {
    reason: String,
    cause: PyErr,
}

impl AllocFailure {
    fn into_pyerr(self, py: Python<'_>) -> PyErr {
        let err = SafetensorError::new_err(self.reason);
        err.set_cause(py, Some(self.cause));
        err
    }
}

/// The planned tensors still to allocate, in plan order.
struct Allocation {
    metas: Vec<SpanMeta>,
    sizes: Vec<usize>,
    /// per span, the planned bytes before it; one more entry for the total
    before: Vec<usize>,
    /// per span, the warm-up block allocated and dropped before it, 0 for none
    warm: Vec<usize>,
    /// off after the first warm-up that fails
    warming: bool,
    next: usize,
    place: Place,
}

impl Allocation {
    fn new(metas: Vec<SpanMeta>, sizes: Vec<usize>, piece: usize, place: Place) -> Self {
        let before = std::iter::once(0)
            .chain(sizes.iter().scan(0, |sum, &n| {
                *sum += n;
                Some(*sum)
            }))
            .collect();
        // a piece is a run of tensors over 1 MiB (smaller ones come from torch's small
        // pool and don't end a run) whose rounded sizes sum to at most `piece`
        let split = |n: usize| n > 1 << 20;
        let mut warm = vec![0; sizes.len()];
        let mut warmed = 0; // tensors before `warmed` have their block
        for i in 0..sizes.len() {
            if piece == 0 || i < warmed || !split(sizes[i]) {
                continue;
            }
            let (mut bytes, mut count, mut end) = (0, 0, i);
            for (j, &n) in sizes.iter().enumerate().skip(i) {
                if split(n) {
                    let n = n.next_multiple_of(512);
                    if bytes + n > piece {
                        break;
                    }
                    bytes += n;
                    count += 1;
                }
                end = j + 1;
            }
            warmed = end.max(i + 1);
            if count > 1 {
                warm[i] = bytes;
            }
        }
        Self {
            metas,
            sizes,
            before,
            warm,
            warming: true,
            next: 0,
            place,
        }
    }

    fn done(&self) -> bool {
        self.next == self.sizes.len()
    }

    /// Allocates the next tensors, on the current stream where there is one, until
    /// the first `target` planned bytes have a destination or the batch is full.
    fn batch(
        &mut self,
        py: Python<'_>,
        target: usize,
    ) -> Result<Vec<(u64, Py<PyAny>)>, AllocFailure> {
        let torch = get_module(py, &TORCH_MODULE).map_err(|cause| AllocFailure {
            reason: "torch is not imported".into(),
            cause,
        })?;
        let place = self.place;
        // one device object for the batch: torch parses a device string at every call
        let cuda = place
            .cuda()
            .map(|device| {
                let kwargs = torch
                    .getattr(intern!(py, "device"))?
                    .call1(("cuda", device))
                    .and_then(|device| [(intern!(py, "device"), device)].into_py_dict(py))?;
                Ok((device, kwargs))
            })
            .transpose()
            .map_err(|cause: PyErr| AllocFailure {
                reason: format!("cannot allocate on {place}: {cause}"),
                cause,
            })?;
        let mut dsts = Vec::new();
        let first = self.before[self.next];
        while !self.done()
            && self.before[self.next] < target
            && self.before[self.next] - first < BATCH_BYTES
            && dsts.len() < BATCH_TENSORS
        {
            let i = self.next;
            let (meta, size) = (&self.metas[i], self.sizes[i]);
            let tensor = match &cuda {
                Some((device, kwargs)) => {
                    if self.warming && self.warm[i] > 0 {
                        // best effort: the tensors may still fit without it. Not past the
                        // free memory: torch would flush the caller's cache to retry
                        let free = || -> PyResult<usize> {
                            let cuda = torch.getattr(intern!(py, "cuda"))?;
                            cuda.call_method1(intern!(py, "mem_get_info"), (*device,))?
                                .get_item(0)?
                                .extract()
                        };
                        if free().is_ok_and(|free| free >= self.warm[i]) {
                            self.warming = get_pydtype(torch, Dtype::U8, false)
                                .and_then(|uint8| {
                                    torch_empty_cuda(py, torch, &uint8, &[self.warm[i]], kwargs)
                                })
                                .is_ok();
                        }
                    }
                    (|| -> PyResult<(u64, Py<PyAny>)> {
                        let tensor =
                            torch_empty_cuda(py, torch, &meta.dtype, &meta.storage_shape, kwargs)?;
                        let tensor = tensor.bind(py);
                        let nbytes: usize = tensor.getattr(intern!(py, "nbytes"))?.extract()?;
                        if nbytes != size {
                            return Err(SafetensorError::new_err(format!(
                                "torch allocated {nbytes} bytes for {size} bytes of data"
                            )));
                        }
                        let ptr: u64 = tensor.call_method0(intern!(py, "data_ptr"))?.extract()?;
                        Ok((ptr, tensor.clone().unbind()))
                    })()
                }
                #[cfg(all(target_os = "macos", target_arch = "aarch64"))]
                None => mps_empty(py, meta, size),
                #[cfg(not(all(target_os = "macos", target_arch = "aarch64")))]
                None => unreachable!("every place but mps is a cuda device"),
            };
            match tensor {
                Ok(dst) => dsts.push(dst),
                Err(cause) => {
                    return Err(AllocFailure {
                        reason: format!(
                            "cannot allocate {} ({size} bytes) on {place}: {cause}",
                            meta.name
                        ),
                        cause,
                    })
                }
            }
            self.next += 1;
        }
        Ok(dsts)
    }
}

/// What a loader runs besides its workers: the thread that allocates as the load
/// advances. Shared with that thread and, weakly, with the `atexit` hook.
struct Background {
    stop: AtomicBool,
    thread: Mutex<Option<JoinHandle<()>>>,
    feed: Feed<Py<PyAny>>,
    /// the CUDA stream current at `prefetch`, made current in the allocator thread
    stream: Option<Py<PyAny>>,
    stream_raw: u64,
    /// torch's error, when the allocator thread failed to allocate
    error: OnceLock<PyErr>,
    /// set once the allocator thread ended, after releasing the tensors not taken
    /// on `error`; set from the start when there is no such thread
    exited: Mutex<bool>,
    exit: Condvar,
}

impl Background {
    /// Stops the allocator thread, then the workers, and returns once they exited;
    /// waits without the GIL. The tensors not taken stay with the loader.
    fn halt(&self, py: Python<'_>) {
        self.stop.store(true, Ordering::Release);
        let thread = self.thread.lock().unwrap().take();
        py.detach(|| {
            // wakes the allocator thread
            self.feed.stop();
            match thread {
                // dropped from the allocator thread itself (a collection it triggered):
                // it returns on its own once it sees `stop`
                Some(t) if t.thread().id() == std::thread::current().id() => {}
                Some(t) => {
                    let _ = t.join();
                }
                // another closer joins it
                None => self.wait_exited(),
            }
            self.feed.halt();
        });
    }

    /// Waits until the allocator thread exited, or returns at once on an allocator
    /// thread, which may be this one. Call without the GIL.
    fn wait_exited(&self) {
        if std::thread::current().name() == Some(ALLOCATOR_THREAD) {
            return;
        }
        let mut exited = self.exited.lock().unwrap();
        while !*exited {
            exited = self.exit.wait(exited).unwrap();
        }
    }

    /// `msg` as a `SafetensorError`, caused by the allocation failure if any. That
    /// failure is reported once the tensors not taken are released.
    fn error(&self, py: Python<'_>, e: &LoaderError, msg: String) -> PyErr {
        let err = SafetensorError::new_err(msg);
        if let (true, Some(cause)) = (e.is_failure(), self.error.get()) {
            // the allocator thread releases them before it exits
            py.detach(|| self.wait_exited());
            err.set_cause(py, Some(cause.clone_ref(py)));
        }
        err
    }
}

/// Every loader started in this process, for [`halt_prefetch_loaders`]; `None`
/// once it ran
static LIVE_LOADERS: Mutex<Option<Vec<Weak<Background>>>> = Mutex::new(Some(Vec::new()));
static ATEXIT_HOOK: PyOnceLock<()> = PyOnceLock::new();
static HOOK_PID: AtomicU32 = AtomicU32::new(0);

/// Registered with `atexit`: no allocator thread may reach for the GIL once the
/// interpreter finalizes, so stop them, and the workers, before.
#[pyfunction]
fn halt_prefetch_loaders(py: Python<'_>) {
    // a forked child has none of its parent's threads: nothing to halt
    if std::process::id() != HOOK_PID.load(Ordering::Relaxed) {
        return;
    }
    let live: Vec<_> = LIVE_LOADERS
        .lock()
        .unwrap()
        .take()
        .unwrap_or_default()
        .iter()
        .filter_map(Weak::upgrade)
        .collect();
    for background in live {
        background.halt(py);
    }
}

/// The allocator thread: keeps the destinations [`ALLOC_AHEAD`] bytes ahead of
/// the read schedule, taking the GIL once per batch. It tops up once the lead
/// drops below half of that. A failure ends the load and releases the tensors not
/// taken.
fn allocate_ahead(background: Arc<Background>, mut alloc: Allocation) {
    let bg = &*background;
    let _exited = Exited(bg);
    let (mut entered, mut failed, mut filling) = (false, false, false);
    while !alloc.done() {
        if bg.stop.load(Ordering::Acquire) || bg.feed.stopped() {
            return;
        }
        let scheduled = bg.feed.scheduled();
        let read = alloc.before[scheduled];
        let lead = alloc.before[bg.feed.provided()].saturating_sub(read);
        filling = lead < ALLOC_AHEAD && (filling || lead < ALLOC_AHEAD / 2);
        if !filling {
            bg.feed.wait_scheduled(scheduled);
            continue;
        }
        let target = read + ALLOC_AHEAD;
        let more = Python::attach(|py| {
            if bg.stop.load(Ordering::Acquire) {
                return false;
            }
            if let (false, Some(stream)) = (entered, &bg.stream) {
                // the current stream is per thread
                entered = true;
                let set = get_module(py, &TORCH_MODULE).and_then(|torch| {
                    torch
                        .getattr(intern!(py, "cuda"))?
                        .call_method1(intern!(py, "set_stream"), (stream.bind(py),))
                });
                if let Err(cause) = set {
                    let reason = format!("cannot enter the allocating stream: {cause}");
                    let _ = bg.error.set(cause);
                    bg.feed.fail(reason);
                    failed = true;
                    return false;
                }
            }
            match alloc.batch(py, target) {
                Ok(dsts) => match bg.feed.provide(dsts, bg.stream_raw) {
                    Ok(()) => true,
                    Err((e, dsts)) => {
                        drop(dsts);
                        bg.feed.fail(e.to_string()); // no-op once closed
                        failed = true;
                        false
                    }
                },
                Err(AllocFailure { reason, cause }) => {
                    let _ = bg.error.set(cause);
                    bg.feed.fail(reason);
                    failed = true;
                    false
                }
            }
        });
        if !more {
            break;
        }
    }
    if failed {
        // joined without the GIL, dropped attached so torch frees them now
        let untaken = bg.feed.close();
        Python::attach(|_| drop(untaken));
    }
}

/// Sets [`Background::exited`] however the allocator thread ends, panics included.
struct Exited<'a>(&'a Background);

impl Drop for Exited<'_> {
    fn drop(&mut self) {
        *self.0.exited.lock().unwrap() = true;
        self.0.exit.notify_all();
    }
}

/// Hands `tensor` out on the consumer's current stream, if its device has any,
/// resolved before the take so it can't fail with the tensor in hand: when that
/// isn't the allocating stream, torch must not reuse its memory before that
/// stream is done with it.
fn hand_out(
    py: Python<'_>,
    tensor: Py<PyAny>,
    current: Option<(PyBound<'_, PyAny>, u64)>,
    alloc_stream: u64,
) -> PyResult<Py<PyAny>> {
    match current {
        Some((current, raw)) if raw != alloc_stream => {
            tensor.call_method1(py, intern!(py, "record_stream"), (current,))?;
        }
        _ => {}
    }
    Ok(tensor)
}

#[pymethods]
impl PrefetchLoader {
    /// The names this loader hands out, in file order. (Not `keys`: a loader is
    /// not a mapping, each tensor is taken once, and `dict(loader)` must iterate.)
    pub fn names(&self) -> Vec<String> {
        self.names.clone()
    }

    fn __len__(&self) -> usize {
        self.names.len()
    }

    /// Takes `name`'s tensor, waiting if its bytes have not landed yet.
    ///
    /// Each tensor can be taken once, by [`Self::take`] or by iterating the
    /// loader; asking again raises. A tensor planned as a slice comes back as that slice.
    pub fn take(&self, py: Python<'_>, name: &str) -> PyResult<Py<PyAny>> {
        let &idx = self.index_map.get(name).ok_or_else(|| {
            SafetensorError::new_err(format!("tensor {name} is not in the prefetch plan"))
        })?;
        let current = place_stream(py, self.place)?;
        let taken = py.detach(|| self.loader.take_tensor(idx));
        let tensor = taken.map_err(|e| self.background.error(py, &e, format!("{name}: {e}")))?;
        hand_out(py, tensor, current, self.alloc_stream)
    }

    /// `(name, tensor)` pairs as their bytes land, so the order is unspecified;
    /// tensors already taken are skipped.
    fn __iter__(slf: PyRef<'_, Self>) -> TensorStream {
        let iter = slf.loader.iter();
        let names = slf.names.clone();
        TensorStream {
            iter,
            names,
            place: slf.place,
            alloc_stream: slf.alloc_stream,
            background: slf.background.clone(),
            _keepalive: Some(Py::from(slf).into_any()),
        }
    }

    /// Stops the background load and releases every tensor not taken yet.
    /// Tensors already handed out keep their memory.
    pub fn close(&self, py: Python<'_>) {
        self.background.halt(py);
        // dropped attached, so torch frees them now
        let untaken = py.detach(|| self.loader.close());
        drop(untaken);
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

impl Drop for PrefetchLoader {
    fn drop(&mut self) {
        Python::attach(|py| self.close(py));
    }
}

struct Open {
    metadata: Metadata,
    offset: usize,
    framework: Framework,
    device: Device,
    storage: Arc<Storage>,
    /// The open file, for loaders started with [`Open::prefetch`]
    file: Arc<File>,
}

impl Open {
    fn new(
        filename: PathBuf,
        framework: Framework,
        device: Option<Device>,
        backend: Backend,
    ) -> PyResult<Self> {
        let file = File::open(&filename).map_err(|_| {
            PyFileNotFoundError::new_err(format!(
                "No such file or directory: {}",
                filename.display()
            ))
        })?;
        let file = Arc::new(file);
        let device = device.unwrap_or(Device::Cpu);
        if device != Device::Cpu
            && framework != Framework::Pytorch
            && framework != Framework::Paddle
        {
            return Err(SafetensorError::new_err(format!(
                "Device {device} is not supported for framework {framework}",
            )));
        }

        // SAFETY: Mmap is used to prevent allocating in Rust
        // before making a copy within Python.
        let buffer = unsafe { MmapOptions::new().map_copy_read_only(&*file)? };

        let (n, metadata) = SafeTensors::read_metadata(&buffer).map_err(|e| {
            SafetensorError::new_err(format!("Error while deserializing header: {e}"))
        })?;

        let offset = n + 8;
        Python::attach(|py| -> PyResult<()> {
            match framework {
                Framework::Pytorch => {
                    let module = PyModule::import(py, intern!(py, "torch"))?;
                    TORCH_MODULE.get_or_init_py_attached(py, || module.into())
                }
                Framework::Paddle => {
                    let module = PyModule::import(py, intern!(py, "paddle"))?;
                    PADDLE_MODULE.get_or_init_py_attached(py, || module.into())
                }
                _ => {
                    let module = PyModule::import(py, intern!(py, "numpy"))?;
                    NUMPY_MODULE.get_or_init_py_attached(py, || module.into())
                }
            };

            Ok(())
        })?;

        if backend == Backend::Pread {
            disable_page_cache_macos(&file);
            return Ok(Self {
                metadata,
                offset,
                framework,
                device,
                storage: Arc::new(Storage::Pread(file.clone())),
                file,
            });
        }

        let storage = match &framework {
            Framework::Paddle => Python::attach(|py| -> PyResult<Storage> {
                let paddle = get_module(py, &PADDLE_MODULE)?;
                let version: String = paddle.getattr(intern!(py, "__version__"))?.extract()?;
                let version = Version::from_string(&version).map_err(SafetensorError::new_err)?;

                // todo: version check, only paddle 3.1.1 or develop
                if version >= Version::new(3, 1, 1) || version == Version::new(0, 0, 0) {
                    let py_filename: Py<PyAny> = filename
                        .to_str()
                        .ok_or_else(|| {
                            SafetensorError::new_err(format!(
                                "Path {} is not valid UTF-8",
                                filename.display()
                            ))
                        })?
                        .into_pyobject(py)?
                        .into();
                    let size: Py<PyAny> = buffer.len().into_pyobject(py)?.into();
                    let init_kargs = [
                        (intern!(py, "filename"), py_filename),
                        (intern!(py, "nbytes"), size),
                    ]
                    .into_py_dict(py)?;
                    let storage = paddle
                        .getattr(intern!(py, "MmapStorage"))?
                        .call((), Some(&init_kargs))?
                        .into_pyobject(py)?
                        .into();
                    let gil_storage = OnceLock::new();
                    gil_storage.get_or_init_py_attached(py, || storage);
                    Ok(Storage::Paddle(gil_storage))
                } else {
                    let module = PyModule::import(py, intern!(py, "numpy"))?;
                    NUMPY_MODULE.get_or_init_py_attached(py, || module.into());
                    Ok(Storage::Mmap(buffer))
                }
            })?,
            Framework::Pytorch => Python::attach(|py| -> PyResult<Storage> {
                let module = get_module(py, &TORCH_MODULE)?;

                let version: String = module.getattr(intern!(py, "__version__"))?.extract()?;
                let version = Version::from_string(&version).map_err(SafetensorError::new_err)?;

                // Untyped storage only exists for versions over 1.11.0
                // Same for torch.asarray which is necessary for zero-copy tensor
                if version >= Version::new(1, 11, 0) {
                    // storage = torch.ByteStorage.from_file(filename, shared=False,
                    // size=size).untyped()
                    let py_filename: Py<PyAny> = filename
                        .to_str()
                        .ok_or_else(|| {
                            SafetensorError::new_err(format!(
                                "Path {} is not valid UTF-8",
                                filename.display()
                            ))
                        })?
                        .into_pyobject(py)?
                        .into();
                    let size: Py<PyAny> = buffer.len().into_pyobject(py)?.into();
                    let shared: Py<PyAny> = PyBool::new(py, false).to_owned().into();
                    let (size_name, storage_name) = if version >= Version::new(2, 0, 0) {
                        (intern!(py, "nbytes"), intern!(py, "UntypedStorage"))
                    } else {
                        (intern!(py, "size"), intern!(py, "ByteStorage"))
                    };

                    let kwargs =
                        [(intern!(py, "shared"), shared), (size_name, size)].into_py_dict(py)?;
                    let storage = module
                        .getattr(storage_name)?
                        // .getattr(intern!(py, "from_file"))?
                        .call_method("from_file", (py_filename,), Some(&kwargs))?;

                    let untyped: PyBound<'_, PyAny> = match storage.getattr(intern!(py, "untyped"))
                    {
                        Ok(untyped) => untyped,
                        Err(_) => storage.getattr(intern!(py, "_untyped"))?,
                    };
                    let storage = untyped.call0()?.into_pyobject(py)?.into();
                    let gil_storage = OnceLock::new();
                    gil_storage.get_or_init_py_attached(py, || storage);

                    Ok(Storage::Torch(gil_storage))
                } else {
                    Ok(Storage::Mmap(buffer))
                }
            })?,
            _ => Storage::Mmap(buffer),
        };

        let storage = Arc::new(storage);

        Ok(Self {
            metadata,
            offset,
            framework,
            device,
            storage,
            file,
        })
    }

    /// Return the special non tensor information in the header
    ///
    /// Returns:
    ///     (`Dict[str, str]`):
    ///         The freeform metadata.
    pub fn metadata(&self) -> Option<BTreeMap<String, String>> {
        self.metadata.metadata().clone().map(BTreeMap::from_iter)
    }

    /// Returns the names of the tensors in the file.
    ///
    /// Returns:
    ///     (`List[str]`):
    ///         The name of the tensors contained in that file
    pub fn keys(&self) -> PyResult<Vec<String>> {
        let mut keys: Vec<String> = self.metadata.tensors().keys().cloned().collect();
        keys.sort();
        Ok(keys)
    }

    /// Returns the names of the tensors in the file, ordered by offset.
    ///
    /// Returns:
    ///     (`List[str]`):
    ///         The name of the tensors contained in that file
    pub fn offset_keys(&self) -> PyResult<Vec<String>> {
        Ok(self.metadata.offset_keys())
    }

    /// Start loading tensors to the device in the background; see `safe_open.prefetch`.
    pub fn prefetch(
        &self,
        py: Python<'_>,
        plan: Option<&PyBound<'_, PyDict>>,
        device: Option<Device>,
        threads: usize,
        direct: Option<bool>,
    ) -> PyResult<PrefetchLoader> {
        if self.framework != Framework::Pytorch {
            return Err(SafetensorError::new_err(
                "prefetch is only compatible with framework=\"pt\"",
            ));
        }

        let device = device.unwrap_or_else(|| self.device.clone());
        let place = match device {
            Device::Cuda(index) | Device::Anonymous(index) => Place::Cuda(index as i32),
            #[cfg(all(target_os = "macos", target_arch = "aarch64"))]
            Device::Mps if torch_supports_mps_dlpack() => Place::Mps,
            #[cfg(all(target_os = "macos", target_arch = "aarch64"))]
            Device::Mps => {
                return Err(SafetensorError::new_err(
                    "prefetch loads to mps with torch 2.9 or later",
                ))
            }
            #[cfg(not(all(target_os = "macos", target_arch = "aarch64")))]
            Device::Mps => {
                return Err(SafetensorError::new_err(
                    "prefetch loads to mps on Apple silicon only",
                ))
            }
            other => {
                return Err(SafetensorError::new_err(format!(
                    "prefetch loads to cuda and mps devices, not {other}"
                )))
            }
        };
        let slices = match plan {
            Some(plan) => Some(self.parse_plan(plan)?),
            None => None,
        };

        let torch = get_module(py, &TORCH_MODULE)?;
        let mut spans = Vec::new();
        let mut gathers = Vec::new();
        let mut sizes = Vec::new();
        let mut metas = Vec::new();
        let mut index_map = HashMap::new();
        for name in self.metadata.offset_keys() {
            let indexers = match &slices {
                None => &[][..],
                Some(slices) => match slices.get(&name) {
                    None => continue,
                    Some(indexers) => indexers.as_slice(),
                },
            };

            let info = self.metadata.info(&name).ok_or_else(|| {
                SafetensorError::new_err(format!("File does not contain tensor {name}"))
            })?;

            // before anything is allocated: F6 has no torch dtype, and older torch
            // lacks the newest ones
            let dtype = get_pydtype(torch, info.dtype, false).map_err(|_| {
                SafetensorError::new_err(format!(
                    "prefetch cannot hand out dtype {:?}: torch has no matching dtype",
                    info.dtype
                ))
            })?;
            #[cfg(all(target_os = "macos", target_arch = "aarch64"))]
            if matches!(place, Place::Mps) && !dlpack::torch_mps_compatible(info.dtype) {
                return Err(SafetensorError::new_err(format!(
                    "prefetch cannot hand out dtype {:?} on mps",
                    info.dtype
                )));
            }

            let (start, end) = info.data_offsets;
            let (read, shape, gather) = if indexers.is_empty() {
                (0..end - start, info.shape.clone(), None)
            } else {
                let layout = slice_layout(info.dtype, &info.shape, indexers).map_err(|e| {
                    SafetensorError::new_err(match e {
                        InvalidSlice::SteppedSlice { dim_index } => format!(
                            "prefetch plan slices must have step 1, got a step on \
                             dimension {dim_index} of {name}"
                        ),
                        InvalidSlice::TooFragmented { .. } => format!(
                            "invalid prefetch plan slice for {name}: {e}; load the tensor \
                             whole and index it on the device instead"
                        ),
                        e => format!("invalid prefetch plan slice for {name}: {e}"),
                    })
                })?;
                (layout.read, layout.shape, layout.gather)
            };
            index_map.insert(name.clone(), spans.len());
            spans.push(Span {
                start: start + read.start,
                end: start + read.end,
            });
            sizes.push(gather.as_ref().map_or(read.len(), |g| g.len));
            gathers.push(gather);
            metas.push(SpanMeta {
                name,
                dtype,
                storage_shape: torch_storage_shape(info.dtype, &shape)?,
                #[cfg(all(target_os = "macos", target_arch = "aarch64"))]
                file_dtype: info.dtype,
            });
        }

        if slices.is_some() && spans.is_empty() {
            return Err(SafetensorError::new_err("prefetch plan selects no tensors"));
        }

        // the tensors are allocated as the load advances, on the stream current here
        // where there is one: the first [`ALLOC_AHEAD`] bytes now, a batch at a time so
        // the workers start on the first, the rest from the allocator thread
        let names: Vec<String> = metas.iter().map(|m| m.name.clone()).collect();
        let file = self.file.clone();
        let loader = match place {
            Place::Cuda(device) => {
                Loader::load(file, direct, self.offset, device, threads, spans, gathers)?
            }
            #[cfg(all(target_os = "macos", target_arch = "aarch64"))]
            Place::Mps => Loader::load_host(file, direct, self.offset, threads, spans, gathers)?,
        };
        let started = (|| -> PyResult<_> {
            let (stream, stream_raw, piece) = match place.cuda() {
                Some(device) => {
                    let (stream, raw) = current_stream(py, device)?;
                    let piece = warm_piece(get_module(py, &TORCH_MODULE)?, device);
                    (Some(stream.unbind()), raw, piece)
                }
                None => (None, 0, 0),
            };
            let mut alloc = Allocation::new(metas, sizes, piece, place);
            let feed = loader.feed();
            while !alloc.done() && alloc.before[alloc.next] < ALLOC_AHEAD {
                let dsts = alloc.batch(py, ALLOC_AHEAD).map_err(|f| f.into_pyerr(py))?;
                feed.provide(dsts, stream_raw)
                    .map_err(|(e, _)| PyErr::from(e))?;
            }
            ATEXIT_HOOK.get_or_try_init(py, || -> PyResult<()> {
                HOOK_PID.store(std::process::id(), Ordering::Relaxed);
                let hook = wrap_pyfunction!(halt_prefetch_loaders, py)?;
                PyModule::import(py, intern!(py, "atexit"))?
                    .call_method1(intern!(py, "register"), (hook,))?;
                Ok(())
            })?;
            let background = Arc::new(Background {
                stop: AtomicBool::new(false),
                thread: Mutex::new(None),
                feed,
                stream,
                stream_raw,
                error: OnceLock::new(),
                exited: Mutex::new(alloc.done()),
                exit: Condvar::new(),
            });
            {
                let mut live = LIVE_LOADERS.lock().unwrap();
                // a thread started now would outlive the hook into finalization
                let live = live.as_mut().ok_or_else(|| {
                    SafetensorError::new_err("cannot prefetch once the interpreter exits")
                })?;
                live.retain(|w| w.strong_count() > 0);
                live.push(Arc::downgrade(&background));
            }
            if !alloc.done() {
                let thread = std::thread::Builder::new()
                    .name(ALLOCATOR_THREAD.into())
                    .spawn({
                        let background = background.clone();
                        move || allocate_ahead(background, alloc)
                    })
                    // the hook may already wait on it
                    .inspect_err(|_| drop(Exited(&background)))?;
                *background.thread.lock().unwrap() = Some(thread);
            }
            Ok((background, stream_raw))
        })();
        let (background, alloc_stream) = match started {
            Ok(started) => started,
            Err(err) => {
                let untaken = py.detach(|| loader.close());
                drop(untaken);
                return Err(err);
            }
        };
        Ok(PrefetchLoader {
            loader,
            names,
            index_map,
            place,
            alloc_stream,
            background,
        })
    }

    /// Parses a prefetch plan into `name -> indexers`, a list per leading
    /// dimension; none for a whole tensor.
    fn parse_plan(
        &self,
        plan: &PyBound<'_, PyDict>,
    ) -> PyResult<HashMap<String, Vec<Vec<TensorIndexer>>>> {
        let mut slices = HashMap::with_capacity(plan.len());
        for (key, value) in plan.iter() {
            let name: String = key
                .extract()
                .map_err(|_| SafetensorError::new_err("prefetch plan keys must be tensor names"))?;

            let info = self.metadata.info(&name).ok_or_else(|| {
                SafetensorError::new_err(format!("File does not contain tensor {name}"))
            })?;

            if value.is_none() {
                slices.insert(name, vec![]);
                continue;
            }

            let invalid = |e: PyErr| {
                SafetensorError::new_err(format!(
                    "invalid prefetch plan slice for {name}: {}",
                    e.value(value.py())
                ))
            };
            // an int or a slice along the first dimension of `shape`
            let one = |it: &PyBound<'_, PyAny>, shape: &[usize]| -> PyResult<TensorIndexer> {
                if it.is_instance_of::<PyEllipsis>()
                    || it.is_instance_of::<PyTuple>()
                    || it.is_instance_of::<PyList>()
                {
                    return Err(SafetensorError::new_err(
                        "next to a list, entries must be ints, slices or lists of those",
                    ));
                }
                Ok(parse_indexers(it, shape)?.remove(0))
            };
            let indexers = match value.cast::<PyTuple>() {
                Ok(tuple) if tuple.iter().any(|it| it.is_instance_of::<PyList>()) => {
                    if tuple.len() > info.shape.len() {
                        return Err(SafetensorError::new_err(format!(
                            "invalid prefetch plan slice for {name}: {}",
                            InvalidSlice::TooManySlices
                        )));
                    }
                    tuple
                        .iter()
                        .enumerate()
                        .map(|(dim, item)| match item.cast::<PyList>() {
                            // an int in a list keeps its dimension, as in numpy
                            Ok(list) => list
                                .iter()
                                .map(|it| match one(&it, &info.shape[dim..])? {
                                    TensorIndexer::Select(i) => Ok(TensorIndexer::Narrow(
                                        Bound::Included(i),
                                        Bound::Excluded(i + 1),
                                        NonZeroUsize::MIN,
                                    )),
                                    ix => Ok(ix),
                                })
                                .collect(),
                            Err(_) => Ok(vec![one(&item, &info.shape[dim..])?]),
                        })
                        .collect::<PyResult<Vec<_>>>()
                        .map_err(invalid)?
                }
                _ => parse_indexers(&value, &info.shape)
                    .map_err(invalid)?
                    .into_iter()
                    .map(|ix| vec![ix])
                    .collect(),
            };
            slices.insert(name, indexers);
        }

        Ok(slices)
    }

    /// Returns a full tensor
    ///
    /// Args:
    ///     name (`str`):
    ///         The name of the tensor you want
    ///
    /// Returns:
    ///     (`Tensor`):
    ///         The tensor in the framework you opened the file for.
    ///
    /// Example:
    /// ```python
    /// from safetensors import safe_open
    ///
    /// with safe_open("model.safetensors", framework="pt", device=0) as f:
    ///     tensor = f.get_tensor("embedding")
    /// ```
    pub fn get_tensor(&self, name: &str) -> PyResult<Py<PyAny>> {
        let info = self.metadata.info(name).ok_or_else(|| {
            SafetensorError::new_err(format!("File does not contain tensor {name}",))
        })?;

        // Pytorch + CUDA: write into a pinned CPU tensor and `.to(cuda)` for
        // async DMA, regardless of backend. The byte source differs per
        // Storage variant (mmap region / pread / torch storage's data_ptr),
        // but the destination + transfer step are identical.
        // TODO: investigate the equivalent for Paddle + GPU using
        // `paddle.empty(..., pin_memory=True)` once we have hardware to test.
        if self.framework == Framework::Pytorch {
            if let Device::Cuda(_) = self.device {
                return self.get_tensor_pinned_cuda(name, info);
            }
        }

        #[cfg(all(target_os = "macos", target_arch = "aarch64"))]
        if self.device == Device::Mps
            && self.framework == Framework::Pytorch
            && dlpack::torch_mps_compatible(info.dtype)
            && torch_supports_mps_dlpack()
        {
            return self.get_tensor_mps(name, info);
        }

        match &self.storage.as_ref() {
            Storage::Mmap(mmap) => {
                let data =
                    &mmap[info.data_offsets.0 + self.offset..info.data_offsets.1 + self.offset];

                let array: Py<PyAny> =
                    Python::attach(|py| PyByteArray::new(py, data).into_any().into());

                create_tensor(
                    &self.framework,
                    info.dtype,
                    &info.shape,
                    array,
                    &self.device,
                )
            }
            Storage::Paddle(storage) => {
                Python::attach(|py| -> PyResult<Py<PyAny>> {
                    let paddle = get_module(py, &PADDLE_MODULE)?;
                    let cur_type = if info.dtype == Dtype::U16 {
                        Dtype::BF16
                    } else {
                        info.dtype
                    };
                    let dtype: Py<PyAny> = get_pydtype(paddle, cur_type, false)?;
                    let paddle_uint8: Py<PyAny> = get_pydtype(paddle, Dtype::U8, false)?;
                    let mut shape = info.shape.to_vec();
                    if cur_type == Dtype::F4 {
                        let n = shape.len();
                        if shape[n - 1] % 2 != 0 {
                            return Err(SafetensorError::new_err(format!(
                        "f4_x2 dtype requires that the last dim be divisible by 2 in torch: got {shape:?}",
                                )));
                        }
                        shape[n - 1] /= 2;
                    }
                    let shape: Py<PyAny> = shape.into_pyobject(py)?.into();
                    let start = (info.data_offsets.0 + self.offset) as isize;
                    let stop = (info.data_offsets.1 + self.offset) as isize;

                    let kwargs = [
                        (intern!(py, "dtype"), paddle_uint8),
                        (intern!(py, "start"), start.into_pyobject(py)?.into()),
                        (intern!(py, "stop"), stop.into_pyobject(py)?.into()),
                    ]
                    .into_py_dict(py)?;
                    let sys = PyModule::import(py, intern!(py, "sys"))?;
                    let byteorder: String = sys.getattr(intern!(py, "byteorder"))?.extract()?;
                    let storage: &Py<PyAny> = storage
                        .get()
                        .ok_or_else(|| SafetensorError::new_err("Could not find storage"))?;
                    let storage: &PyBound<PyAny> = storage.bind(py);
                    let storage_slice = storage
                        .getattr(intern!(py, "get_slice"))?
                        .call((), Some(&kwargs))?;
                    let mut tensor = storage_slice
                        .getattr(intern!(py, "view"))?
                        .call1((dtype,))?;

                    if byteorder == "big" {
                        let inplace_kwargs =
                            [(intern!(py, "inplace"), PyBool::new(py, false))].into_py_dict(py)?;

                        let intermediary_dtype = match cur_type {
                            Dtype::BF16 => Some(Dtype::F16),
                            Dtype::F8_E5M2 => Some(Dtype::U8),
                            Dtype::F8_E4M3 => Some(Dtype::U8),
                            Dtype::F8_E8M0 => Some(Dtype::U8),
                            _ => None,
                        };
                        if let Some(intermediary_dtype) = intermediary_dtype {
                            // Reinterpret to f16 for numpy compatibility.
                            let dtype: Py<PyAny> = get_pydtype(paddle, intermediary_dtype, false)?;
                            tensor = tensor.getattr(intern!(py, "view"))?.call1((dtype,))?;
                        }
                        let numpy = tensor
                            .getattr(intern!(py, "numpy"))?
                            .call0()?
                            .getattr("byteswap")?
                            .call((), Some(&inplace_kwargs))?;
                        tensor = paddle.getattr(intern!(py, "to_tensor"))?.call1((numpy,))?;
                        if intermediary_dtype.is_some() {
                            // Reinterpret to f16 for numpy compatibility.
                            let dtype: Py<PyAny> = get_pydtype(paddle, cur_type, false)?;
                            tensor = tensor.getattr(intern!(py, "view"))?.call1((dtype,))?;
                        }
                    }

                    if self.device != Device::Cpu {
                        let device: Py<PyAny> = if let Device::Cuda(index) = self.device {
                            format!("gpu:{index}").into_pyobject(py)?.into()
                        } else {
                            self.device.clone().into_pyobject(py)?.into()
                        };
                        let kwargs = PyDict::new(py);
                        tensor = tensor.call_method("to", (device,), Some(&kwargs))?;
                    }

                    let tensor = tensor.getattr(intern!(py, "reshape"))?.call1((shape,))?;
                    // Paddle's MmapStorage.get_slice() doesn't keep the storage alive,
                    // so we attach it to the tensor to prevent it from being garbage collected
                    tensor.setattr(intern!(py, "_safetensors_storage"), storage)?;
                    Ok(tensor.into_pyobject(py)?.into())
                })
            }
            Storage::Torch(storage) => {
                Python::attach(|py| -> PyResult<Py<PyAny>> {
                    let torch = get_module(py, &TORCH_MODULE)?;
                    let dtype: Py<PyAny> = get_pydtype(torch, info.dtype, false)?;
                    let torch_uint8: Py<PyAny> = get_pydtype(torch, Dtype::U8, false)?;
                    let device: Py<PyAny> = self.device.clone().into_pyobject(py)?.into();
                    let kwargs = [
                        (intern!(py, "dtype"), torch_uint8),
                        (intern!(py, "device"), device),
                    ]
                    .into_py_dict(py)?;
                    let view_kwargs = [(intern!(py, "dtype"), dtype)].into_py_dict(py)?;
                    let mut shape = info.shape.to_vec();
                    if info.dtype == Dtype::F4 {
                        let n = shape.len();
                        if shape[n - 1] % 2 != 0 {
                            return Err(SafetensorError::new_err(format!(
                    "f4_x2 dtype requires that the last dim be divisible by 2 in torch: got {shape:?}",
                )));
                        }
                        shape[n - 1] /= 2;
                    }
                    let shape: Py<PyAny> = shape.into_pyobject(py)?.into();

                    let start = (info.data_offsets.0 + self.offset) as isize;
                    let stop = (info.data_offsets.1 + self.offset) as isize;
                    let slice = PySlice::new(py, start, stop, 1);
                    let storage: &Py<PyAny> = storage
                        .get()
                        .ok_or_else(|| SafetensorError::new_err("Could not find storage"))?;
                    let storage: &PyBound<PyAny> = storage.bind(py);
                    let storage_slice = storage
                        .getattr(intern!(py, "__getitem__"))?
                        .call1((slice,))?;

                    let sys = PyModule::import(py, intern!(py, "sys"))?;
                    let byteorder: String = sys.getattr(intern!(py, "byteorder"))?.extract()?;

                    let mut tensor = torch
                        .getattr(intern!(py, "asarray"))?
                        .call((storage_slice,), Some(&kwargs))?
                        .getattr(intern!(py, "view"))?
                        .call((), Some(&view_kwargs))?;

                    if byteorder == "big" {
                        let inplace_kwargs =
                            [(intern!(py, "inplace"), PyBool::new(py, false))].into_py_dict(py)?;

                        let intermediary_dtype = match info.dtype {
                            Dtype::BF16 => Some(Dtype::F16),
                            Dtype::F8_E5M2 => Some(Dtype::U8),
                            Dtype::F8_E4M3 => Some(Dtype::U8),
                            Dtype::F8_E8M0 => Some(Dtype::U8),
                            _ => None,
                        };
                        if let Some(intermediary_dtype) = intermediary_dtype {
                            // Reinterpret to f16 for numpy compatibility.
                            let dtype: Py<PyAny> = get_pydtype(torch, intermediary_dtype, false)?;
                            let view_kwargs = [(intern!(py, "dtype"), dtype)].into_py_dict(py)?;
                            tensor = tensor
                                .getattr(intern!(py, "view"))?
                                .call((), Some(&view_kwargs))?;
                        }
                        let numpy = tensor
                            .getattr(intern!(py, "numpy"))?
                            .call0()?
                            .getattr("byteswap")?
                            .call((), Some(&inplace_kwargs))?;
                        tensor = torch.getattr(intern!(py, "from_numpy"))?.call1((numpy,))?;
                        if intermediary_dtype.is_some() {
                            // Reinterpret to f16 for numpy compatibility.
                            let dtype: Py<PyAny> = get_pydtype(torch, info.dtype, false)?;
                            let view_kwargs = [(intern!(py, "dtype"), dtype)].into_py_dict(py)?;
                            tensor = tensor
                                .getattr(intern!(py, "view"))?
                                .call((), Some(&view_kwargs))?;
                        }
                    }

                    tensor = tensor.getattr(intern!(py, "reshape"))?.call1((shape,))?;
                    Ok(tensor.into_pyobject(py)?.into())
                })
            }
            Storage::Pread(file) => self.get_tensor_pread(name, info, file),
        }
    }

    fn get_tensor_pread(
        &self,
        name: &str,
        info: &TensorInfo,
        file: &Arc<File>,
    ) -> PyResult<Py<PyAny>> {
        let (begin, end) = info.data_offsets;
        let nbytes = end - begin;
        let file_offset = (self.offset + begin) as u64;

        let array: Py<PyAny> = Python::attach(|py| -> PyResult<Py<PyAny>> {
            let pyarray = PyByteArray::new_with(py, nbytes, |dst| {
                if !dst.is_empty() {
                    read_exact_at(file, dst, file_offset).map_err(|e| {
                        SafetensorError::new_err(format!(
                            "Could not read tensor {name} from file: {e}"
                        ))
                    })?;
                }
                Ok(())
            })?;
            Ok(pyarray.into_any().into())
        })?;
        create_tensor(
            &self.framework,
            info.dtype,
            &info.shape,
            array,
            &self.device,
        )
    }

    /// Pytorch + CUDA fast path used regardless of backend: allocate a
    /// `pin_memory=True` CPU tensor, fill it from whichever source the
    /// `Storage` variant exposes, then `.to(cuda)` for async DMA. Avoids
    /// CUDA's internal bounce buffer that pageable sources incur, and the
    /// pinned host buffer is dropped right after the transfer so peak host
    /// residency stays at one tensor.
    fn get_tensor_pinned_cuda(&self, name: &str, info: &TensorInfo) -> PyResult<Py<PyAny>> {
        let (begin, end) = info.data_offsets;
        let nbytes = end - begin;

        Python::attach(|py| -> PyResult<Py<PyAny>> {
            let torch = get_module(py, &TORCH_MODULE)?;
            let dest = PinnedCpuDest::new(py, torch, info.dtype, &info.shape, nbytes)?;

            if nbytes > 0 {
                let write_ptr = dest.write_ptr;
                match self.storage.as_ref() {
                    Storage::Mmap(mmap) => {
                        let src_off = self.offset + begin;
                        let src = &mmap[src_off..src_off + nbytes];
                        py.detach(|| {
                            // SAFETY: write_ptr/nbytes name `dest.tensor`'s
                            // pinned storage; src is the live mmap region.
                            let dst = unsafe {
                                std::slice::from_raw_parts_mut(write_ptr as *mut u8, nbytes)
                            };
                            dst.copy_from_slice(src);
                        });
                    }
                    Storage::Pread(file) => {
                        let file_offset = (self.offset + begin) as u64;
                        let read_result: std::io::Result<()> = py.detach(|| {
                            // SAFETY: write_ptr/nbytes name `dest.tensor`'s
                            // pinned storage.
                            let buf = unsafe {
                                std::slice::from_raw_parts_mut(write_ptr as *mut u8, nbytes)
                            };
                            read_exact_at(file, buf, file_offset)
                        });
                        read_result.map_err(|e| {
                            SafetensorError::new_err(format!("pread failed for tensor {name}: {e}"))
                        })?;
                    }
                    Storage::Torch(storage) => {
                        let storage_obj: &Py<PyAny> = storage
                            .get()
                            .ok_or_else(|| SafetensorError::new_err("Could not find storage"))?;
                        let storage_obj = storage_obj.bind(py);
                        let src_data_ptr: usize = storage_obj
                            .call_method0(intern!(py, "data_ptr"))?
                            .extract()?;
                        let src_addr = src_data_ptr + self.offset + begin;
                        py.detach(|| {
                            // SAFETY: src_addr is the data_ptr of the torch
                            // UntypedStorage held alive by `self.storage`,
                            // valid for `self.offset+begin..+nbytes`.
                            // write_ptr/nbytes name `dest.tensor`'s pinned
                            // storage.
                            unsafe {
                                std::ptr::copy_nonoverlapping(
                                    src_addr as *const u8,
                                    write_ptr as *mut u8,
                                    nbytes,
                                );
                            }
                        });
                    }
                    Storage::Paddle(_) => {
                        // Paddle has its own CUDA path via Storage::Paddle.
                        unreachable!("Storage::Paddle does not route through pinned CUDA path");
                    }
                }
            }

            let cuda_device: Py<PyAny> = self.device.clone().into_pyobject(py)?.into();
            let kwargs = PyDict::new(py);
            let cuda_tensor = dest
                .tensor
                .call_method("to", (cuda_device,), Some(&kwargs))?;
            Ok(cuda_tensor.into_pyobject(py)?.into())
        })
    }

    /// Single-tensor MPS path: alloc a Shared `MTLBuffer`, fill it from the
    /// active storage source, and hand off via DLPack.
    #[cfg(all(target_os = "macos", target_arch = "aarch64"))]
    fn get_tensor_mps(&self, name: &str, info: &TensorInfo) -> PyResult<Py<PyAny>> {
        let (begin, end) = info.data_offsets;
        let nbytes = end - begin;

        let mut buf = crate::metal::MTLBuffer::alloc_shared(nbytes)
            .map_err(|e| SafetensorError::new_err(format!("MTLBuffer alloc for {name}: {e}")))?;
        if nbytes > 0 {
            let dst = buf.as_mut_slice();
            match self.storage.as_ref() {
                Storage::Mmap(mmap) => {
                    dst.copy_from_slice(&mmap[self.offset + begin..self.offset + end]);
                }
                Storage::Pread(file) => {
                    read_exact_at(file, dst, (self.offset + begin) as u64)
                        .map_err(|e| SafetensorError::new_err(format!("pread for {name}: {e}")))?;
                }
                Storage::Torch(storage) => Python::attach(|py| -> PyResult<()> {
                    let storage_obj = storage
                        .get()
                        .ok_or_else(|| SafetensorError::new_err("Could not find storage"))?
                        .bind(py);
                    let src_data_ptr: usize = storage_obj
                        .call_method0(intern!(py, "data_ptr"))?
                        .extract()?;
                    let src_addr = src_data_ptr + self.offset + begin;
                    // SAFETY: src_addr..+nbytes spans the torch UntypedStorage
                    // held alive by `self.storage` for this call.
                    let src = unsafe { std::slice::from_raw_parts(src_addr as *const u8, nbytes) };
                    dst.copy_from_slice(src);
                    Ok(())
                })?,
                Storage::Paddle(_) => {
                    return Err(SafetensorError::new_err(
                        "Paddle + MPS is not a supported combination",
                    ));
                }
            }
        }

        Python::attach(|py| mps_tensor_from_buf(py, &self.framework, info.dtype, &info.shape, buf))
    }

    /// Returns every tensor in the file as a `{name: Tensor}` dict.
    ///
    /// Default behavior is a sequential loop over `get_tensor`. Pytorch loads
    /// through [`Open::prefetch`] on a CUDA device when a CUDA runtime is loaded,
    /// and on MPS when torch takes Metal buffers and the file's dtypes. Pytorch
    /// on the CPU with the `pread` backend on Linux reads the file in parallel,
    /// see [`Self::get_tensors_pread_cpu`].
    pub fn get_tensors(&self) -> PyResult<Py<PyDict>> {
        #[cfg(all(target_os = "linux", target_endian = "little"))]
        if self.framework == Framework::Pytorch && self.device == Device::Cpu {
            if let Storage::Pread(file) = self.storage.as_ref() {
                if let Some(tensors) = self.get_tensors_pread_cpu(file)? {
                    return Ok(tensors);
                }
            }
        }

        let prefetched = match self.device {
            Device::Cuda(_) => crate::engine::available(),
            #[cfg(all(target_os = "macos", target_arch = "aarch64"))]
            Device::Mps => {
                torch_supports_mps_dlpack()
                    && self
                        .metadata
                        .tensors()
                        .values()
                        .all(|info| dlpack::torch_mps_compatible(info.dtype))
            }
            _ => false,
        };
        if self.framework == Framework::Pytorch && prefetched {
            return Python::attach(|py| -> PyResult<Py<PyDict>> {
                let loader = self.prefetch(py, None, None, GET_TENSORS_THREADS, None)?;
                let dict = PyDict::new(py);
                for name in &loader.names {
                    dict.set_item(name, loader.take(py, name)?)?;
                }
                Ok(dict.into())
            });
        }

        Python::attach(|py| -> PyResult<Py<PyDict>> {
            let dict = PyDict::new(py);
            for name in self.metadata.offset_keys() {
                let tensor = self.get_tensor(&name)?;
                dict.set_item(&name, tensor)?;
            }
            Ok(dict.into())
        })
    }

    /// Allocates every tensor with torch, then reads the file into them in
    /// parallel. `None` when a dtype needs the per-tensor path: `F4`, or one torch
    /// lacks.
    #[cfg(all(target_os = "linux", target_endian = "little"))]
    fn get_tensors_pread_cpu(&self, file: &Arc<File>) -> PyResult<Option<Py<PyDict>>> {
        Python::attach(|py| {
            let torch = get_module(py, &TORCH_MODULE)?;
            let mut infos = Vec::new();
            for name in self.metadata.offset_keys() {
                let info = self.metadata.info(&name).ok_or_else(|| {
                    SafetensorError::new_err(format!("File does not contain tensor {name}"))
                })?;
                match get_pydtype(torch, info.dtype, false) {
                    Ok(dtype) if info.dtype != Dtype::F4 => infos.push((name, info, dtype)),
                    _ => return Ok(None),
                }
            }
            let dict = PyDict::new(py);
            let mut tensors = Vec::with_capacity(infos.len());
            for (name, info, dtype) in infos {
                let kwargs = [(intern!(py, "dtype"), dtype)].into_py_dict(py)?;
                let shape = info.shape.clone();
                let tensor = torch.call_method(intern!(py, "empty"), (shape,), Some(&kwargs))?;
                let (start, end) = info.data_offsets;
                let ptr: u64 = tensor.call_method0(intern!(py, "data_ptr"))?.extract()?;
                tensors.push((start..end, ptr));
                dict.set_item(name, tensor)?;
            }
            let file = file.clone();
            // SAFETY: each address is a tensor just allocated with its range's
            // bytes, which `dict` keeps alive and only this call accesses
            let read = py.detach(|| unsafe {
                crate::engine::read(file, self.offset, &tensors, CPU_READ_THREADS)
            });
            read.map_err(|e| {
                SafetensorError::new_err(format!("Could not read tensors from file: {e}"))
            })?;
            Ok(Some(dict.into()))
        })
    }

    /// Returns a full slice view object
    ///
    /// Args:
    ///     name (`str`):
    ///         The name of the tensor you want
    ///
    /// Returns:
    ///     (`PySafeSlice`):
    ///         A dummy object you can slice into to get a real tensor
    /// Example:
    /// ```python
    /// from safetensors import safe_open
    ///
    /// with safe_open("model.safetensors", framework="pt", device=0) as f:
    ///     tensor_part = f.get_slice("embedding")[:, ::8]
    /// ```
    pub fn get_slice(&self, name: &str) -> PyResult<PySafeSlice> {
        if let Some(info) = self.metadata.info(name) {
            Ok(PySafeSlice {
                info: info.clone(),
                framework: self.framework.clone(),
                offset: self.offset,
                device: self.device.clone(),
                storage: self.storage.clone(),
            })
        } else {
            Err(SafetensorError::new_err(format!(
                "File does not contain tensor {name}",
            )))
        }
    }

    /// A tensor's header entry.
    pub fn get_tensor_meta(&self, name: &str) -> PyResult<TensorMeta> {
        let info = self.metadata.info(name).ok_or_else(|| {
            SafetensorError::new_err(format!("File does not contain tensor {name}"))
        })?;
        Ok(TensorMeta {
            dtype: info.dtype.to_string(),
            shape: info.shape.clone(),
            data_offsets: info.data_offsets,
        })
    }
}

impl From<LoaderError> for PyErr {
    fn from(value: LoaderError) -> Self {
        SafetensorError::new_err(value.to_string())
    }
}

/// Iterator over a [`PrefetchLoader`]: `(name, tensor)` pairs in readiness
/// order, each tensor delivered once.
#[pyclass]
pub struct TensorStream {
    iter: TensorIter<Py<PyAny>>,
    /// per span, aligned with the loader's spans
    names: Vec<String>,
    place: Place,
    alloc_stream: u64,
    background: Arc<Background>,
    _keepalive: Option<Py<PyAny>>,
}

#[pymethods]
impl TensorStream {
    fn __iter__(slf: PyRef<'_, Self>) -> PyRef<'_, Self> {
        slf
    }

    fn __next__(&mut self, py: Python<'_>) -> PyResult<Option<(String, Py<PyAny>)>> {
        let current = place_stream(py, self.place)?;
        match py.detach(|| self.iter.next()) {
            None => Ok(None),
            Some(Err(e)) => Err(self.background.error(py, &e, e.to_string())),
            Some(Ok((idx, tensor))) => {
                let tensor = hand_out(py, tensor, current, self.alloc_stream)?;
                Ok(Some((std::mem::take(&mut self.names[idx]), tensor)))
            }
        }
    }
}

/// Opens a safetensors lazily and returns tensors as asked
///
/// Args:
///     filename (`str`, or `os.PathLike`):
///         The filename to open
///
///     framework (`str`):
///         The framework you want you tensors in. Supported values:
///         `pt`, `tf`, `flax`, `numpy`.
///
///     device (`str`, defaults to `"cpu"`):
///         The device on which you want the tensors.
///
///     backend (`str`, *keyword-only*, defaults to `"mmap"`):
///         Storage backend used to serve tensor bytes. `"mmap"` (the default)
///         memory-maps the file; `"pread"` reads tensor bytes with `pread(2)`.
///         On Apple-silicon MPS, `get_tensor` with `"pread"` reads straight into
///         the tensor's Metal buffer.
#[pyclass(frozen)]
#[allow(non_camel_case_types)]
struct safe_open {
    /// `None` once closed. Each call works on its own reference, so closing never
    /// waits for one: the file closes when the last call returns.
    inner: Mutex<Option<Arc<Open>>>,
}

impl safe_open {
    fn inner(&self) -> PyResult<Arc<Open>> {
        let inner = self.inner.lock().unwrap().clone();
        inner.ok_or_else(|| SafetensorError::new_err("File is closed".to_string()))
    }
}

#[pymethods]
impl safe_open {
    #[new]
    #[pyo3(signature = (filename, framework, device=Some(Device::Cpu), *, backend=Backend::Mmap))]
    fn new(
        filename: PathBuf,
        framework: Framework,
        device: Option<Device>,
        backend: Backend,
    ) -> PyResult<Self> {
        let open = Open::new(filename, framework, device, backend)?;
        Ok(Self {
            inner: Mutex::new(Some(Arc::new(open))),
        })
    }

    /// Start loading the file's tensors to a CUDA device, or to MPS on Apple
    /// silicon, in the background.
    ///
    /// Returns a [`PrefetchLoader`] that hands each tensor out once, by
    /// [`PrefetchLoader::take`] or by iterating it as `(name, tensor)` pairs in
    /// readiness order, as tensors allocated as the load advances, at most
    /// about 1 GiB ahead of the reads. The handle itself
    /// is unchanged: its header queries and [`Self::get_slice`] keep working, and several loaders can be
    /// started from one handle (for instance one per device). Requires
    /// `framework="pt"`; works with either backend. Each loader runs `threads`
    /// reader threads; all CUDA loaders in the process share one pinned staging
    /// pool.
    ///
    /// On CUDA, torch allocates the tensors on the stream current at the call, and their
    /// copies wait for the work queued on that stream when they are allocated. To load
    /// independently of that work, call `prefetch` under a side stream
    /// (`with torch.cuda.stream(side):`); torch then keeps a separate cache for it.
    ///
    /// On MPS, which needs torch 2.9 or later, each tensor wraps a Metal buffer
    /// allocated by safetensors, as with `get_tensor`, and reads go through the
    /// page cache.
    ///
    /// Args:
    ///     plan (`Dict[str, Optional[Any]]`, *optional*):
    ///         Which tensors to load, and which part of each: keys are tensor
    ///         names, values `None` for the whole tensor or any index `get_slice`
    ///         takes (an int, a slice, `...`, or a tuple of those). In a tuple, a
    ///         dimension's entry can also be a list of ints and slices, kept in
    ///         order along that dimension (e.g. the two halves of a fused
    ///         projection); an int in a list keeps its dimension, as in numpy.
    ///         Tensors absent from the plan are not loaded by this loader.
    ///         `None` (the default) loads every tensor whole.
    ///     device (`str` or `int`, *optional*):
    ///         The CUDA device to load to, or `"mps"`; defaults to the handle's
    ///         `device`.
    ///     threads (`int`, defaults to 16):
    ///         Reader threads for this loader.
    ///     direct (`bool`, *optional*):
    ///         Whether to read the file directly from the disk, bypassing the page
    ///         cache. Direct reads are faster on fast disks but leave the file
    ///         uncached for later loads. `None` (the default) reads the parts already
    ///         cached from the cache and the rest directly; `True` reads everything
    ///         directly; `False` reads everything through the cache. Filesystems that
    ///         do not support direct reads always go through the cache. On macOS,
    ///         `None` reads through the cache too, and `True` still takes from the
    ///         cache what it holds.
    ///
    /// Raises if a planned tensor has a dtype the installed torch cannot represent
    /// (F6, F8_E8M0 before torch 2.7, F4 before 2.8), if a slice has a step other
    /// than 1, if the entries of a dimension overlap or are out of order, if a part
    /// is too fragmented to load, or if torch can't allocate the first 1 GiB.
    /// A later allocation failure ends the load: `take` and iteration raise, and the
    /// tensors not taken are released at once; after any other failure they are
    /// released at `close`.
    ///
    /// Example:
    /// ```python
    /// from safetensors import safe_open
    ///
    /// with safe_open("model.safetensors", framework="pt") as f:
    ///     with f.prefetch(device="cuda:0") as loader:
    ///         for name, tensor in loader:
    ///             ...
    /// ```
    #[pyo3(signature = (plan=None, *, device=None, threads=16, direct=None))]
    pub fn prefetch(
        &self,
        py: Python<'_>,
        plan: Option<&PyBound<'_, PyDict>>,
        device: Option<Device>,
        threads: usize,
        direct: Option<bool>,
    ) -> PyResult<PrefetchLoader> {
        self.inner()?.prefetch(py, plan, device, threads, direct)
    }

    /// Return the special non tensor information in the header
    ///
    /// Returns:
    ///     (`Dict[str, str]`):
    ///         The freeform metadata.
    pub fn metadata(&self) -> PyResult<Option<BTreeMap<String, String>>> {
        Ok(self.inner()?.metadata())
    }

    /// Returns the names of the tensors in the file.
    ///
    /// Returns:
    ///     (`List[str]`):
    ///         The name of the tensors contained in that file
    pub fn keys(&self) -> PyResult<Vec<String>> {
        self.inner()?.keys()
    }

    /// Returns a tensor's header entry without reading any data.
    ///
    /// Args:
    ///     name (`str`):
    ///         The name of the tensor
    ///
    /// Returns:
    ///     (`TensorMeta`):
    ///         `dtype` (safetensors name, e.g. `"F32"`), `shape` and `data_offsets`
    ///         as written in the file header.
    pub fn get_tensor_meta(&self, name: &str) -> PyResult<TensorMeta> {
        self.inner()?.get_tensor_meta(name)
    }

    /// Returns the names of the tensors in the file, ordered by offset.
    ///
    /// Returns:
    ///     (`List[str]`):
    ///         The name of the tensors contained in that file
    pub fn offset_keys(&self) -> PyResult<Vec<String>> {
        self.inner()?.offset_keys()
    }

    /// Returns a full tensor
    ///
    /// Args:
    ///     name (`str`):
    ///         The name of the tensor you want
    ///
    /// Returns:
    ///     (`Tensor`):
    ///         The tensor in the framework you opened the file for.
    ///
    /// Example:
    /// ```python
    /// from safetensors import safe_open
    ///
    /// with safe_open("model.safetensors", framework="pt", device=0) as f:
    ///     tensor = f.get_tensor("embedding")
    /// ```
    pub fn get_tensor(&self, name: &str) -> PyResult<Py<PyAny>> {
        self.inner()?.get_tensor(name)
    }

    /// Returns every tensor in the file as a dict keyed by name.
    ///
    /// Equivalent to iterating `offset_keys()` and calling `get_tensor` on
    /// each. On a CUDA device with PyTorch, and on Apple-silicon MPS with torch
    /// 2.9 or later, it loads through `prefetch`. On the CPU with PyTorch
    /// and the `"pread"` backend, on Linux, it reads the file in parallel.
    ///
    /// Returns:
    ///     (`Dict[str, Tensor]`):
    ///         A dict of all tensors in the file.
    ///
    /// Example:
    /// ```python
    /// from safetensors import safe_open
    ///
    /// with safe_open("model.safetensors", framework="pt", device="mps") as f:
    ///     state_dict = f.get_tensors()
    /// ```
    pub fn get_tensors(&self) -> PyResult<Py<PyDict>> {
        self.inner()?.get_tensors()
    }

    /// Returns a full slice view object
    ///
    /// Args:
    ///     name (`str`):
    ///         The name of the tensor you want
    ///
    /// Returns:
    ///     (`PySafeSlice`):
    ///         A dummy object you can slice into to get a real tensor
    /// Example:
    /// ```python
    /// from safetensors import safe_open
    ///
    /// with safe_open("model.safetensors", framework="pt", device=0) as f:
    ///     tensor_part = f.get_slice("embedding")[:, ::8]
    /// ```
    pub fn get_slice(&self, name: &str) -> PyResult<PySafeSlice> {
        self.inner()?.get_slice(name)
    }

    /// Start the context manager
    pub fn __enter__(slf: Py<Self>) -> Py<Self> {
        slf
    }

    /// Exits the context manager
    pub fn __exit__(&self, _exc_type: Py<PyAny>, _exc_value: Py<PyAny>, _traceback: Py<PyAny>) {
        // dropped once the lock is released
        let _open = self.inner.lock().unwrap().take();
    }
}

/// A tensor's header entry, as written in the file.
#[pyclass(frozen, get_all)]
#[derive(Debug)]
pub struct TensorMeta {
    dtype: String,
    shape: Vec<usize>,
    /// `(start, end)` of the tensor's bytes within the data section
    data_offsets: (usize, usize),
}

#[pymethods]
impl TensorMeta {
    fn __repr__(&self) -> String {
        format!("{self:?}")
    }
}

#[pyclass]
struct PySafeSlice {
    info: TensorInfo,
    framework: Framework,
    offset: usize,
    device: Device,
    storage: Arc<Storage>,
}

use std::fmt;

use crate::engine::{Feed, Loader, LoaderError, Span, TensorIter};

struct Disp(Vec<TensorIndexer>);

/// Should be more readable that the standard
/// `Debug`
impl fmt::Display for Disp {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "[")?;
        for (i, item) in self.0.iter().enumerate() {
            write!(f, "{prefix}{item}", prefix = if i == 0 { "" } else { ", " })?;
        }
        write!(f, "]")
    }
}

impl PySafeSlice {
    fn slice_bytes_to_tensor(
        &self,
        slices: &PyBound<'_, PyAny>,
        data: &[u8],
    ) -> PyResult<Py<PyAny>> {
        let indexers = parse_indexers(slices, &self.info.shape)?;

        let tensor = TensorView::new(self.info.dtype, self.info.shape.clone(), data)
            .map_err(|e| SafetensorError::new_err(format!("Error preparing tensor view: {e}")))?;

        let iterator = tensor.sliced_data(&indexers).map_err(|e| {
            SafetensorError::new_err(format!(
                "Error during slicing {} with shape {:?}: {e}",
                Disp(indexers),
                self.info.shape,
            ))
        })?;
        let newshape = iterator.newshape();
        let length = iterator.remaining_byte_len();

        let mut offset = 0;
        Python::attach(|py| {
            let array: Py<PyAny> = PyByteArray::new_with(py, length, |bytes: &mut [u8]| {
                for slice in iterator {
                    let len = slice.len();
                    bytes[offset..offset + len].copy_from_slice(slice);
                    offset += len;
                }
                Ok(())
            })?
            .into_any()
            .into();
            create_tensor(
                &self.framework,
                self.info.dtype,
                &newshape,
                array,
                &self.device,
            )
        })
    }

    /// MPS slice path: alloc a Shared `MTLBuffer` sized for the post-slice
    /// tensor, fill each source segment directly into it via pread/memcpy,
    /// then hand off via DLPack. Skips the bytearray + CPU tensor +
    /// `.to("mps")` round-trip the general path takes.
    #[cfg(all(target_os = "macos", target_arch = "aarch64"))]
    fn slice_mps(&self, slices: &PyBound<'_, PyAny>) -> PyResult<Py<PyAny>> {
        let indexers = parse_indexers(slices, &self.info.shape)?;

        let (ranges, newshape) =
            safetensors::slice::slice_byte_ranges(self.info.dtype, &self.info.shape, &indexers)
                .map_err(|e| {
                    SafetensorError::new_err(format!(
                        "Error during slicing {} with shape {:?}: {e}",
                        Disp(indexers.clone()),
                        self.info.shape,
                    ))
                })?;

        let total: usize = ranges.iter().map(|(a, b)| b - a).sum();
        let mut buf = crate::metal::MTLBuffer::alloc_shared(total)
            .map_err(|e| SafetensorError::new_err(format!("MTLBuffer alloc for slice: {e}")))?;
        if total > 0 {
            let tensor_base = self.offset + self.info.data_offsets.0;
            let dst = buf.as_mut_slice();

            match self.storage.as_ref() {
                Storage::Mmap(mmap) => {
                    let mut dst_off = 0usize;
                    for &(src_lo, src_hi) in &ranges {
                        let len = src_hi - src_lo;
                        dst[dst_off..dst_off + len]
                            .copy_from_slice(&mmap[tensor_base + src_lo..tensor_base + src_hi]);
                        dst_off += len;
                    }
                }
                Storage::Pread(file) => {
                    let mut dst_off = 0usize;
                    for &(src_lo, src_hi) in &ranges {
                        let len = src_hi - src_lo;
                        read_exact_at(
                            file,
                            &mut dst[dst_off..dst_off + len],
                            (tensor_base + src_lo) as u64,
                        )
                        .map_err(|e| SafetensorError::new_err(format!("pread for slice: {e}")))?;
                        dst_off += len;
                    }
                }
                Storage::Torch(storage) => Python::attach(|py| -> PyResult<()> {
                    let storage_obj = storage
                        .get()
                        .ok_or_else(|| SafetensorError::new_err("Could not find storage"))?
                        .bind(py);
                    let src_data_ptr: usize = storage_obj
                        .call_method0(intern!(py, "data_ptr"))?
                        .extract()?;
                    let (_, src_end) = self.info.data_offsets;
                    // SAFETY: the storage spans the tensor's bytes; held alive
                    // by `self.storage` for this call.
                    let src = unsafe {
                        std::slice::from_raw_parts(
                            (src_data_ptr + tensor_base) as *const u8,
                            src_end - self.info.data_offsets.0,
                        )
                    };
                    let mut dst_off = 0usize;
                    for &(src_lo, src_hi) in &ranges {
                        let len = src_hi - src_lo;
                        dst[dst_off..dst_off + len].copy_from_slice(&src[src_lo..src_hi]);
                        dst_off += len;
                    }
                    Ok(())
                })?,
                Storage::Paddle(_) => unreachable!("Paddle excluded at __getitem__ entry"),
            }
        }

        Python::attach(|py| {
            mps_tensor_from_buf(py, &self.framework, self.info.dtype, &newshape, buf)
        })
    }
}

#[pymethods]
impl PySafeSlice {
    /// Returns the shape of the full underlying tensor
    ///
    /// Returns:
    ///     (`List[int]`):
    ///         The shape of the full tensor
    ///
    /// Example:
    /// ```python
    /// from safetensors import safe_open
    ///
    /// with safe_open("model.safetensors", framework="pt", device=0) as f:
    ///     tslice = f.get_slice("embedding")
    ///     shape = tslice.get_shape()
    ///     dim = shape // 8
    ///     tensor = tslice[:, :dim]
    /// ```
    pub fn get_shape(&self, py: Python) -> PyResult<Py<PyAny>> {
        let shape = self.info.shape.clone();
        let shape: Py<PyAny> = shape.into_pyobject(py)?.into();
        Ok(shape)
    }

    /// Returns the dtype of the full underlying tensor
    ///
    /// Returns:
    ///     (`str`):
    ///         The dtype of the full tensor
    ///
    /// Example:
    /// ```python
    /// from safetensors import safe_open
    ///
    /// with safe_open("model.safetensors", framework="pt", device=0) as f:
    ///     tslice = f.get_slice("embedding")
    ///     dtype = tslice.get_dtype() # "F32"
    /// ```
    pub fn get_dtype(&self, py: Python) -> PyResult<Py<PyAny>> {
        Ok(self.info.dtype.to_string().into_pyobject(py)?.into())
    }

    pub fn __getitem__(&self, slices: &PyBound<'_, PyAny>) -> PyResult<Py<PyAny>> {
        #[cfg(all(target_os = "macos", target_arch = "aarch64"))]
        if self.device == Device::Mps
            && self.framework == Framework::Pytorch
            && !matches!(self.storage.as_ref(), Storage::Paddle(_))
            && dlpack::torch_mps_compatible(self.info.dtype)
            && torch_supports_mps_dlpack()
        {
            return self.slice_mps(slices);
        }

        match &self.storage.as_ref() {
            Storage::Mmap(mmap) => {
                let data = &mmap[self.info.data_offsets.0 + self.offset
                    ..self.info.data_offsets.1 + self.offset];
                self.slice_bytes_to_tensor(slices, data)
            }
            Storage::Pread(file) => {
                let (begin, end) = self.info.data_offsets;
                let nbytes = end - begin;
                let mut data = vec![0u8; nbytes];
                if nbytes > 0 {
                    read_exact_at(file, &mut data, (self.offset + begin) as u64).map_err(|e| {
                        SafetensorError::new_err(format!(
                            "Could not read tensor bytes for slicing: {e}"
                        ))
                    })?;
                }
                self.slice_bytes_to_tensor(slices, &data)
            }
            Storage::Torch(storage) => Python::attach(|py| -> PyResult<Py<PyAny>> {
                let torch = get_module(py, &TORCH_MODULE)?;
                let dtype: Py<PyAny> = get_pydtype(torch, self.info.dtype, false)?;
                let torch_uint8: Py<PyAny> = get_pydtype(torch, Dtype::U8, false)?;
                let kwargs = [(intern!(py, "dtype"), torch_uint8)].into_py_dict(py)?;
                let view_kwargs = [(intern!(py, "dtype"), dtype)].into_py_dict(py)?;
                let shape = self.info.shape.to_vec();
                let shape: Py<PyAny> = shape.into_pyobject(py)?.into();

                let start = (self.info.data_offsets.0 + self.offset) as isize;
                let stop = (self.info.data_offsets.1 + self.offset) as isize;
                let slice = PySlice::new(py, start, stop, 1);
                let storage: &Py<PyAny> = storage
                    .get()
                    .ok_or_else(|| SafetensorError::new_err("Could not find storage"))?;
                let storage: &PyBound<'_, PyAny> = storage.bind(py);

                let storage_slice = storage
                    .getattr(intern!(py, "__getitem__"))?
                    .call1((slice,))?;

                let slices = slices.into_pyobject(py)?;

                let sys = PyModule::import(py, intern!(py, "sys"))?;
                let byteorder: String = sys.getattr(intern!(py, "byteorder"))?.extract()?;

                let mut tensor = torch
                    .getattr(intern!(py, "asarray"))?
                    .call((storage_slice,), Some(&kwargs))?
                    .getattr(intern!(py, "view"))?
                    .call((), Some(&view_kwargs))?;
                if byteorder == "big" {
                    // Important, do NOT use inplace otherwise the slice itself
                    // is byteswapped, meaning multiple calls will fails
                    let inplace_kwargs =
                        [(intern!(py, "inplace"), PyBool::new(py, false))].into_py_dict(py)?;

                    let intermediary_dtype = match self.info.dtype {
                        Dtype::BF16 => Some(Dtype::F16),
                        Dtype::F8_E5M2 => Some(Dtype::U8),
                        Dtype::F8_E4M3 => Some(Dtype::U8),
                        Dtype::F8_E8M0 => Some(Dtype::U8),
                        _ => None,
                    };
                    if let Some(intermediary_dtype) = intermediary_dtype {
                        // Reinterpret to f16 for numpy compatibility.
                        let dtype: Py<PyAny> = get_pydtype(torch, intermediary_dtype, false)?;
                        let view_kwargs = [(intern!(py, "dtype"), dtype)].into_py_dict(py)?;
                        tensor = tensor
                            .getattr(intern!(py, "view"))?
                            .call((), Some(&view_kwargs))?;
                    }
                    let numpy = tensor
                        .getattr(intern!(py, "numpy"))?
                        .call0()?
                        .getattr("byteswap")?
                        .call((), Some(&inplace_kwargs))?;
                    tensor = torch.getattr(intern!(py, "from_numpy"))?.call1((numpy,))?;
                    if intermediary_dtype.is_some() {
                        // Reinterpret to f16 for numpy compatibility.
                        let dtype: Py<PyAny> = get_pydtype(torch, self.info.dtype, false)?;
                        let view_kwargs = [(intern!(py, "dtype"), dtype)].into_py_dict(py)?;
                        tensor = tensor
                            .getattr(intern!(py, "view"))?
                            .call((), Some(&view_kwargs))?;
                    }
                }
                tensor = tensor
                    .getattr(intern!(py, "reshape"))?
                    .call1((shape,))?
                    .getattr(intern!(py, "__getitem__"))?
                    .call1((slices,))?;
                if self.device != Device::Cpu {
                    let device: Py<PyAny> = self.device.clone().into_pyobject(py)?.into();
                    let kwargs = PyDict::new(py);
                    tensor = tensor.call_method("to", (device,), Some(&kwargs))?;
                }
                Ok(tensor.into())
            }),
            Storage::Paddle(storage) => Python::attach(|py| -> PyResult<Py<PyAny>> {
                let paddle = get_module(py, &PADDLE_MODULE)?;
                let cur_type = if self.info.dtype == Dtype::U16 {
                    Dtype::BF16
                } else {
                    self.info.dtype
                };
                let dtype: Py<PyAny> = get_pydtype(paddle, cur_type, false)?;
                let paddle_uint8: Py<PyAny> = get_pydtype(paddle, Dtype::U8, false)?;
                let shape = self.info.shape.to_vec();
                let shape: Py<PyAny> = shape.into_pyobject(py)?.into();
                let start = (self.info.data_offsets.0 + self.offset) as isize;
                let stop = (self.info.data_offsets.1 + self.offset) as isize;
                let slices = slices.into_pyobject(py)?;
                let storage: &Py<PyAny> = storage
                    .get()
                    .ok_or_else(|| SafetensorError::new_err("Could not find storage"))?;
                let storage: &PyBound<'_, PyAny> = storage.bind(py);
                let slice_kwargs = [
                    (intern!(py, "dtype"), paddle_uint8),
                    (intern!(py, "start"), start.into_pyobject(py)?.into()),
                    (intern!(py, "stop"), stop.into_pyobject(py)?.into()),
                ]
                .into_py_dict(py)?;
                let storage_slice = storage
                    .getattr(intern!(py, "get_slice"))?
                    .call((), Some(&slice_kwargs))?;
                let mut tensor = storage_slice
                    .getattr(intern!(py, "view"))?
                    .call1((dtype,))?;
                let sys = PyModule::import(py, intern!(py, "sys"))?;
                let byteorder: String = sys.getattr(intern!(py, "byteorder"))?.extract()?;
                if byteorder == "big" {
                    let inplace_kwargs =
                        [(intern!(py, "inplace"), PyBool::new(py, false))].into_py_dict(py)?;

                    let intermediary_dtype = match cur_type {
                        Dtype::BF16 => Some(Dtype::F16),
                        Dtype::F8_E5M2 => Some(Dtype::U8),
                        Dtype::F8_E4M3 => Some(Dtype::U8),
                        Dtype::F8_E8M0 => Some(Dtype::U8),
                        _ => None,
                    };
                    if let Some(intermediary_dtype) = intermediary_dtype {
                        // Reinterpret to f16 for numpy compatibility.
                        let dtype: Py<PyAny> = get_pydtype(paddle, intermediary_dtype, false)?;
                        tensor = tensor.getattr(intern!(py, "view"))?.call1((dtype,))?;
                    }
                    let numpy = tensor
                        .getattr(intern!(py, "numpy"))?
                        .call0()?
                        .getattr("byteswap")?
                        .call((), Some(&inplace_kwargs))?;
                    tensor = paddle.getattr(intern!(py, "to_tensor"))?.call1((numpy,))?;
                    if intermediary_dtype.is_some() {
                        // Reinterpret to f16 for numpy compatibility.
                        let dtype: Py<PyAny> = get_pydtype(paddle, cur_type, false)?;
                        tensor = tensor.getattr(intern!(py, "view"))?.call1((dtype,))?;
                    }
                }
                tensor = tensor
                    .getattr(intern!(py, "reshape"))?
                    .call1((shape,))?
                    .getattr(intern!(py, "__getitem__"))?
                    .call1((slices,))?;
                if self.device != Device::Cpu {
                    let device: Py<PyAny> = if let Device::Cuda(index) = self.device {
                        format!("gpu:{index}").into_pyobject(py)?.into()
                    } else {
                        self.device.clone().into_pyobject(py)?.into()
                    };
                    let kwargs = PyDict::new(py);
                    tensor = tensor.call_method("to", (device,), Some(&kwargs))?;
                }
                // Paddle's MmapStorage.get_slice() doesn't keep the storage alive,
                // so we attach it to the tensor to prevent it from being garbage collected
                tensor.setattr(intern!(py, "_safetensors_storage"), storage)?;
                Ok(tensor.into())
            }),
        }
    }
}

fn get_module<'a>(
    py: Python<'a>,
    cell: &'static OnceLock<Py<PyModule>>,
) -> PyResult<&'a PyBound<'a, PyModule>> {
    let module: &PyBound<'a, PyModule> = cell
        .get()
        .ok_or_else(|| SafetensorError::new_err("Could not find module"))?
        .bind(py);
    Ok(module)
}

/// Mark `file` to bypass the page cache for `read`/`pread` calls on macOS.
///
/// `F_NOCACHE` only affects I/O syscalls on this fd; it has no effect on
/// pages accessed via `mmap`, so this is only useful for fds we'll
/// `pread()` from. Best-effort: failures (e.g. on filesystems that don't
/// honor it) are ignored. No-op on non-macOS.
#[allow(unused_variables)]
fn disable_page_cache_macos(file: &File) {
    #[cfg(target_os = "macos")]
    {
        use std::os::fd::AsRawFd;
        // SAFETY: fcntl with F_NOCACHE takes (fd, cmd, int) and returns an
        // int. We pass an owned, valid fd from `file` and ignore the result
        // since this is a hint, not a correctness requirement.
        unsafe {
            libc::fcntl(file.as_raw_fd(), libc::F_NOCACHE, 1);
        }
    }
}

/// Portable positional read: fills `buf` from `file` starting at `offset`.
///
/// **Thread-safety:** safe to call concurrently from multiple threads on the
/// same `File`. Both backends (Unix `pread`, Windows `ReadFile` with an
/// `OVERLAPPED` offset) take the read position as an explicit parameter and
/// do not consult the file handle's seek cursor. Windows does still update
/// the synchronous handle's internal cursor as a side-effect (so it ends up
/// at an unspecified position after concurrent calls), but we never read
/// from that cursor; every call passes its own `offset`.
fn read_exact_at(file: &File, buf: &mut [u8], offset: u64) -> std::io::Result<()> {
    #[cfg(unix)]
    {
        use std::os::unix::fs::FileExt;
        file.read_exact_at(buf, offset)
    }
    #[cfg(windows)]
    {
        use std::os::windows::fs::FileExt;
        let mut written = 0;
        while written < buf.len() {
            let n = file.seek_read(&mut buf[written..], offset + written as u64)?;
            if n == 0 {
                return Err(std::io::Error::new(
                    std::io::ErrorKind::UnexpectedEof,
                    "read_exact_at: early EOF",
                ));
            }
            written += n;
        }
        Ok(())
    }
}

/// Storage shape for torch tensors. F4 packs two elements per byte, so the
/// `torch.empty` shape halves the last dim relative to the logical shape
/// recorded in the safetensors header.
fn torch_storage_shape(dtype: Dtype, logical_shape: &[usize]) -> PyResult<Vec<usize>> {
    let mut shape = logical_shape.to_vec();
    if dtype == Dtype::F4 {
        let n = shape.len();
        if n == 0 || !shape[n - 1].is_multiple_of(2) {
            return Err(SafetensorError::new_err(format!(
                "f4_x2 dtype requires the last dim be divisible by 2 in torch: got {logical_shape:?}",
            )));
        }
        shape[n - 1] /= 2;
    }
    Ok(shape)
}

/// `torch.empty(storage_shape, dtype=dtype, **kwargs)`; sets `dtype` in `kwargs`
fn torch_empty_cuda(
    py: Python<'_>,
    torch: &PyBound<PyModule>,
    dtype: &Py<PyAny>,
    storage_shape: &[usize],
    kwargs: &PyBound<PyDict>,
) -> PyResult<Py<PyAny>> {
    let shape: Py<PyAny> = storage_shape.to_vec().into_pyobject(py)?.into();
    kwargs.set_item(intern!(py, "dtype"), dtype)?;
    Ok(torch
        .call_method(intern!(py, "empty"), (shape,), Some(kwargs))?
        .unbind())
}

/// Hand a filled `MTLBuffer` to the framework as a tensor via DLPack.
///
/// Zero-byte tensors arrive as a clamp-allocated buffer (`alloc_shared`); the
/// DLPack shape still carries the zero dim so `numel == 0` for the consumer.
///
/// For dtypes torch's DLPack doesn't accept natively (F4/F8 variants), the
/// capsule's wire dtype is `uint8` and we `.view(target)` on the torch side:
/// same bytes, correct dtype, no copy.
#[cfg(all(target_os = "macos", target_arch = "aarch64"))]
fn mps_tensor_from_buf(
    py: Python<'_>,
    framework: &Framework,
    dtype: Dtype,
    logical_shape: &[usize],
    buf: crate::metal::MTLBuffer,
) -> PyResult<Py<PyAny>> {
    let storage_shape = torch_storage_shape(dtype, logical_shape)?;
    mps_tensor_from_storage(py, framework, dtype, &storage_shape, buf)
}

/// As [`mps_tensor_from_buf`], from the shape torch stores `dtype` with.
#[cfg(all(target_os = "macos", target_arch = "aarch64"))]
fn mps_tensor_from_storage(
    py: Python<'_>,
    framework: &Framework,
    dtype: Dtype,
    storage_shape: &[usize],
    buf: crate::metal::MTLBuffer,
) -> PyResult<Py<PyAny>> {
    let shape_i64: Vec<i64> = storage_shape.iter().map(|&n| n as i64).collect();

    let view_target = if dlpack::dlpack_supported_native(dtype) {
        None
    } else {
        dlpack::torch_view_target(dtype)
    };
    let dl_dtype = if view_target.is_some() {
        dlpack::uint8_dlpack()
    } else {
        dlpack::dtype_to_dlpack(dtype)
    };

    let device = dlpack::metal_device();
    let tensor = dlpack::to_capsule(py, buf, shape_i64, dl_dtype, device, |capsule| {
        ingest_dlpack_mps(py, framework, capsule)
    })?;

    if let Some(name) = view_target {
        // Reinterpret uint8 bytes as the actual dtype. Currently torch-only;
        // dispatch gate ensures other frameworks don't reach here for
        // non-native dtypes.
        let torch = get_module(py, &TORCH_MODULE)?;
        let target = torch.getattr(name)?;
        let viewed = tensor
            .bind(py)
            .call_method1(intern!(py, "view"), (target,))?;
        Ok(viewed.unbind())
    } else {
        Ok(tensor)
    }
}

/// A torch tensor on a new Metal buffer of `size` bytes for `meta`'s span, and
/// the host address of its bytes.
#[cfg(all(target_os = "macos", target_arch = "aarch64"))]
fn mps_empty(py: Python<'_>, meta: &SpanMeta, size: usize) -> PyResult<(u64, Py<PyAny>)> {
    let buf = crate::metal::MTLBuffer::alloc_shared(size)
        .map_err(|e| SafetensorError::new_err(e.to_string()))?;
    let addr = buf.contents_ptr() as u64;
    let tensor = mps_tensor_from_storage(
        py,
        &Framework::Pytorch,
        meta.file_dtype,
        &meta.storage_shape,
        buf,
    )?;
    Ok((addr, tensor))
}

/// Whether torch's `from_dlpack` accepts an MPS-device capsule, added in
/// torch 2.9. We advertise support back to 2.6, so older torch takes the
/// generic CPU-tensor + `.to("mps")` path instead: correct on every version,
/// but it copies host->device rather than aliasing the `MTLBuffer`.
///
/// The result is process-cached; first call imports torch under the GIL.
#[cfg(all(target_os = "macos", target_arch = "aarch64"))]
fn torch_supports_mps_dlpack() -> bool {
    if let Some(&supported) = TORCH_MPS_DLPACK.get() {
        return supported;
    }
    *TORCH_MPS_DLPACK.get_or_init(|| {
        Python::attach(|py| {
            let Ok(torch) = get_module(py, &TORCH_MODULE) else {
                return false;
            };
            let Ok(version) = torch
                .getattr(intern!(py, "__version__"))
                .and_then(|v| v.extract::<String>())
            else {
                return false;
            };
            Version::from_string(&version)
                .map(|v| v >= Version::new(2, 9, 0))
                .unwrap_or(false)
        })
    })
}

/// Framework-specific `from_dlpack` dispatch for a Metal-typed capsule.
///
/// Today only Pytorch is wired. `Open::new`'s framework guard rejects
/// non-Cpu device for all other frameworks, so this is the only reachable
/// arm. The remaining frameworks return an explicit error to keep the
/// surface honest if the upstream guard is ever widened:
///
/// - **MLX**: no `from_dlpack` exists; `mx.array(capsule)` fails. The
///   public Metal-zero-copy path tracked at ml-explore/mlx#2855 isn't
///   shipped yet (needs allocator::Buffer wrapping + Deleter plumbing).
///   The numpy-buffer-protocol path is *not* end-to-end zero-copy:
///   MLX copies into its own MTLBuffer on first GPU dispatch.
/// - **Numpy / Jax / Paddle / Flax**: not relevant for MPS; none have a
///   functioning MPS device path.
#[cfg(all(target_os = "macos", target_arch = "aarch64"))]
fn ingest_dlpack_mps(
    py: Python<'_>,
    framework: &Framework,
    capsule: &PyBound<'_, PyAny>,
) -> PyResult<Py<PyAny>> {
    match framework {
        Framework::Pytorch => {
            let torch = get_module(py, &TORCH_MODULE)?;
            Ok(torch
                .call_method1(intern!(py, "from_dlpack"), (capsule,))?
                .into())
        }
        Framework::Tensorflow => Err(SafetensorError::new_err(
            "Tensorflow uses the copy path; should not reach DLPack ingestion",
        )),
        other => Err(SafetensorError::new_err(format!(
            "MPS DLPack ingestion is not implemented for framework {other}"
        ))),
    }
}

/// Pre-allocated pinned-CPU torch destination.
struct PinnedCpuDest<'py> {
    tensor: PyBound<'py, PyAny>,
    write_ptr: usize,
}

impl<'py> PinnedCpuDest<'py> {
    /// Allocate `torch.empty(...,device="cpu", pin_memory=True)` and extract
    /// the data pointer for direct pread.
    fn new(
        py: Python<'py>,
        torch: &PyBound<'py, PyModule>,
        dtype: Dtype,
        logical_shape: &[usize],
        nbytes: usize,
    ) -> PyResult<Self> {
        let dtype_obj: Py<PyAny> = get_pydtype(torch, dtype, false)?;
        let storage_shape = torch_storage_shape(dtype, logical_shape)?;
        let shape_obj: Py<PyAny> = storage_shape.into_pyobject(py)?.into();
        let cpu_device: Py<PyAny> = "cpu".into_pyobject(py)?.into();
        let pin: Py<PyAny> = PyBool::new(py, true).to_owned().into_any().into();
        let kwargs = [
            (intern!(py, "dtype"), dtype_obj),
            (intern!(py, "device"), cpu_device),
            (intern!(py, "pin_memory"), pin),
        ]
        .into_py_dict(py)?;
        let tensor = torch.call_method("empty", (shape_obj,), Some(&kwargs))?;

        if nbytes == 0 {
            return Ok(Self {
                tensor,
                write_ptr: 0,
            });
        }

        let write_ptr: usize = tensor.call_method0(intern!(py, "data_ptr"))?.extract()?;
        Ok(Self { tensor, write_ptr })
    }
}

fn create_tensor<'a>(
    framework: &'a Framework,
    dtype: Dtype,
    shape: &'a [usize],
    array: Py<PyAny>,
    device: &'a Device,
) -> PyResult<Py<PyAny>> {
    Python::attach(|py| -> PyResult<Py<PyAny>> {
        let (module, is_numpy): (&PyBound<'_, PyModule>, bool) = match framework {
            Framework::Pytorch => (
                TORCH_MODULE
                    .get()
                    .ok_or_else(|| {
                        SafetensorError::new_err(format!("Could not find module {framework}",))
                    })?
                    .bind(py),
                false,
            ),
            frame => {
                // Attempt to load the frameworks
                // Those are needed to prepare the ml dtypes
                // like bfloat16
                match frame {
                    Framework::Tensorflow => {
                        let _ = PyModule::import(py, intern!(py, "tensorflow"));
                    }
                    Framework::Flax => {
                        let _ = PyModule::import(py, intern!(py, "flax"));
                    }
                    Framework::Paddle => {
                        let _ = PyModule::import(py, intern!(py, "paddle"));
                    }
                    _ => {}
                };

                (get_module(py, &NUMPY_MODULE)?, true)
            }
        };
        let dtype: Py<PyAny> = get_pydtype(module, dtype, is_numpy)?;
        let count: usize = shape.iter().product();
        let shape = shape.to_vec();
        let tensor = if count == 0 {
            // Torch==1.10 does not allow frombuffer on empty buffers so we create
            // the tensor manually.
            // let zeros = module.getattr(intern!(py, "zeros"))?;
            let shape: Py<PyAny> = shape.clone().into_pyobject(py)?.into();
            let args = (shape,);
            let kwargs = [(intern!(py, "dtype"), dtype)].into_py_dict(py)?;
            module.call_method("zeros", args, Some(&kwargs))?
        } else {
            // let frombuffer = module.getattr(intern!(py, "frombuffer"))?;
            let kwargs = [
                (intern!(py, "buffer"), array),
                (intern!(py, "dtype"), dtype),
            ]
            .into_py_dict(py)?;
            let mut tensor = module.call_method("frombuffer", (), Some(&kwargs))?;
            let sys = PyModule::import(py, intern!(py, "sys"))?;
            let byteorder: String = sys.getattr(intern!(py, "byteorder"))?.extract()?;
            if byteorder == "big" {
                let inplace_kwargs =
                    [(intern!(py, "inplace"), PyBool::new(py, false))].into_py_dict(py)?;
                tensor = tensor
                    .getattr("byteswap")?
                    .call((), Some(&inplace_kwargs))?;
            }
            tensor
        };
        let mut tensor: PyBound<'_, PyAny> = tensor.call_method1("reshape", (shape,))?;
        let tensor = match framework {
            Framework::Flax => {
                let module = Python::attach(|py| -> PyResult<&Py<PyModule>> {
                    let module = PyModule::import(py, intern!(py, "jax"))?;
                    Ok(FLAX_MODULE.get_or_init_py_attached(py, || module.into()))
                })?
                .bind(py);
                module
                    .getattr(intern!(py, "numpy"))?
                    .getattr(intern!(py, "array"))?
                    .call1((tensor,))?
            }
            Framework::Tensorflow => {
                let module = Python::attach(|py| -> PyResult<&Py<PyModule>> {
                    let module = PyModule::import(py, intern!(py, "tensorflow"))?;
                    Ok(TENSORFLOW_MODULE.get_or_init_py_attached(py, || module.into()))
                })?
                .bind(py);
                module
                    .getattr(intern!(py, "convert_to_tensor"))?
                    .call1((tensor,))?
            }
            Framework::Mlx => {
                let module = Python::attach(|py| -> PyResult<&Py<PyModule>> {
                    let module = PyModule::import(py, intern!(py, "mlx"))?;
                    Ok(MLX_MODULE.get_or_init_py_attached(py, || module.into()))
                })?
                .bind(py);
                module
                    .getattr(intern!(py, "core"))?
                    // .getattr(intern!(py, "array"))?
                    .call_method1("array", (tensor,))?
            }
            Framework::Paddle => {
                let module = Python::attach(|py| -> PyResult<&Py<PyModule>> {
                    let module = PyModule::import(py, intern!(py, "paddle"))?;
                    Ok(PADDLE_MODULE.get_or_init_py_attached(py, || module.into()))
                })?
                .bind(py);
                let device: Py<PyAny> = if let Device::Cuda(index) = device {
                    format!("gpu:{index}").into_pyobject(py)?.into()
                } else {
                    device.clone().into_pyobject(py)?.into()
                };
                let kwargs = [(intern!(py, "place"), device)].into_py_dict(py)?;
                let tensor = module
                    .getattr(intern!(py, "to_tensor"))?
                    .call((tensor,), Some(&kwargs))?;
                tensor
            }
            Framework::Pytorch => {
                if device != &Device::Cpu {
                    let device: Py<PyAny> = device.clone().into_pyobject(py)?.into();
                    let kwargs = PyDict::new(py);
                    tensor = tensor.call_method("to", (device,), Some(&kwargs))?;
                }
                tensor
            }
            Framework::Numpy => tensor,
        };
        // let tensor = tensor.into_py_bound(py);
        Ok(tensor.into())
    })
}

fn get_pydtype(
    module: &PyBound<'_, PyModule>,
    dtype: Dtype,
    is_numpy: bool,
) -> PyResult<Py<PyAny>> {
    Python::attach(|py| {
        let dtype: Py<PyAny> = match dtype {
            Dtype::F64 => module.getattr(intern!(py, "float64"))?.into(),
            Dtype::F32 => module.getattr(intern!(py, "float32"))?.into(),
            Dtype::BF16 => {
                if is_numpy {
                    module
                        .getattr(intern!(py, "dtype"))?
                        .call1(("bfloat16",))?
                        .into()
                } else {
                    module.getattr(intern!(py, "bfloat16"))?.into()
                }
            }
            Dtype::F16 => module.getattr(intern!(py, "float16"))?.into(),
            Dtype::U64 => module.getattr(intern!(py, "uint64"))?.into(),
            Dtype::I64 => module.getattr(intern!(py, "int64"))?.into(),
            Dtype::U32 => module.getattr(intern!(py, "uint32"))?.into(),
            Dtype::I32 => module.getattr(intern!(py, "int32"))?.into(),
            Dtype::U16 => module.getattr(intern!(py, "uint16"))?.into(),
            Dtype::I16 => module.getattr(intern!(py, "int16"))?.into(),
            Dtype::U8 => module.getattr(intern!(py, "uint8"))?.into(),
            Dtype::I8 => module.getattr(intern!(py, "int8"))?.into(),
            Dtype::BOOL => {
                if is_numpy {
                    py.import("builtins")?.getattr(intern!(py, "bool"))?.into()
                } else {
                    module.getattr(intern!(py, "bool"))?.into()
                }
            }
            Dtype::F8_E4M3 => module.getattr(intern!(py, "float8_e4m3fn"))?.into(),
            Dtype::F8_E4M3FNUZ => module.getattr(intern!(py, "float8_e4m3fnuz"))?.into(),
            Dtype::F8_E5M2 => module.getattr(intern!(py, "float8_e5m2"))?.into(),
            Dtype::F8_E5M2FNUZ => module.getattr(intern!(py, "float8_e5m2fnuz"))?.into(),
            Dtype::F8_E8M0 => module.getattr(intern!(py, "float8_e8m0fnu"))?.into(),
            Dtype::F4 => module.getattr(intern!(py, "float4_e2m1fn_x2"))?.into(),
            Dtype::C64 => module.getattr(intern!(py, "complex64"))?.into(),
            dtype => {
                return Err(SafetensorError::new_err(format!(
                    "Dtype not understood: {dtype}"
                )))
            }
        };
        Ok(dtype)
    })
}

pyo3::create_exception!(
    safetensors._safetensors_rust,
    SafetensorError,
    PyException,
    "Custom Python Exception for Safetensor errors."
);

#[pyclass(frozen)]
#[allow(non_camel_case_types)]
struct _safe_open_handle {
    /// `None` once closed. Each call works on its own reference, so closing never
    /// waits for one: the file closes when the last call returns.
    inner: Mutex<Option<Arc<Open>>>,
}

impl _safe_open_handle {
    fn inner(&self) -> PyResult<Arc<Open>> {
        let inner = self.inner.lock().unwrap().clone();
        inner.ok_or_else(|| SafetensorError::new_err("File is closed".to_string()))
    }
}

#[pymethods]
impl _safe_open_handle {
    #[new]
    #[pyo3(signature = (f, framework, device=Some(Device::Cpu), *, backend=Backend::Mmap))]
    fn new(
        f: Py<PyAny>,
        framework: Framework,
        device: Option<Device>,
        backend: Backend,
    ) -> PyResult<Self> {
        let filename = Python::attach(|py| -> PyResult<PathBuf> {
            let _ = f.getattr(py, "fileno")?;
            let filename = f.getattr(py, "name")?;
            let filename: PathBuf = filename.extract(py)?;
            Ok(filename)
        })?;
        let open = Open::new(filename, framework, device, backend)?;
        Ok(Self {
            inner: Mutex::new(Some(Arc::new(open))),
        })
    }

    /// See [`safe_open::prefetch`].
    #[pyo3(signature = (plan=None, *, device=None, threads=16, direct=None))]
    pub fn prefetch(
        &self,
        py: Python<'_>,
        plan: Option<&PyBound<'_, PyDict>>,
        device: Option<Device>,
        threads: usize,
        direct: Option<bool>,
    ) -> PyResult<PrefetchLoader> {
        self.inner()?.prefetch(py, plan, device, threads, direct)
    }

    /// Return the special non tensor information in the header
    ///
    /// Returns:
    ///     (`Dict[str, str]`):
    ///         The freeform metadata.
    pub fn metadata(&self) -> PyResult<Option<BTreeMap<String, String>>> {
        Ok(self.inner()?.metadata())
    }

    /// Returns the names of the tensors in the file.
    ///
    /// Returns:
    ///     (`List[str]`):
    ///         The name of the tensors contained in that file
    pub fn keys(&self) -> PyResult<Vec<String>> {
        self.inner()?.keys()
    }

    /// Returns a tensor's header entry without reading any data.
    ///
    /// Args:
    ///     name (`str`):
    ///         The name of the tensor
    ///
    /// Returns:
    ///     (`TensorMeta`):
    ///         `dtype` (safetensors name, e.g. `"F32"`), `shape` and `data_offsets`
    ///         as written in the file header. Works on every backend.
    pub fn get_tensor_meta(&self, name: &str) -> PyResult<TensorMeta> {
        self.inner()?.get_tensor_meta(name)
    }

    /// Returns the names of the tensors in the file, ordered by offset.
    ///
    /// Returns:
    ///     (`List[str]`):
    ///         The name of the tensors contained in that file
    pub fn offset_keys(&self) -> PyResult<Vec<String>> {
        self.inner()?.offset_keys()
    }

    /// Returns a full tensor
    ///
    /// Args:
    ///     name (`str`):
    ///         The name of the tensor you want
    ///
    /// Returns:
    ///     (`Tensor`):
    ///         The tensor in the framework you opened the file for.
    ///
    /// Example:
    /// ```python
    /// from safetensors import safe_open
    ///
    /// with safe_open("model.safetensors", framework="pt", device=0) as f:
    ///     tensor = f.get_tensor("embedding")
    /// ```
    pub fn get_tensor(&self, name: &str) -> PyResult<Py<PyAny>> {
        self.inner()?.get_tensor(name)
    }

    /// Returns every tensor in the file as a dict keyed by name.
    ///
    /// See `safe_open.get_tensors` for the device-specific dispatch.
    pub fn get_tensors(&self) -> PyResult<Py<PyDict>> {
        self.inner()?.get_tensors()
    }

    /// Returns a full slice view object
    ///
    /// Args:
    ///     name (`str`):
    ///         The name of the tensor you want
    ///
    /// Returns:
    ///     (`PySafeSlice`):
    ///         A dummy object you can slice into to get a real tensor
    /// Example:
    /// ```python
    /// from safetensors import safe_open
    ///
    /// with safe_open("model.safetensors", framework="pt", device=0) as f:
    ///     tensor_part = f.get_slice("embedding")[:, ::8]
    /// ```
    pub fn get_slice(&self, name: &str) -> PyResult<PySafeSlice> {
        self.inner()?.get_slice(name)
    }

    /// Start the context manager
    pub fn __enter__(slf: Py<Self>) -> Py<Self> {
        slf
    }

    /// Exits the context manager
    pub fn __exit__(&self, _exc_type: Py<PyAny>, _exc_value: Py<PyAny>, _traceback: Py<PyAny>) {
        // dropped once the lock is released
        let _open = self.inner.lock().unwrap().take();
    }
}

/// A Python module implemented in Rust.
#[pymodule(gil_used = false)]
fn _safetensors_rust(m: &PyBound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(serialize, m)?)?;
    m.add_function(wrap_pyfunction!(serialize_file, m)?)?;
    m.add_function(wrap_pyfunction!(deserialize, m)?)?;
    m.add_class::<TensorStream>()?;
    m.add_class::<PrefetchLoader>()?;
    m.add_class::<TensorSpec>()?;
    m.add_class::<TensorMeta>()?;
    m.add_class::<PySafeSlice>()?;
    m.add_class::<safe_open>()?;
    m.add_class::<_safe_open_handle>()?;
    #[cfg(unix)]
    {
        m.add_class::<reader::Reader>()?;
        m.add_class::<reader::ReadRequest>()?;
    }
    m.add("SafetensorError", m.py().get_type::<SafetensorError>())?;
    m.add("__version__", env!("CARGO_PKG_VERSION"))?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn version_parse() {
        let torch_version = "1.1.1";
        let version = Version::from_string(torch_version).unwrap();
        assert_eq!(version, Version::new(1, 1, 1));

        let torch_version = "2.0.0a0+gitd1123c9";
        let version = Version::from_string(torch_version).unwrap();
        assert_eq!(version, Version::new(2, 0, 0));

        let torch_version = "something";
        let version = Version::from_string(torch_version);
        assert!(version.is_err());
    }
}
