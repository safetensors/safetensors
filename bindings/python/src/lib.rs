#![deny(missing_docs)]
//! Dummy doc
#[cfg(all(target_os = "macos", target_arch = "aarch64"))]
mod dlpack;
#[cfg(all(target_os = "macos", target_arch = "aarch64"))]
mod metal;

use core::slice;
use memmap2::{Mmap, MmapOptions};
use pyo3::exceptions::{PyException, PyFileNotFoundError};
use pyo3::prelude::*;
use pyo3::sync::OnceLockExt;
use pyo3::types::IntoPyDict;
use pyo3::types::{PyBool, PyByteArray, PyBytes, PyDict, PyEllipsis, PyList, PySlice, PyTuple};
use pyo3::Bound as PyBound;
use pyo3::{intern, PyErr};
use safetensors::slice::TensorIndexer;
use safetensors::tensor::{Dtype, Metadata, SafeTensors, TensorInfo, TensorView};
use safetensors::View;
use std::borrow::Cow;
use std::collections::HashMap;
use std::fs::File;
use std::num::NonZeroUsize;
use std::ops::Bound;
use std::path::PathBuf;
use std::sync::Arc;
use std::sync::OnceLock;

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
/// Constructed from Python as `TensorSpec(dtype, shape, data_ptr, data_len)`.
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
/// SAFETY: `data_ptr` is a raw memory address. The caller must ensure the
/// underlying buffer stays alive for the duration of every `serialize` /
/// `serialize_file` call that consumes this spec.
#[pyclass(frozen, from_py_object)]
#[derive(Clone, Debug)]
struct TensorSpec {
    dtype: Dtype,
    shape: Vec<usize>,
    data_ptr: u64,
    data_len: usize,
}

#[pymethods]
impl TensorSpec {
    #[new]
    #[pyo3(signature = (*, dtype, shape, data_ptr, data_len))]
    fn new(dtype: &str, shape: Vec<usize>, data_ptr: u64, data_len: usize) -> PyResult<Self> {
        let dtype = parse_dtype_str(dtype)?;
        let mut shape = shape;
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
        Ok(Self {
            dtype,
            shape,
            data_ptr,
            data_len,
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

    /// The raw memory address of the tensor's contiguous buffer.
    #[getter]
    fn data_ptr(&self) -> u64 {
        self.data_ptr
    }

    /// The length of the tensor's buffer in bytes.
    #[getter]
    fn data_len(&self) -> usize {
        self.data_len
    }

    fn __repr__(&self) -> String {
        format!(
            "TensorSpec(dtype='{}', shape={:?}, data_ptr={}, data_len={})",
            self.dtype(),
            self.shape,
            self.data_ptr,
            self.data_len
        )
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
        let p = self.data_ptr as *const u8;
        // SAFETY: validated by the caller; see the struct-level safety note.
        unsafe {
            let slice = slice::from_raw_parts(p, self.data_len);
            Cow::Borrowed(slice)
        }
    }

    fn data_len(&self) -> usize {
        self.data_len
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
/// NOTE: the caller is required to ensure any pointer passed via `TensorSpec.data_ptr` is valid
/// and stays alive for the duration of the serialization.
/// We will remove the need for the caller to hold references themselves when we drop support for
/// python versions prior to 3.11 where the `PyBuffer` API is available.
/// Creating a `PyBuffer` will enable us to hold a reference to each passed in data array,
/// increasing its ref count preventing the gc from collecting it while we serialize.
///
/// Args:
///     tensor_dict (`Dict[str, TensorSpec]`):
///         Mapping of tensor name to its `TensorSpec`, e.g.:
///             {"tensor_name": TensorSpec(dtype="float32", shape=[2, 3], data_ptr=1234, data_len=24)}
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

/// Serializes raw data into file.
///
/// NOTE: the caller is required to ensure any pointer passed via `TensorSpec.data_ptr` is valid
/// and stays alive for the duration of the serialization.
/// We will remove the need for the caller to hold references themselves when we drop support for
/// python versions prior to 3.11 where the `PyBuffer` API is available.
/// Creating a `PyBuffer` will enable us to hold a reference to each passed in data array,
/// increasing its ref count preventing the gc from collecting it while we serialize.
///
/// Args:
///     tensor_dict (`Dict[str, TensorSpec]`):
///         Mapping of tensor name to its `TensorSpec`, e.g.:
///             {"tensor_name": TensorSpec(dtype="float32", shape=[2, 3], data_ptr=1234, data_len=24)}
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
    py.detach(|| {
        safetensors::tensor::serialize_to_file(
            tensor_dict.iter().map(|(k, v)| (k.as_str(), v.get())),
            metadata,
            filename.as_path(),
        )
        .map_err(|e| SafetensorError::new_err(format!("Error while serializing: {e}")))
    })?;

    Ok(())
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
            let start: Option<usize> = slice.getattr(intern!(py, "start"))?.extract()?;
            let stop: Option<usize> = slice.getattr(intern!(py, "stop"))?.extract()?;
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

struct Open {
    metadata: Metadata,
    offset: usize,
    framework: Framework,
    device: Device,
    storage: Arc<Storage>,
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
        let buffer = unsafe { MmapOptions::new().map_copy_read_only(&file)? };

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
                storage: Arc::new(Storage::Pread(Arc::new(file))),
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
        })
    }

    /// Return the special non tensor information in the header
    ///
    /// Returns:
    ///     (`Dict[str, str]`):
    ///         The freeform metadata.
    pub fn metadata(&self) -> Option<HashMap<String, String>> {
        self.metadata.metadata().clone()
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
    /// Default behavior is a sequential loop over `get_tensor`. Pytorch on
    /// Apple-silicon MPS with the `pread` backend instead bulk-allocates
    /// Shared `MTLBuffer`s, parallel-`pread`s into them, and hands them to
    /// torch via DLPack (1x model memory, the MTLBuffer is the destination).
    /// The `mmap` backend uses the sequential loop, reading through the mmap.
    pub fn get_tensors(&self) -> PyResult<Py<PyDict>> {
        // The bulk parallel-pread path is gated on the `pread` backend so the
        // `backend` choice is honored: `mmap` falls through to the per-tensor
        // loop below, which reads through the mmap via `get_tensor_mps`.
        #[cfg(all(target_os = "macos", target_arch = "aarch64"))]
        if self.device == Device::Mps
            && self.framework == Framework::Pytorch
            && torch_supports_mps_dlpack()
            && self
                .metadata
                .tensors()
                .values()
                .all(|info| dlpack::torch_mps_compatible(info.dtype))
        {
            if let Storage::Pread(file) = self.storage.as_ref() {
                return self.get_tensors_mps(file);
            }
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

    /// Bulk-allocates Shared-mode `MTLBuffer`s, parallel-`pread`s straight
    /// into each buffer's host-coherent contents pointer, then DLPack-hands
    /// the buffers off to the framework. `file` is the `pread` backend's fd
    /// (already `F_NOCACHE`d); the caller selects this path only for `Pread`.
    #[cfg(all(target_os = "macos", target_arch = "aarch64"))]
    fn get_tensors_mps(&self, file: &File) -> PyResult<Py<PyDict>> {
        let keys = self.metadata.offset_keys();
        let mut bufs: Vec<crate::metal::MTLBuffer> = Vec::with_capacity(keys.len());
        let mut jobs: Vec<PreadJob> = Vec::with_capacity(keys.len());
        let mut metas: Vec<(String, Dtype, Vec<usize>)> = Vec::with_capacity(keys.len());

        for name in &keys {
            let info = self.metadata.info(name).ok_or_else(|| {
                SafetensorError::new_err(format!("Missing tensor info for {name}"))
            })?;
            let (begin, end) = info.data_offsets;
            let nbytes = end - begin;
            metas.push((name.clone(), info.dtype, info.shape.clone()));

            let buf = crate::metal::MTLBuffer::alloc_shared(nbytes).map_err(|e| {
                SafetensorError::new_err(format!("MTLBuffer alloc for {name}: {e}"))
            })?;
            if nbytes > 0 {
                jobs.push(PreadJob {
                    name: name.clone(),
                    file_offset: (self.offset + begin) as u64,
                    nbytes,
                    write_ptr: buf.contents_ptr() as usize,
                });
            }
            bufs.push(buf);
        }

        Python::attach(|py| -> PyResult<Py<PyDict>> {
            if let Err((name, e)) = parallel_pread(py, file, &jobs) {
                return Err(SafetensorError::new_err(format!(
                    "pread failed for tensor {name}: {e}"
                )));
            }

            let dict = PyDict::new(py);
            for ((name, dtype, shape), buf) in metas.into_iter().zip(bufs) {
                let tensor = mps_tensor_from_buf(py, &self.framework, dtype, &shape, buf)?;
                dict.set_item(&name, tensor)?;
            }
            Ok(dict.into())
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
#[pyclass]
#[allow(non_camel_case_types)]
struct safe_open {
    inner: Option<Open>,
}

impl safe_open {
    fn inner(&self) -> PyResult<&Open> {
        let inner = self
            .inner
            .as_ref()
            .ok_or_else(|| SafetensorError::new_err("File is closed".to_string()))?;
        Ok(inner)
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
        let inner = Some(Open::new(filename, framework, device, backend)?);
        Ok(Self { inner })
    }

    /// Return the special non tensor information in the header
    ///
    /// Returns:
    ///     (`Dict[str, str]`):
    ///         The freeform metadata.
    pub fn metadata(&self) -> PyResult<Option<HashMap<String, String>>> {
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
    /// each; for `device="mps"` it instead bulk-allocates Shared MTLBuffers,
    /// parallel-`pread(2)`s into them, and hands them off via DLPack.
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
    pub fn __exit__(&mut self, _exc_type: Py<PyAny>, _exc_value: Py<PyAny>, _traceback: Py<PyAny>) {
        self.inner = None;
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

/// One pread(2) job: read `nbytes` from `file_offset` into `write_ptr`.
#[allow(dead_code)]
struct PreadJob {
    name: String,
    file_offset: u64,
    nbytes: usize,
    write_ptr: usize,
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

/// Run a set of pread(2) jobs in parallel without holding the GIL.
///
/// Caller must ensure each `(write_ptr, nbytes)` names a distinct, mutable,
/// allocated buffer that outlives this call (the GIL is released for the
/// duration). The number of workers is capped at 8: beyond that, NVMe and
/// Apple SSD reads should be I/O-bound rather than CPU-bound.
#[cfg_attr(
    not(all(target_os = "macos", target_arch = "aarch64")),
    allow(dead_code)
)]
fn parallel_pread(
    py: Python<'_>,
    file: &File,
    jobs: &[PreadJob],
) -> Result<(), (String, std::io::Error)> {
    use std::sync::atomic::{AtomicUsize, Ordering};

    py.detach(|| {
        let next = AtomicUsize::new(0);
        const MAX_WORKERS: usize = 8;
        let n_workers = std::thread::available_parallelism()
            .map_or(4, |n| n.get())
            .min(MAX_WORKERS)
            .min(jobs.len().max(1));

        std::thread::scope(|s| -> Result<(), (String, std::io::Error)> {
            let mut handles = Vec::with_capacity(n_workers);
            for _ in 0..n_workers {
                let next = &next;
                handles.push(s.spawn(move || -> Result<(), (String, std::io::Error)> {
                    loop {
                        let i = next.fetch_add(1, Ordering::Relaxed);
                        if i >= jobs.len() {
                            return Ok(());
                        }
                        let job = &jobs[i];
                        // SAFETY: caller contract (distinct, alive, mutable buffers).
                        let buf = unsafe {
                            std::slice::from_raw_parts_mut(job.write_ptr as *mut u8, job.nbytes)
                        };
                        read_exact_at(file, buf, job.file_offset)
                            .map_err(|e| (job.name.clone(), e))?;
                    }
                }));
            }
            for h in handles {
                h.join().map_err(|_| {
                    (
                        "<worker panic>".to_string(),
                        std::io::Error::other("worker panicked"),
                    )
                })??;
            }
            Ok(())
        })
    })
}

/// Storage shape for torch tensors. F4 packs two elements per byte, so the
/// `torch.empty` shape halves the last dim relative to the logical shape
/// recorded in the safetensors header.
fn torch_storage_shape(dtype: Dtype, logical_shape: &[usize]) -> PyResult<Vec<usize>> {
    let mut shape = logical_shape.to_vec();
    if dtype == Dtype::F4 {
        let n = shape.len();
        if n == 0 || shape[n - 1] % 2 != 0 {
            return Err(SafetensorError::new_err(format!(
                "f4_x2 dtype requires the last dim be divisible by 2 in torch: got {logical_shape:?}",
            )));
        }
        shape[n - 1] /= 2;
    }
    Ok(shape)
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
    let capsule = dlpack::to_capsule(py, buf, shape_i64, dl_dtype, device)?;
    let tensor = ingest_dlpack_mps(py, framework, capsule)?;

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

/// Whether torch's `from_dlpack` accepts an MPS-device capsule, added in
/// torch 2.9. We advertise support back to 2.4, so older torch takes the
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
    capsule: Py<PyAny>,
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

#[pyclass]
#[allow(non_camel_case_types)]
struct _safe_open_handle {
    inner: Option<Open>,
}

impl _safe_open_handle {
    fn inner(&self) -> PyResult<&Open> {
        let inner = self
            .inner
            .as_ref()
            .ok_or_else(|| SafetensorError::new_err("File is closed".to_string()))?;
        Ok(inner)
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
        let inner = Some(Open::new(filename, framework, device, backend)?);
        Ok(Self { inner })
    }

    /// Return the special non tensor information in the header
    ///
    /// Returns:
    ///     (`Dict[str, str]`):
    ///         The freeform metadata.
    pub fn metadata(&self) -> PyResult<Option<HashMap<String, String>>> {
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
    pub fn __exit__(&mut self, _exc_type: Py<PyAny>, _exc_value: Py<PyAny>, _traceback: Py<PyAny>) {
        self.inner = None;
    }
}

/// A Python module implemented in Rust.
#[pymodule(gil_used = false)]
fn _safetensors_rust(m: &PyBound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(serialize, m)?)?;
    m.add_function(wrap_pyfunction!(serialize_file, m)?)?;
    m.add_function(wrap_pyfunction!(deserialize, m)?)?;
    m.add_class::<TensorSpec>()?;
    m.add_class::<safe_open>()?;
    m.add_class::<_safe_open_handle>()?;
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
