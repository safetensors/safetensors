use crate::SafetensorError;
#[cfg(feature = "py311")]
use pyo3::buffer::PyBuffer;
use pyo3::prelude::*;
#[cfg(feature = "py38")]
use pyo3::types::PyBytes;
use pyo3::types::PyDict;
use pyo3::Bound as PyBound;
use safetensors::{Dtype, View};
use std::borrow::Cow;
use std::collections::HashMap;

#[cfg(feature = "py38")]
pub struct PyView<'a> {
    shape: Vec<usize>,
    dtype: Dtype,
    data: PyBound<'a, PyBytes>,
    data_len: usize,
}

#[cfg(feature = "py311")]
pub struct PyView<'a> {
    shape: Vec<usize>,
    dtype: Dtype,
    data: PyBuffer<u8>,
    data_len: usize,
    // Kept to keep the GIL open while we hold the buffer
    _py: Python<'a>,
}

impl View for &PyView<'_> {
    #[cfg(feature = "py38")]
    fn data(&self) -> std::borrow::Cow<[u8]> {
        Cow::Borrowed(self.data.as_bytes())
    }
    #[cfg(feature = "py311")]
    fn data(&self) -> std::borrow::Cow<[u8]> {
        // We already checked this in the Python side.
        assert!(self.data.is_c_contiguous());
        // XXX: Ideally we could have at least readonly tensors
        // assert!(self.data.readonly());
        // SAFETY:
        // This is actually totally unsafe, PyBuffer is not immutable and could be changed from
        // under us.
        // This is made safer because we're still hanging to the GIL while treating
        // this structure
        Cow::Borrowed(unsafe {
            std::slice::from_raw_parts(self.data.buf_ptr() as *const u8, self.data.item_count())
        })
    }
    fn shape(&self) -> &[usize] {
        &self.shape
    }
    fn dtype(&self) -> Dtype {
        self.dtype
    }
    fn data_len(&self) -> usize {
        self.data_len
    }
}

pub fn prepare(tensor_dict: HashMap<String, PyBound<PyDict>>) -> PyResult<HashMap<String, PyView>> {
    let mut tensors = HashMap::with_capacity(tensor_dict.len());
    for (tensor_name, tensor_desc) in &tensor_dict {
        let mut shape: Vec<usize> = tensor_desc
            .get_item("shape")?
            .ok_or_else(|| SafetensorError::new_err(format!("Missing `shape` in {tensor_desc:?}")))?
            .extract()?;
        let pydata: PyBound<PyAny> = tensor_desc.get_item("data")?.ok_or_else(|| {
            SafetensorError::new_err(format!("Missing `data` in {tensor_desc:?}"))
        })?;

        let pydtype = tensor_desc.get_item("dtype")?.ok_or_else(|| {
            SafetensorError::new_err(format!("Missing `dtype` in {tensor_desc:?}"))
        })?;
        let dtype: String = pydtype.extract()?;
        let dtype = match dtype.as_ref() {
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
            "float8_e8m0fnu" => Dtype::E8M0,
            "float4_e2m1fn_x2" => Dtype::F4,
            "complex64" => Dtype::C64,
            dtype_str => {
                return Err(SafetensorError::new_err(format!(
                    "dtype {dtype_str} is not covered",
                )));
            }
        };
        if dtype == Dtype::F4 {
            let n = shape.len();
            shape[n - 1] *= 2;
        }

        #[cfg(feature = "py311")]
        let tensor = {
            let data: PyBuffer<u8> = pydata.extract()?;
            if !data.is_c_contiguous() {
                return Err(SafetensorError::new_err("Python buffer is not contiguous"));
            }
            // XXX Ideally this would be true.
            // if !data.readonly() {
            //     return Err(SafetensorError::new_err("Python buffer is not readonly"));
            // }
            let data_len = data.item_count();
            let py = pydata.py();
            PyView {
                shape,
                dtype,
                data,
                data_len,
                _py: py,
            }
        };

        #[cfg(feature = "py38")]
        let tensor = {
            let data: &[u8] = pydata.extract()?;
            let data_len = data.len();
            let data: PyBound<PyBytes> = pydata.extract()?;
            PyView {
                shape,
                dtype,
                data,
                data_len,
            }
        };

        tensors.insert(tensor_name.to_string(), tensor);
    }
    Ok(tensors)
}
