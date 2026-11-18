//! DLPack v0 producer + Python capsule wrapping.

use std::ffi::{c_void, CStr};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;

use dlpark::ffi::{DLDataType, DLDataTypeCode, DLDevice, DLManagedTensor};
use dlpark::legacy;
use dlpark::metadata::Dynamic;
use pyo3::exceptions::PyBufferError;
use pyo3::prelude::*;

use safetensors::Dtype;

#[cfg(all(target_os = "macos", target_arch = "aarch64"))]
use crate::metal::MTLBuffer;

const USED_CAPSULE_NAME: &CStr = c"used_dltensor";

/// Source of the `data` pointer [`to_capsule`] writes into the `DLTensor`.
///
/// Invariant relied on by [`to_capsule`]: the pointer must stay valid as long
/// as `self` is alive and be unaffected by moving `self`. `to_capsule` reads
/// it once, then moves the buffer into [`Owner`], which keeps it alive until
/// the DLPack deleter runs.
pub(crate) trait AsDevicePtr: Send + 'static {
    fn as_device_ptr(&self) -> *mut c_void;
}

#[cfg(all(target_os = "macos", target_arch = "aarch64"))]
impl AsDevicePtr for MTLBuffer {
    fn as_device_ptr(&self) -> *mut c_void {
        // PyTorch's MPS from_dlpack reads `data` as `id<MTLBuffer>` and
        // looks it up in the MPS allocator's buffer table; passing
        // `contents()` is interpreted as a key into a different region
        // and segfaults.
        //
        // This is the address of the Obj-C buffer object, not the
        // `Retained`/`MTLBuffer` wrapper, so it survives the wrapper moving
        // into `Owner` and stays valid until that owner drops the `Retained`,
        // satisfying the `AsDevicePtr` invariant.
        self.as_metal_id_ptr()
    }
}

/// Managed tensor context: owns the device buffer and records its release.
struct Owner<B: AsDevicePtr> {
    _device_buf: B,
    deleted: Arc<AtomicBool>,
}

impl<B: AsDevicePtr> Drop for Owner<B> {
    fn drop(&mut self) {
        self.deleted.store(true, Ordering::Release);
    }
}

pub(crate) fn dtype_to_dlpack(dtype: Dtype) -> DLDataType {
    let bits = dtype.bitsize() as u8;
    let code = match dtype {
        Dtype::BOOL => DLDataTypeCode::BOOL,
        Dtype::I8 | Dtype::I16 | Dtype::I32 | Dtype::I64 => DLDataTypeCode::INT,
        Dtype::U8 | Dtype::U16 | Dtype::U32 | Dtype::U64 => DLDataTypeCode::UINT,
        Dtype::F16 | Dtype::F32 | Dtype::F64 => DLDataTypeCode::FLOAT,
        Dtype::BF16 => DLDataTypeCode::BFLOAT,
        Dtype::C64 => DLDataTypeCode::COMPLEX,
        _ => DLDataTypeCode::OPAQUEHANDLE,
    };

    DLDataType {
        code,
        bits,
        lanes: 1,
    }
}

/// Whether torch's `from_dlpack` accepts this dtype natively. F4/F6/F8
/// variants have DLPack v1.0 codes (8..=18) that PyTorch doesn't accept yet
/// (raises `BufferError: Unsupported code 3`); for those the MPS path emits
/// a `uint8` capsule and the consumer `.view()`s to reinterpret.
pub(crate) fn dlpack_supported_native(dtype: Dtype) -> bool {
    matches!(
        dtype,
        Dtype::BOOL
            | Dtype::I8
            | Dtype::I16
            | Dtype::I32
            | Dtype::I64
            | Dtype::U8
            | Dtype::U16
            | Dtype::U32
            | Dtype::U64
            | Dtype::F16
            | Dtype::F32
            | Dtype::F64
            | Dtype::BF16
            | Dtype::C64
    )
}

/// For dtypes torch doesn't accept via DLPack natively, the corresponding
/// `torch.<dtype>` name we can `.view()` to after importing as `uint8`.
/// `None` means there's no torch equivalent at all (fall back to copy path).
pub(crate) fn torch_view_target(dtype: Dtype) -> Option<&'static str> {
    Some(match dtype {
        Dtype::F4 => "float4_e2m1fn_x2",
        Dtype::F8_E5M2 => "float8_e5m2",
        Dtype::F8_E4M3 => "float8_e4m3fn",
        Dtype::F8_E4M3FNUZ => "float8_e4m3fnuz",
        Dtype::F8_E5M2FNUZ => "float8_e5m2fnuz",
        Dtype::F8_E8M0 => "float8_e8m0fnu",
        _ => return None,
    })
}

/// `uint8` capsule dtype. Used as the wire dtype for `view`-cast targets.
pub(crate) fn uint8_dlpack() -> DLDataType {
    DLDataType::U8
}

/// Whether torch's MPS fast path can ingest this dtype, either via native
/// DLPack support or via the `uint8 + view`-cast workaround.
#[cfg(all(target_os = "macos", target_arch = "aarch64"))]
pub(crate) fn torch_mps_compatible(dtype: Dtype) -> bool {
    dlpack_supported_native(dtype) || torch_view_target(dtype).is_some()
}

#[cfg(all(target_os = "macos", target_arch = "aarch64"))]
pub(crate) fn metal_device() -> DLDevice {
    DLDevice::metal(0)
}

/// Row-major strides as torch computes them for a contiguous tensor, where a
/// zero dim counts as one (`[2, 0, 3]` gives `[3, 3, 1]`).
fn torch_contiguous_strides(shape: &[i64]) -> Option<Vec<i64>> {
    let mut strides = vec![1_i64; shape.len()];
    for i in (1..shape.len()).rev() {
        strides[i - 1] = strides[i].checked_mul(shape[i].max(1))?;
    }
    Some(strides)
}

/// Wraps `device_buf` in a legacy `dltensor` capsule and passes it to `consume`.
///
/// The buffer is dropped once: by the DLPack deleter after a consumer took the
/// capsule, or with the capsule if it was never consumed.
pub(crate) fn to_capsule<'py, B: AsDevicePtr, R>(
    py: Python<'py>,
    device_buf: B,
    shape: Vec<i64>,
    dtype: DLDataType,
    device: DLDevice,
    consume: impl FnOnce(&Bound<'py, PyAny>) -> PyResult<R>,
) -> PyResult<R> {
    let data = device_buf.as_device_ptr();
    let deleted = Arc::new(AtomicBool::new(false));
    let owner = Box::new(Owner {
        _device_buf: device_buf,
        deleted: deleted.clone(),
    });
    let strides = torch_contiguous_strides(&shape)
        .ok_or_else(|| PyBufferError::new_err("tensor strides overflow i64"))?;
    let mut tensor = Dynamic::new(shape, strides)
        .initialize_as::<DLManagedTensor>(owner)
        .map_err(|e| PyBufferError::new_err(e.to_string()))?;
    tensor.set_data(data).set_device(device).set_dtype(dtype);
    // SAFETY: `data` stays valid until `Owner` drops; shape and strides live in the allocation.
    let tensor: legacy::Dlpack = unsafe { tensor.finish() };
    let capsule = tensor.into_pyobject(py)?;
    let result = consume(&capsule);
    // torch only renames the capsule once `from_dlpack` succeeds; if it throws
    // after its storage took ownership, the tensor is already freed.
    if deleted.load(Ordering::Acquire) {
        // SAFETY: `capsule` is a live capsule and the name is a static C string.
        unsafe { pyo3::ffi::PyCapsule_SetName(capsule.as_ptr(), USED_CAPSULE_NAME.as_ptr()) };
    }
    result
}

#[cfg(test)]
mod tests {
    use super::*;
    use pyo3::types::PyCapsule;
    use std::ptr::NonNull;
    use std::sync::atomic::AtomicUsize;

    struct Counted(Arc<AtomicUsize>);

    impl Drop for Counted {
        fn drop(&mut self) {
            self.0.fetch_add(1, Ordering::SeqCst);
        }
    }

    impl AsDevicePtr for Counted {
        fn as_device_ptr(&self) -> *mut c_void {
            NonNull::<u8>::dangling().as_ptr().cast()
        }
    }

    /// Returns how often the owner was dropped once the capsule is gone, and
    /// whether the capsule was marked used.
    fn run(consume: impl FnOnce(&Bound<'_, PyAny>) -> PyResult<()>) -> (usize, bool) {
        let drops = Arc::new(AtomicUsize::new(0));
        Python::initialize();
        let used = Python::attach(|py| {
            let shape = vec![2, 3];
            let dtype = dtype_to_dlpack(Dtype::BF16);
            let device = metal_device();
            let mut kept = None;
            let _ = to_capsule(
                py,
                Counted(drops.clone()),
                shape,
                dtype,
                device,
                |capsule| {
                    kept = Some(capsule.clone());
                    consume(capsule)
                },
            );
            let capsule = kept.unwrap();
            let capsule = capsule.cast::<PyCapsule>().unwrap();
            capsule.is_valid_checked(Some(USED_CAPSULE_NAME))
        });
        (drops.load(Ordering::SeqCst), used)
    }

    fn take(capsule: &Bound<'_, PyAny>) -> *mut DLManagedTensor {
        let capsule = capsule.cast::<PyCapsule>().unwrap();
        let ptr = capsule.pointer_checked(Some(c"dltensor")).unwrap();
        ptr.as_ptr().cast()
    }

    fn delete(managed: *mut DLManagedTensor) {
        // SAFETY: `managed` comes from a capsule built by `to_capsule` and is deleted once.
        unsafe { ((*managed).deleter.unwrap())(managed) };
    }

    #[test]
    fn strides_match_torch_for_zero_dims() {
        assert_eq!(torch_contiguous_strides(&[]), Some(vec![]));
        assert_eq!(torch_contiguous_strides(&[2, 3]), Some(vec![3, 1]));
        assert_eq!(torch_contiguous_strides(&[2, 0, 3]), Some(vec![3, 3, 1]));
        assert_eq!(torch_contiguous_strides(&[0, i64::MAX, 2]), None);
    }

    #[test]
    fn unconsumed_capsule_drops_owner() {
        assert_eq!(run(|_| Ok(())), (1, false));
    }

    #[test]
    fn consumed_capsule_drops_owner_once() {
        assert_eq!(
            run(|capsule| {
                let managed = take(capsule);
                // SAFETY: `capsule` is live and the name is a static C string.
                unsafe {
                    pyo3::ffi::PyCapsule_SetName(capsule.as_ptr(), USED_CAPSULE_NAME.as_ptr())
                };
                delete(managed);
                Ok(())
            }),
            (1, true)
        );
    }

    #[test]
    fn consumer_that_frees_and_raises_marks_capsule_used() {
        assert_eq!(
            run(|capsule| {
                delete(take(capsule));
                Err(PyBufferError::new_err("consumer failed"))
            }),
            (1, true)
        );
    }
}
