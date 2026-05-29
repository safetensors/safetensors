//! DLPack v0 producer + Python capsule wrapping.

use std::ffi::{c_char, c_void};

use pyo3::prelude::*;

use safetensors::Dtype;

use crate::metal::MTLBuffer;

// The structs and enums below are the ABI consumers read across the FFI
// boundary, so they must match the DLPack C definitions byte-for-byte
// (layout, field order, enum discriminants). Do not reorder or change
// representations without checking against the canonical header:
// https://github.com/dmlc/dlpack/blob/main/include/dlpack/dlpack.h

#[repr(i32)]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum DLDeviceType {
    Cpu = 1,
    Cuda = 2,
    Metal = 8,
}

#[repr(u8)]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum DLDataTypeCode {
    Int = 0,
    UInt = 1,
    Float = 2,
    OpaqueHandle = 3,
    Bfloat = 4,
    Complex = 5,
    Bool = 6,
}

#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct DLDataType {
    pub code: DLDataTypeCode,
    pub bits: u8,
    pub lanes: u16,
}

#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct DLDevice {
    pub device_type: DLDeviceType,
    pub device_id: i32,
}

#[repr(C)]
#[derive(Debug)]
pub struct DLTensor {
    pub data: *mut c_void,
    pub device: DLDevice,
    pub ndim: i32,
    pub dtype: DLDataType,
    pub shape: *mut i64,
    /// `null` means row-major C-contiguous per the DLPack spec; we rely
    /// on that to skip allocating a strides array.
    pub strides: *mut i64,
    pub byte_offset: u64,
}

#[repr(C)]
pub struct DLManagedTensor {
    pub dl_tensor: DLTensor,
    pub manager_ctx: *mut c_void,
    pub deleter: Option<unsafe extern "C" fn(*mut DLManagedTensor)>,
}

const CAPSULE_NAME: &[u8] = b"dltensor\0";

/// Source of the `data` pointer [`to_capsule`] writes into the `DLTensor`.
///
/// Invariant relied on by [`to_capsule`]: the pointer must stay valid as long
/// as `self` is alive and be unaffected by moving `self`. `to_capsule` reads
/// it once, then moves the buffer into [`ManagedCtx`], which keeps it alive
/// until `managed_tensor_deleter` runs.
pub(crate) trait AsDevicePtr: Send + 'static {
    fn as_device_ptr(&self) -> *mut c_void;
}

impl AsDevicePtr for MTLBuffer {
    fn as_device_ptr(&self) -> *mut c_void {
        // PyTorch's MPS from_dlpack reads `data` as `id<MTLBuffer>` and
        // looks it up in the MPS allocator's buffer table; passing
        // `contents()` is interpreted as a key into a different region
        // and segfaults.
        //
        // This is the address of the Obj-C buffer object, not the
        // `Retained`/`MTLBuffer` wrapper, so it survives the wrapper moving
        // into `ManagedCtx` and stays valid until that ctx drops the
        // `Retained`, satisfying the `AsDevicePtr` invariant.
        self.as_metal_id_ptr()
    }
}

struct ManagedCtx<B: AsDevicePtr> {
    _device_buf: B,
    /// Boxed slice (not `Vec`) so the pointer + length stay stable without
    /// tracking capacity; `DLTensor.shape` points into this.
    shape: Box<[i64]>,
}

/// # Safety
///
/// `self_ptr` must have been produced by [`to_capsule`] with the same `B`
/// and not deleted before.
unsafe extern "C" fn managed_tensor_deleter<B: AsDevicePtr>(self_ptr: *mut DLManagedTensor) {
    if self_ptr.is_null() {
        return;
    }
    let ctx_ptr = unsafe { (*self_ptr).manager_ctx as *mut ManagedCtx<B> };
    if !ctx_ptr.is_null() {
        unsafe { drop(Box::from_raw(ctx_ptr)) };
    }
    unsafe { drop(Box::from_raw(self_ptr)) };
}

pub(crate) fn dtype_to_dlpack(dtype: Dtype) -> DLDataType {
    let bits = dtype.bitsize() as u8;
    let code = match dtype {
        Dtype::BOOL => DLDataTypeCode::Bool,
        Dtype::I8 | Dtype::I16 | Dtype::I32 | Dtype::I64 => DLDataTypeCode::Int,
        Dtype::U8 | Dtype::U16 | Dtype::U32 | Dtype::U64 => DLDataTypeCode::UInt,
        Dtype::F16 | Dtype::F32 | Dtype::F64 => DLDataTypeCode::Float,
        Dtype::BF16 => DLDataTypeCode::Bfloat,
        Dtype::C64 => DLDataTypeCode::Complex,
        _ => DLDataTypeCode::OpaqueHandle,
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
        Dtype::F8_E8M0 => "float8_e8m0fnu",
        _ => return None,
    })
}

/// `uint8` capsule dtype. Used as the wire dtype for `view`-cast targets.
pub(crate) fn uint8_dlpack() -> DLDataType {
    DLDataType {
        code: DLDataTypeCode::UInt,
        bits: 8,
        lanes: 1,
    }
}

/// Whether torch's MPS fast path can ingest this dtype, either via native
/// DLPack support or via the `uint8 + view`-cast workaround.
pub(crate) fn torch_mps_compatible(dtype: Dtype) -> bool {
    dlpack_supported_native(dtype) || torch_view_target(dtype).is_some()
}

#[allow(dead_code)]
pub(crate) fn cpu_device() -> DLDevice {
    DLDevice {
        device_type: DLDeviceType::Cpu,
        device_id: 0,
    }
}

#[allow(dead_code)]
pub(crate) fn cuda_device(ordinal: i32) -> DLDevice {
    DLDevice {
        device_type: DLDeviceType::Cuda,
        device_id: ordinal,
    }
}

pub(crate) fn metal_device() -> DLDevice {
    DLDevice {
        device_type: DLDeviceType::Metal,
        device_id: 0,
    }
}

/// On `PyCapsule_New` failure we invoke the deleter so the device buffer
/// doesn't leak.
pub(crate) fn to_capsule<B: AsDevicePtr>(
    py: Python<'_>,
    device_buf: B,
    shape: Vec<i64>,
    dtype: DLDataType,
    device: DLDevice,
) -> PyResult<Py<PyAny>> {
    let ndim = shape.len() as i32;
    let data = device_buf.as_device_ptr();

    let ctx = Box::new(ManagedCtx {
        _device_buf: device_buf,
        shape: shape.into_boxed_slice(),
    });

    let shape_ptr = ctx.shape.as_ptr() as *mut i64;
    let ctx_ptr = Box::into_raw(ctx);

    // Hand the `DLManagedTensor` to the capsule. `managed_tensor_deleter`
    // (invoked when the capsule is dropped or consumed) reclaims both the
    // `ManagedCtx` (which owns the device buffer + shape) AND the
    // `DLManagedTensor` itself via `Box::from_raw`, so no leak on drop.
    let managed = Box::into_raw(Box::new(DLManagedTensor {
        dl_tensor: DLTensor {
            data,
            device,
            ndim,
            dtype,
            shape: shape_ptr,
            strides: std::ptr::null_mut(),
            byte_offset: 0,
        },
        manager_ctx: ctx_ptr as *mut c_void,
        deleter: Some(managed_tensor_deleter::<B>),
    }));

    let name_ptr = CAPSULE_NAME.as_ptr() as *const c_char;
    // SAFETY: for all the following unsafe calls, all data was produced by this function and is valid each the call.
    // In case of failure, we clean up the managed tensor ourselves to avoid leaks, since the capsule destructor won't run.
    let capsule_ptr = unsafe {
        pyo3::ffi::PyCapsule_New(managed as *mut c_void, name_ptr, Some(capsule_destructor))
    };
    if capsule_ptr.is_null() {
        unsafe { managed_tensor_deleter::<B>(managed) };
        return Err(PyErr::fetch(py));
    }
    Ok(unsafe { Bound::from_owned_ptr(py, capsule_ptr) }.unbind())
}

unsafe extern "C" fn capsule_destructor(capsule: *mut pyo3::ffi::PyObject) {
    let name_ptr = CAPSULE_NAME.as_ptr() as *const c_char;
    if unsafe { pyo3::ffi::PyCapsule_IsValid(capsule, name_ptr) } == 0 {
        return;
    }
    let ptr = unsafe { pyo3::ffi::PyCapsule_GetPointer(capsule, name_ptr) };
    if ptr.is_null() {
        return;
    }
    let managed = ptr as *mut DLManagedTensor;
    // SAFETY: we assume `managed` was put into the capsule by `to_capsule` and the
    // capsule was never consumed (IsValid passed).
    unsafe {
        if let Some(deleter) = (*managed).deleter {
            deleter(managed);
        }
    }
}
