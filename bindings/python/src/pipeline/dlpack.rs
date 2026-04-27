//! Python-side DLPack v1 wrapping. Thin layer over `ionic_rs::dlpack` —
//! adds the `PyCapsule` envelope + the safetensors `Dtype` translation.
//!
//! ## Lifetime / consumption protocol
//!
//! The capsule's destructor checks the capsule's name. Per the DLPack
//! consumer protocol, a consumer (`torch.from_dlpack`, etc.) renames the
//! capsule from `"dltensor_versioned"` to `"used_dltensor_versioned"` once
//! it takes ownership of the underlying memory. After that, the destructor
//! observes the renamed capsule and does nothing — the consumer is now in
//! charge of freeing via the deleter pointer it lifted out.
//!
//! If the capsule is never consumed (caller drops it, or torch raises
//! before consuming), the destructor sees the original name, finds the
//! pointer, and calls the producer-side deleter — keeping us leak-free.

use std::ffi::c_char;

use pyo3::prelude::*;

use ionic_rs::dlpack::{DLDataType, DLDevice, DLDeviceType, DLManagedTensor, CAPSULE_NAME};
use safetensors::Dtype;

use crate::SafetensorError;

// ── Dtype mapping ───────────────────────────────────────────────────────

/// Map a safetensors `Dtype` onto a DLPack v1 `DLDataType`.
///
/// Standardized codes (Int / UInt / Float / Bfloat / Bool / Complex) cover
/// the mainstream framework-recognized dtypes. Exotic variants (F4, F6,
/// F8) get `OpaqueHandle` — a consumer that recognizes the byte width can
/// reinterpret; one that doesn't sees raw bytes rather than crashing.
pub(crate) fn dtype_to_dlpack(dtype: Dtype) -> DLDataType {
    let bits = dtype.bitsize() as u8;
    match dtype {
        Dtype::BOOL => DLDataType::bool_(),
        Dtype::I8 | Dtype::I16 | Dtype::I32 | Dtype::I64 => DLDataType::int(bits),
        Dtype::U8 | Dtype::U16 | Dtype::U32 | Dtype::U64 => DLDataType::uint(bits),
        Dtype::F16 | Dtype::F32 | Dtype::F64 => DLDataType::float(bits),
        Dtype::BF16 => DLDataType::bfloat(bits),
        // Sub-byte and FP8 variants: the DLPack v1.0 wire format reserves
        // codes for some of these (kDLFloat8_e*, kDLFloat4_*) but framework
        // support is patchy. Opaque keeps us correct on the byte-layout
        // axis; consumers that know the framework-specific encoding can
        // reinterpret. Not great for ergonomics but doesn't lose data.
        _ => DLDataType::opaque(bits),
    }
}

pub(crate) fn cuda_device(ordinal: i32) -> DLDevice {
    DLDevice {
        device_type: DLDeviceType::Cuda,
        device_id: ordinal,
    }
}

// ── Capsule wrapping ────────────────────────────────────────────────────

/// Wrap a producer-side `*mut DLManagedTensor` in a Python
/// `PyCapsule`. Returns the capsule as a `Py<PyAny>` ready to hand to
/// `torch.from_dlpack` (or any v1-aware consumer).
///
/// On failure (PyCapsule_New returns null), the underlying managed tensor
/// is freed via its own deleter so we don't leak the device allocation.
pub(crate) fn make_capsule(
    py: Python<'_>,
    managed: *mut DLManagedTensor,
) -> PyResult<Py<PyAny>> {
    if managed.is_null() {
        return Err(SafetensorError::new_err(
            "make_capsule called with null managed tensor",
        ));
    }
    let name_ptr = CAPSULE_NAME.as_ptr() as *const c_char;
    // SAFETY: name is a static, NUL-terminated byte string of v1-spec
    // bytes (`b"dltensor_versioned\0"`). `managed` outlives the capsule
    // creation call.
    let capsule_ptr = unsafe {
        pyo3::ffi::PyCapsule_New(
            managed as *mut std::ffi::c_void,
            name_ptr,
            Some(capsule_destructor),
        )
    };
    if capsule_ptr.is_null() {
        // PyCapsule_New failed before we handed off ownership. Free the
        // managed tensor ourselves so the device buffer doesn't leak.
        // SAFETY: `managed` is still owned by us at this point.
        unsafe {
            if let Some(deleter) = (*managed).deleter {
                deleter(managed);
            }
        }
        return Err(PyErr::fetch(py));
    }
    // SAFETY: PyCapsule_New returned a new reference; `from_owned_ptr`
    // takes ownership.
    Ok(unsafe { Py::from_owned_ptr(py, capsule_ptr) })
}

/// PyCapsule destructor — fires on Python GC of the capsule. If the
/// capsule was already consumed (renamed by a `from_dlpack` call), this
/// is a no-op; otherwise it invokes the producer-side deleter so the
/// device allocation isn't leaked.
unsafe extern "C" fn capsule_destructor(capsule: *mut pyo3::ffi::PyObject) {
    let name_ptr = CAPSULE_NAME.as_ptr() as *const c_char;
    // PyCapsule_IsValid: 1 if name matches, 0 otherwise. No exception
    // is raised either way — safe inside a destructor.
    if unsafe { pyo3::ffi::PyCapsule_IsValid(capsule, name_ptr) } == 0 {
        // Capsule was consumed (renamed by from_dlpack) → consumer owns
        // the lifecycle now. Or the name is wrong (programmer error).
        // Either way, nothing for us to do.
        return;
    }
    let ptr = unsafe { pyo3::ffi::PyCapsule_GetPointer(capsule, name_ptr) };
    if ptr.is_null() {
        return;
    }
    let managed = ptr as *mut DLManagedTensor;
    // SAFETY: `managed` was put into the capsule by us via `make_capsule`
    // and the capsule was never consumed (IsValid passed). The deleter
    // is the standard `managed_tensor_deleter` populated at construction.
    unsafe {
        if let Some(deleter) = (*managed).deleter {
            deleter(managed);
        }
    }
}
