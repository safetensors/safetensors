//! DLPack v0 producer — pure C ABI, pyo3-free.
//!
//! This module provides the data structures and the deleter machinery for
//! handing off device-resident tensors to any framework that consumes
//! DLPack. The Python layer (in `safetensors-python`) wraps a
//! [`DLManagedTensor`] pointer in a `PyCapsule` named `"dltensor"`;
//! non-Python Rust consumers can use this module directly.
//!
//! ## Spec choice: v0
//!
//! We emit the v0 capsule (`DLManagedTensor`, capsule name `dltensor`) for
//! maximum framework compat. PyTorch's `from_dlpack` accepts a v0 capsule
//! directly across every supported version; v1 (`DLManagedTensorVersioned`,
//! capsule name `dltensor_versioned`) only flows through the
//! `__dlpack__(max_version=...)` consumer-negotiation protocol, which
//! requires producer-side `__dlpack__` / `__dlpack_device__` methods on a
//! Python wrapper object.
//!
//! Our pipeline already synchronizes both streams at the end of `run`
//! before this module ever sees a buffer, so v1's stream-handoff field
//! buys us nothing today. If a future consumer wants async handoff via
//! the v1 protocol, we add a tiny pyclass wrapper exposing `__dlpack__`
//! that returns one of these v0 capsules — that's the cheapest evolution.
//!
//! ## Lifetimes / safety
//!
//! Producer side (us):
//! - We allocate a [`DLManagedTensorVersioned`] on the heap (Box), populate
//!   `manager_ctx` with a pointer to a Rust struct ([`ManagedCtx`]) that
//!   owns the underlying [`DeviceBuf`] and the heap-allocated shape array,
//!   and set `deleter` to [`managed_tensor_deleter`].
//! - The consumer (PyTorch's `from_dlpack`, etc.) is contractually
//!   responsible for invoking the deleter exactly once when its tensor
//!   handle drops.
//! - If the `PyCapsule` is destroyed without ever being consumed, the
//!   capsule destructor (in the bindings crate) invokes the deleter
//!   directly so we don't leak the device allocation.

use std::ffi::c_void;

use crate::cuda::DeviceBuf;

// ── DLPack v1 ABI types ─────────────────────────────────────────────────

/// Subset of `DLDeviceType` we actually emit. Adding more is a one-line
/// extension; we only enumerate what the CUDA pipeline produces today.
#[repr(i32)]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum DLDeviceType {
    /// `kDLCPU` — host memory.
    Cpu = 1,
    /// `kDLCUDA` — device memory (regular CUDA, not host or unified).
    Cuda = 2,
}

/// Type code of a `DLDataType`. Only the variants we currently emit.
#[repr(u8)]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum DLDataTypeCode {
    Int = 0,
    UInt = 1,
    Float = 2,
    /// Opaque handle — used for dtypes the framework doesn't model
    /// natively (e.g. our F4, F6, F8 variants beyond what DLPack
    /// standardizes); consumers typically reinterpret the bytes.
    OpaqueHandle = 3,
    Bfloat = 4,
    Complex = 5,
    Bool = 6,
}

#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct DLDataType {
    pub code: u8,
    pub bits: u8,
    pub lanes: u16,
}

impl DLDataType {
    pub const fn new(code: DLDataTypeCode, bits: u8) -> Self {
        Self {
            code: code as u8,
            bits,
            lanes: 1,
        }
    }

    pub const fn int(bits: u8) -> Self {
        Self::new(DLDataTypeCode::Int, bits)
    }
    pub const fn uint(bits: u8) -> Self {
        Self::new(DLDataTypeCode::UInt, bits)
    }
    pub const fn float(bits: u8) -> Self {
        Self::new(DLDataTypeCode::Float, bits)
    }
    pub const fn bfloat(bits: u8) -> Self {
        Self::new(DLDataTypeCode::Bfloat, bits)
    }
    pub const fn bool_() -> Self {
        Self::new(DLDataTypeCode::Bool, 8)
    }
    pub const fn complex(bits: u8) -> Self {
        Self::new(DLDataTypeCode::Complex, bits)
    }
    pub const fn opaque(bits: u8) -> Self {
        Self::new(DLDataTypeCode::OpaqueHandle, bits)
    }
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
    /// `null` signals "row-major C-contiguous" per the DLPack spec; we
    /// rely on this to avoid heap-allocating a strides array for every
    /// tensor we hand off.
    pub strides: *mut i64,
    pub byte_offset: u64,
}

/// DLPack v0 managed tensor. `dl_tensor` carries the data; `manager_ctx`
/// is opaque to consumers and points back to producer-side state; the
/// consumer calls `deleter(self)` exactly once when done.
#[repr(C)]
pub struct DLManagedTensor {
    pub dl_tensor: DLTensor,
    pub manager_ctx: *mut c_void,
    pub deleter: Option<unsafe extern "C" fn(*mut DLManagedTensor)>,
}

/// Capsule name used in the Python wrapping layer.
pub const CAPSULE_NAME: &[u8] = b"dltensor\0";

// ── Producer side ───────────────────────────────────────────────────────

/// Owns the producer-side state behind a [`DLManagedTensorVersioned`].
/// The deleter takes ownership via `Box::from_raw` and drops it, which
/// frees the device allocation (`cuMemFree`) and the heap-allocated
/// shape array.
struct ManagedCtx {
    /// Owns the device allocation. `Drop` calls `cuMemFree_v2`.
    _device_buf: DeviceBuf,
    /// Boxed slice rather than `Vec` so the pointer + length stay stable
    /// without us tracking capacity. `DLTensor.shape` points into this.
    shape: Box<[i64]>,
}

/// Allocate a heap [`DLManagedTensor`] that owns `device_buf` and `shape`,
/// with the standard deleter wired up. Returns a raw pointer the caller
/// hands to a consumer (typically wrapped in a `PyCapsule`).
///
/// Once handed off, the pointer is owned by the consumer protocol — call
/// `deleter` exactly once to release.
pub fn make_managed(
    device_buf: DeviceBuf,
    shape: Vec<i64>,
    dtype: DLDataType,
    device: DLDevice,
) -> *mut DLManagedTensor {
    let ndim = shape.len() as i32;
    // Snapshot the data pointer before moving `device_buf` into the ctx.
    let data = device_buf.as_device_ptr() as *mut c_void;

    let ctx = Box::new(ManagedCtx {
        _device_buf: device_buf,
        shape: shape.into_boxed_slice(),
    });
    // Pointer into the boxed slice; valid as long as `ctx` is alive (i.e.
    // until the deleter fires).
    let shape_ptr = ctx.shape.as_ptr() as *mut i64;
    let ctx_ptr = Box::into_raw(ctx);

    let managed = Box::new(DLManagedTensor {
        dl_tensor: DLTensor {
            data,
            device,
            ndim,
            dtype,
            shape: shape_ptr,
            // null = compact row-major; consumers infer strides from shape.
            strides: std::ptr::null_mut(),
            byte_offset: 0,
        },
        manager_ctx: ctx_ptr as *mut c_void,
        deleter: Some(managed_tensor_deleter),
    });
    Box::into_raw(managed)
}

/// Standard deleter: takes ownership of the managed tensor + its
/// `manager_ctx` and drops both. The drop of `ManagedCtx` cascades into
/// `DeviceBuf::Drop` which calls `cuMemFree`.
///
/// # Safety
///
/// `self_ptr` must have been produced by [`make_managed`] and must not
/// have been deleted before. The DLPack contract guarantees exactly-once
/// delete by the consumer.
pub unsafe extern "C" fn managed_tensor_deleter(self_ptr: *mut DLManagedTensor) {
    if self_ptr.is_null() {
        return;
    }
    // Reclaim ManagedCtx first so its Drop runs before the outer Box.
    let ctx_ptr = unsafe { (*self_ptr).manager_ctx as *mut ManagedCtx };
    if !ctx_ptr.is_null() {
        unsafe { drop(Box::from_raw(ctx_ptr)) };
    }
    unsafe { drop(Box::from_raw(self_ptr)) };
}
