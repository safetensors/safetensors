//! Raw `libcuda.so.1` FFI — types, error codes, and the `CudaLib` function
//! pointer table resolved via `libloading` at first use.
//!
//! We intentionally link against the driver API (`libcuda.so.1`), not the
//! runtime API (`libcudart`), because the driver API:
//!   - Is the stable ABI that every CUDA installation ships (runtime is a
//!     user-space wrapper whose version floats with the CUDA Toolkit).
//!   - Supports primary-context attach (`cuDevicePrimaryCtxRetain`), which
//!     lets us co-exist with PyTorch's context rather than compete with it.
//!   - Is what ionic uses — same symbols, same semantics.
//!
//! All symbols are resolved at first use by `cuda::lib()` and cached in a
//! `OnceLock`. Runtime discovery means one wheel ships everywhere; absent
//! driver → `Error::CudaUnavailable` on first call to a CUDA primitive.

use std::ffi::{c_char, c_int, c_uint, c_void};

use libloading::Library;

use crate::error::{Error, Result};

// ── C types ─────────────────────────────────────────────────────────────

/// Driver result code. Zero means success; non-zero maps to a `cudaError_enum`
/// variant. We treat it as opaque and surface via `Error::Cuda`.
pub type CUresult = c_int;
pub type CUdevice = c_int;
pub type CUcontext = *mut c_void;
pub type CUstream = *mut c_void;
pub type CUevent = *mut c_void;
/// Device-space pointer. Always 64-bit in the driver ABI regardless of host
/// word size.
pub type CUdeviceptr = u64;

pub const CUDA_SUCCESS: CUresult = 0;
/// Returned by `cuEventQuery` when the event's captured work is still
/// outstanding. Non-blocking polls treat this as "not done yet," not an error.
pub const CUDA_ERROR_NOT_READY: CUresult = 600;

// Stream creation flags. `NON_BLOCKING` means this stream does not
// implicitly synchronize with the legacy default stream — critical for
// overlap with other CUDA work (e.g. PyTorch's compute streams).
pub const CU_STREAM_NON_BLOCKING: c_uint = 0x1;

// Event flags. Timing disabled because we only use events for happens-before,
// not profiling, and timing-disabled events are materially cheaper to record.
pub const CU_EVENT_DISABLE_TIMING: c_uint = 0x2;

// Host allocation flags. `PORTABLE` makes the pinning visible to all CUDA
// contexts in the process — important once multi-GPU lands.
pub const CU_MEMHOSTALLOC_PORTABLE: c_uint = 0x1;

// ── Function pointer table ──────────────────────────────────────────────
//
// Each field is a raw C ABI function pointer resolved once at load. Keeping
// the `Library` alive in `_lib` is load-bearing: dropping it would unload the
// .so and invalidate every pointer. `_lib` is never accessed directly after
// construction.

#[allow(non_snake_case)]
pub struct CudaLib {
    _lib: Library,
    pub cuInit: unsafe extern "C" fn(c_uint) -> CUresult,
    pub cuDeviceGetCount: unsafe extern "C" fn(*mut c_int) -> CUresult,
    pub cuDeviceGet: unsafe extern "C" fn(*mut CUdevice, c_int) -> CUresult,
    pub cuDeviceGetName: unsafe extern "C" fn(*mut c_char, c_int, CUdevice) -> CUresult,
    pub cuDeviceGetPCIBusId: unsafe extern "C" fn(*mut c_char, c_int, CUdevice) -> CUresult,
    pub cuDevicePrimaryCtxRetain:
        unsafe extern "C" fn(*mut CUcontext, CUdevice) -> CUresult,
    pub cuDevicePrimaryCtxRelease: unsafe extern "C" fn(CUdevice) -> CUresult,
    pub cuCtxPushCurrent: unsafe extern "C" fn(CUcontext) -> CUresult,
    pub cuCtxPopCurrent: unsafe extern "C" fn(*mut CUcontext) -> CUresult,
    pub cuStreamCreate: unsafe extern "C" fn(*mut CUstream, c_uint) -> CUresult,
    pub cuStreamDestroy: unsafe extern "C" fn(CUstream) -> CUresult,
    pub cuStreamSynchronize: unsafe extern "C" fn(CUstream) -> CUresult,
    pub cuEventCreate: unsafe extern "C" fn(*mut CUevent, c_uint) -> CUresult,
    pub cuEventDestroy: unsafe extern "C" fn(CUevent) -> CUresult,
    pub cuEventRecord: unsafe extern "C" fn(CUevent, CUstream) -> CUresult,
    pub cuEventSynchronize: unsafe extern "C" fn(CUevent) -> CUresult,
    /// Non-blocking event status: returns `CUDA_SUCCESS` if all captured
    /// work has completed, `CUDA_ERROR_NOT_READY` (=600) if work is still
    /// outstanding, or another error code on failure.
    pub cuEventQuery: unsafe extern "C" fn(CUevent) -> CUresult,
    pub cuMemAlloc: unsafe extern "C" fn(*mut CUdeviceptr, usize) -> CUresult,
    pub cuMemFree: unsafe extern "C" fn(CUdeviceptr) -> CUresult,
    pub cuMemHostAlloc: unsafe extern "C" fn(*mut *mut c_void, usize, c_uint) -> CUresult,
    pub cuMemFreeHost: unsafe extern "C" fn(*mut c_void) -> CUresult,
    pub cuMemcpyHtoDAsync:
        unsafe extern "C" fn(CUdeviceptr, *const c_void, usize, CUstream) -> CUresult,
    /// Synchronous device-to-host copy. Tests use this to read GPU buffers
    /// back to host for verification; production paths run async on a stream.
    pub cuMemcpyDtoH: unsafe extern "C" fn(*mut c_void, CUdeviceptr, usize) -> CUresult,
    pub cuGetErrorString: unsafe extern "C" fn(CUresult, *mut *const c_char) -> CUresult,
}

/// Resolve a C symbol name from the loaded library, dereference the resulting
/// `Symbol` into the raw function pointer, and propagate a useful error.
///
/// `*sym` is safe here: `Symbol::Target` is `Copy` for function-pointer types
/// and the pointer stays valid as long as the backing `Library` is alive —
/// which we ensure by storing `_lib` alongside the pointers.
macro_rules! load_fn {
    ($lib:expr, $name:literal) => {{
        let sym: libloading::Symbol<_> = unsafe {
            $lib.get(concat!($name, "\0").as_bytes())
        }
        .map_err(|e| {
            Error::CudaUnavailable(format!("resolving {}: {e}", $name))
        })?;
        *sym
    }};
}

impl CudaLib {
    /// Open `libcuda.so.1` and resolve every symbol we use. Returns
    /// `CudaUnavailable` if the library is absent (no driver installed, or
    /// CPU-only box) or if any symbol is missing (incompatible driver).
    ///
    /// Does *not* call `cuInit` — callers do that after wrapping.
    pub fn load() -> Result<Self> {
        // `libloading::Library::new` is `unsafe` because loading an arbitrary
        // shared library can execute initializer code. We trust the system's
        // `libcuda.so.1`.
        let lib = unsafe { Library::new("libcuda.so.1") }
            .map_err(|e| Error::CudaUnavailable(format!("dlopen libcuda.so.1: {e}")))?;

        Ok(Self {
            cuInit: load_fn!(&lib, "cuInit"),
            cuDeviceGetCount: load_fn!(&lib, "cuDeviceGetCount"),
            cuDeviceGet: load_fn!(&lib, "cuDeviceGet"),
            cuDeviceGetName: load_fn!(&lib, "cuDeviceGetName"),
            cuDeviceGetPCIBusId: load_fn!(&lib, "cuDeviceGetPCIBusId"),
            cuDevicePrimaryCtxRetain: load_fn!(&lib, "cuDevicePrimaryCtxRetain"),
            cuDevicePrimaryCtxRelease: load_fn!(&lib, "cuDevicePrimaryCtxRelease"),
            // Versioned suffix `_v2` matches the ABI the driver has exported
            // since CUDA 4.x. The unsuffixed symbols are deprecated shims.
            cuCtxPushCurrent: load_fn!(&lib, "cuCtxPushCurrent_v2"),
            cuCtxPopCurrent: load_fn!(&lib, "cuCtxPopCurrent_v2"),
            cuStreamCreate: load_fn!(&lib, "cuStreamCreate"),
            cuStreamDestroy: load_fn!(&lib, "cuStreamDestroy_v2"),
            cuStreamSynchronize: load_fn!(&lib, "cuStreamSynchronize"),
            cuEventCreate: load_fn!(&lib, "cuEventCreate"),
            cuEventDestroy: load_fn!(&lib, "cuEventDestroy_v2"),
            cuEventRecord: load_fn!(&lib, "cuEventRecord"),
            cuEventSynchronize: load_fn!(&lib, "cuEventSynchronize"),
            cuEventQuery: load_fn!(&lib, "cuEventQuery"),
            cuMemAlloc: load_fn!(&lib, "cuMemAlloc_v2"),
            cuMemFree: load_fn!(&lib, "cuMemFree_v2"),
            cuMemHostAlloc: load_fn!(&lib, "cuMemHostAlloc"),
            cuMemFreeHost: load_fn!(&lib, "cuMemFreeHost"),
            cuMemcpyHtoDAsync: load_fn!(&lib, "cuMemcpyHtoDAsync_v2"),
            cuMemcpyDtoH: load_fn!(&lib, "cuMemcpyDtoH_v2"),
            cuGetErrorString: load_fn!(&lib, "cuGetErrorString"),
            _lib: lib,
        })
    }
}

// SAFETY: Raw function pointers are trivially `Send + Sync`. The CUDA driver
// API is documented as thread-safe for the symbols we use here (device
// queries, context push/pop, stream/event ops, memory alloc/free/copy). The
// `libloading::Library` itself is `Send + Sync` on unix.
unsafe impl Send for CudaLib {}
unsafe impl Sync for CudaLib {}
