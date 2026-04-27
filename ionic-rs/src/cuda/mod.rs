//! Safe Rust wrappers over the CUDA driver API.
//!
//! Each handle (`CuContext`, `CuStream`, `CuEvent`, `DeviceBuf`, `PinnedBuf`)
//! owns a driver-side resource and releases it on `Drop`. Fallible methods
//! return `Result<_>`; errors carry the CUDA symbol and result code
//! (see `Error::Cuda`).
//!
//! **Context discipline.** Memory allocation, stream creation, and H2D copy
//! require the calling thread to have a current CUDA context. PyTorch
//! establishes the primary context for a device on its first call; we attach
//! to that same primary context rather than creating our own via
//! `cuCtxCreate` (that would split ownership and confuse the scheduler).
//! Use `CuContext::primary_retain(dev)` + `ctx.with_current(|| { ... })` to
//! scope push/pop pairs.

pub mod sys;

use std::ffi::{c_char, CStr};
use std::sync::OnceLock;

use sys::{CudaLib, CUcontext, CUdevice, CUdeviceptr, CUevent, CUresult, CUstream, CUDA_SUCCESS};

use crate::error::{Error, Result};

// ── Library access ──────────────────────────────────────────────────────

fn check(rc: CUresult, symbol: &'static str) -> Result<()> {
    if rc == CUDA_SUCCESS {
        Ok(())
    } else {
        Err(Error::Cuda {
            symbol,
            code: rc as i32,
        })
    }
}

/// Resolve and initialize the CUDA driver once, then hand out shared
/// references. `cuInit(0)` is called on first access — safe to call multiple
/// times per the driver's contract, but we only pay for it once.
pub fn lib() -> Result<&'static CudaLib> {
    // The OnceLock stores std::result::Result, not our crate Result alias —
    // the cached error is a flat String (cheaply cloneable) rather than the
    // full Error enum, so we can hand out `&'static CudaLib` references and
    // re-construct an Error on each failure path.
    static CUDA_LIB: OnceLock<std::result::Result<CudaLib, String>> = OnceLock::new();
    let res = CUDA_LIB.get_or_init(|| {
        // Two failure modes to collapse into one string: dlopen failure or
        // cuInit failure. The `OnceLock` value stores a flat `String` so the
        // error message is cheap to clone on subsequent calls.
        CudaLib::load()
            .and_then(|lib| {
                let rc = unsafe { (lib.cuInit)(0) };
                check(rc, "cuInit").map(|_| lib)
            })
            .map_err(|e| e.to_string())
    });
    res.as_ref()
        .map_err(|s| Error::CudaUnavailable(s.clone()))
}

// ── Device ──────────────────────────────────────────────────────────────

#[derive(Clone, Copy, Debug)]
pub struct CuDevice(CUdevice);

impl CuDevice {
    pub fn count() -> Result<i32> {
        let cuda = lib()?;
        let mut n: i32 = 0;
        check(unsafe { (cuda.cuDeviceGetCount)(&mut n) }, "cuDeviceGetCount")?;
        Ok(n)
    }

    pub fn get(ordinal: i32) -> Result<Self> {
        let cuda = lib()?;
        let mut dev: CUdevice = 0;
        check(
            unsafe { (cuda.cuDeviceGet)(&mut dev, ordinal) },
            "cuDeviceGet",
        )?;
        Ok(Self(dev))
    }

    pub fn name(self) -> Result<String> {
        let cuda = lib()?;
        let mut buf = [0 as c_char; 256];
        check(
            unsafe { (cuda.cuDeviceGetName)(buf.as_mut_ptr(), buf.len() as i32, self.0) },
            "cuDeviceGetName",
        )?;
        // SAFETY: driver writes a NUL-terminated string of length <= buf.len().
        let cstr = unsafe { CStr::from_ptr(buf.as_ptr()) };
        Ok(cstr.to_string_lossy().into_owned())
    }

    /// PCI bus ID in sysfs-friendly form (lowercase `DDDD:BB:DD.F`). The
    /// driver returns uppercase hex; we normalize because sysfs paths are
    /// case-sensitive and use lowercase.
    pub fn pci_bus_id(self) -> Result<String> {
        let cuda = lib()?;
        let mut buf = [0 as c_char; 32];
        check(
            unsafe { (cuda.cuDeviceGetPCIBusId)(buf.as_mut_ptr(), buf.len() as i32, self.0) },
            "cuDeviceGetPCIBusId",
        )?;
        let cstr = unsafe { CStr::from_ptr(buf.as_ptr()) };
        Ok(cstr.to_string_lossy().to_lowercase())
    }

    pub fn as_raw(self) -> CUdevice {
        self.0
    }
}

// ── Primary context ─────────────────────────────────────────────────────

/// RAII handle for a retained primary context. `Drop` releases the retention
/// so PyTorch's refcount on the same primary context isn't disturbed.
///
/// Use `with_current` to push/pop for the duration of a call scope. Push/pop
/// must balance — `with_current` enforces that even on panic, the pop runs.
pub struct CuContext {
    device: CuDevice,
    ctx: CUcontext,
}

// SAFETY: a primary context can be made current on multiple threads
// independently (each thread maintains its own context stack via
// push/pop). Sharing `&CuContext` across threads so each can call
// `with_current` is the documented use pattern.
unsafe impl Send for CuContext {}
unsafe impl Sync for CuContext {}

impl CuContext {
    pub fn primary_retain(device: CuDevice) -> Result<Self> {
        let cuda = lib()?;
        let mut ctx: CUcontext = std::ptr::null_mut();
        check(
            unsafe { (cuda.cuDevicePrimaryCtxRetain)(&mut ctx, device.as_raw()) },
            "cuDevicePrimaryCtxRetain",
        )?;
        Ok(Self { device, ctx })
    }

    /// Run `f` with this context current on the calling thread. The closure
    /// returns a `Result`; the outer method returns the same,
    /// flattened. The context is popped before this returns, even on panic —
    /// the `PopGuard` Drop pops unconditionally.
    pub fn with_current<F, R>(&self, f: F) -> Result<R>
    where
        F: FnOnce() -> Result<R>,
    {
        let cuda = lib()?;
        check(
            unsafe { (cuda.cuCtxPushCurrent)(self.ctx) },
            "cuCtxPushCurrent_v2",
        )?;
        struct PopGuard;
        impl Drop for PopGuard {
            fn drop(&mut self) {
                if let Ok(cuda) = lib() {
                    let mut _out: CUcontext = std::ptr::null_mut();
                    // Swallow the pop's result: a pop failure here is both
                    // extremely unlikely and has no sensible recovery path
                    // from a destructor.
                    unsafe { (cuda.cuCtxPopCurrent)(&mut _out) };
                }
            }
        }
        let _guard = PopGuard;
        f()
    }

    pub fn device(&self) -> CuDevice {
        self.device
    }
}

impl Drop for CuContext {
    fn drop(&mut self) {
        if let Ok(cuda) = lib() {
            // Release, not destroy — this only decrements the driver's
            // refcount on the primary context. PyTorch's retention keeps
            // it alive.
            unsafe { (cuda.cuDevicePrimaryCtxRelease)(self.device.as_raw()) };
        }
    }
}

// ── Stream ──────────────────────────────────────────────────────────────

pub struct CuStream {
    stream: CUstream,
}

// SAFETY: CUDA driver API documents stream-side ops (`cuMemcpyHtoDAsync`,
// `cuStreamSynchronize`, `cuEventRecord(_, stream)`) as thread-safe. The
// `CUstream` handle is just an opaque pointer the driver looks up; nothing
// in our wrapper holds thread-affine state.
unsafe impl Send for CuStream {}
unsafe impl Sync for CuStream {}

impl CuStream {
    /// Create a non-blocking stream. Non-blocking means it does not
    /// implicitly serialize with the legacy default stream — essential for
    /// overlap with PyTorch's compute streams and with other pipeline
    /// operations.
    pub fn new() -> Result<Self> {
        let cuda = lib()?;
        let mut s: CUstream = std::ptr::null_mut();
        check(
            unsafe { (cuda.cuStreamCreate)(&mut s, sys::CU_STREAM_NON_BLOCKING) },
            "cuStreamCreate",
        )?;
        Ok(Self { stream: s })
    }

    pub fn synchronize(&self) -> Result<()> {
        let cuda = lib()?;
        check(
            unsafe { (cuda.cuStreamSynchronize)(self.stream) },
            "cuStreamSynchronize",
        )
    }

    pub fn as_raw(&self) -> CUstream {
        self.stream
    }
}

impl Drop for CuStream {
    fn drop(&mut self) {
        if let Ok(cuda) = lib() {
            unsafe { (cuda.cuStreamDestroy)(self.stream) };
        }
    }
}

// ── Event ───────────────────────────────────────────────────────────────

pub struct CuEvent {
    event: CUevent,
}

// SAFETY: same rationale as CuStream — `cuEventRecord`, `cuEventQuery`,
// `cuEventSynchronize`, `cuEventDestroy` are all documented thread-safe by
// the CUDA driver. Two-thread pipeline records on the worker thread and
// the events live for the lifetime of the pipeline.
unsafe impl Send for CuEvent {}
unsafe impl Sync for CuEvent {}

impl CuEvent {
    /// Timing-disabled event. We only use these for happens-before fences,
    /// not for profiling — timing-disabled events are cheaper to record.
    pub fn new() -> Result<Self> {
        let cuda = lib()?;
        let mut e: CUevent = std::ptr::null_mut();
        check(
            unsafe { (cuda.cuEventCreate)(&mut e, sys::CU_EVENT_DISABLE_TIMING) },
            "cuEventCreate",
        )?;
        Ok(Self { event: e })
    }

    pub fn record(&self, stream: &CuStream) -> Result<()> {
        let cuda = lib()?;
        check(
            unsafe { (cuda.cuEventRecord)(self.event, stream.as_raw()) },
            "cuEventRecord",
        )
    }

    pub fn synchronize(&self) -> Result<()> {
        let cuda = lib()?;
        check(
            unsafe { (cuda.cuEventSynchronize)(self.event) },
            "cuEventSynchronize",
        )
    }

    /// Non-blocking status check. Returns `Ok(true)` if all work captured
    /// by the event has completed, `Ok(false)` if any is still in flight,
    /// `Err(...)` for any other CUDA error. The async pipeline polls this
    /// to recycle slots without ever calling `cuStreamSynchronize`.
    pub fn query(&self) -> Result<bool> {
        let cuda = lib()?;
        let rc = unsafe { (cuda.cuEventQuery)(self.event) };
        match rc {
            sys::CUDA_SUCCESS => Ok(true),
            sys::CUDA_ERROR_NOT_READY => Ok(false),
            _ => Err(Error::Cuda {
                symbol: "cuEventQuery",
                code: rc as i32,
            }),
        }
    }
}

impl Drop for CuEvent {
    fn drop(&mut self) {
        if let Ok(cuda) = lib() {
            unsafe { (cuda.cuEventDestroy)(self.event) };
        }
    }
}

// ── Device memory ───────────────────────────────────────────────────────

/// Owned device allocation. `Drop` calls `cuMemFree_v2`.
pub struct DeviceBuf {
    ptr: CUdeviceptr,
    bytes: usize,
}

impl DeviceBuf {
    pub fn alloc(bytes: usize) -> Result<Self> {
        let cuda = lib()?;
        let mut ptr: CUdeviceptr = 0;
        check(
            unsafe { (cuda.cuMemAlloc)(&mut ptr, bytes) },
            "cuMemAlloc_v2",
        )?;
        Ok(Self { ptr, bytes })
    }

    pub fn as_device_ptr(&self) -> CUdeviceptr {
        self.ptr
    }

    pub fn len(&self) -> usize {
        self.bytes
    }
}

impl Drop for DeviceBuf {
    fn drop(&mut self) {
        if let Ok(cuda) = lib() {
            unsafe { (cuda.cuMemFree)(self.ptr) };
        }
    }
}

// ── Pinned host memory ──────────────────────────────────────────────────

/// Owned page-locked host allocation. Required for true async `cuMemcpyH2D`
/// at full PCIe bandwidth — pageable host memory silently stages through the
/// driver's own pinned bounce buffer, one chunk at a time, and serializes.
pub struct PinnedBuf {
    ptr: *mut u8,
    bytes: usize,
}

// SAFETY: pinned host memory is just `malloc`-style memory with a page-lock
// attribute. Access is thread-safe modulo ordinary race conditions (no
// driver-side mutation).
unsafe impl Send for PinnedBuf {}
unsafe impl Sync for PinnedBuf {}

impl PinnedBuf {
    pub fn alloc(bytes: usize) -> Result<Self> {
        let cuda = lib()?;
        let mut ptr: *mut std::ffi::c_void = std::ptr::null_mut();
        check(
            unsafe {
                (cuda.cuMemHostAlloc)(&mut ptr, bytes, sys::CU_MEMHOSTALLOC_PORTABLE)
            },
            "cuMemHostAlloc",
        )?;
        Ok(Self {
            ptr: ptr as *mut u8,
            bytes,
        })
    }

    pub fn as_ptr(&self) -> *const u8 {
        self.ptr as *const u8
    }

    pub fn as_mut_ptr(&self) -> *mut u8 {
        self.ptr
    }

    /// Mutable slice view for CPU-side staging writes. The driver does not
    /// touch this memory asynchronously unless we hand a pointer into it to
    /// `cuMemcpyHtoDAsync`; callers must not mutate the slice while such a
    /// copy is in flight.
    pub fn as_mut_slice(&mut self) -> &mut [u8] {
        // SAFETY: we own the allocation; bytes is the exact size returned
        // by cuMemHostAlloc.
        unsafe { std::slice::from_raw_parts_mut(self.ptr, self.bytes) }
    }

    /// Immutable slice view of the pinned region.
    pub fn as_slice(&self) -> &[u8] {
        // SAFETY: we own the allocation; bytes is the exact size returned
        // by cuMemHostAlloc.
        unsafe { std::slice::from_raw_parts(self.ptr, self.bytes) }
    }

    pub fn len(&self) -> usize {
        self.bytes
    }
}

impl Drop for PinnedBuf {
    fn drop(&mut self) {
        if let Ok(cuda) = lib() {
            unsafe { (cuda.cuMemFreeHost)(self.ptr as *mut std::ffi::c_void) };
        }
    }
}

// ── Transfer ────────────────────────────────────────────────────────────

/// Async H2D memcpy enqueued on `stream`. Returns immediately; the caller
/// must synchronize (on the stream or via an event) before reading the
/// destination from another context. The source must remain valid until the
/// copy completes — pin it or hold it via `PinnedBuf`.
///
/// Context must be current on the calling thread (wrap in `with_current`).
pub fn memcpy_h2d_async(
    dst: CUdeviceptr,
    src: *const u8,
    bytes: usize,
    stream: &CuStream,
) -> Result<()> {
    let cuda = lib()?;
    check(
        unsafe {
            (cuda.cuMemcpyHtoDAsync)(dst, src as *const std::ffi::c_void, bytes, stream.as_raw())
        },
        "cuMemcpyHtoDAsync_v2",
    )
}

/// Synchronous D2H copy. Used by tests to verify GPU contents; the
/// production prefetch path doesn't read back from the device.
///
/// Context must be current on the calling thread (wrap in `with_current`).
///
/// # Safety
///
/// `dst` must point to writable host memory of at least `bytes` length, and
/// `src` must be a valid CUdeviceptr with at least `bytes` reachable.
pub unsafe fn memcpy_d2h(dst: *mut u8, src: CUdeviceptr, bytes: usize) -> Result<()> {
    let cuda = lib()?;
    check(
        unsafe { (cuda.cuMemcpyDtoH)(dst as *mut std::ffi::c_void, src, bytes) },
        "cuMemcpyDtoH_v2",
    )
}
