//! Minimal CUDA runtime API bound at runtime via `dlopen`.
//!
//! There is no link-time CUDA dependency: wheels build with zero CUDA
//! toolchain installed. At runtime we prefer the `libcudart` that torch (or
//! any other consumer) already mapped into the process (found by scanning
//! `/proc/self/maps`, reopened with `RTLD_NOLOAD`), falling back to a
//! by-name `dlopen`. When no runtime can be found, [`api`] returns `None`
//! and callers degrade gracefully.

use std::ffi::{c_char, c_int, c_uint, c_void, CStr, CString};
use std::sync::OnceLock;

/// `cudaError_t`. Zero is success.
pub type CudaError = c_int;

/// `cudaStream_t` / `cudaEvent_t` are opaque handles. They are stored as
/// `usize` so values are trivially `Send`/`Sync`; the CUDA runtime API is
/// documented thread-safe.
pub type Stream = usize;
/// See [`Stream`].
pub type Event = usize;

/// `cudaMemcpyHostToDevice`.
pub const MEMCPY_HOST_TO_DEVICE: c_int = 1;
/// `cudaEventDisableTiming`: cheapest event flavor, sync/query only.
pub const EVENT_DISABLE_TIMING: c_uint = 0x02;
/// `cudaStreamNonBlocking`: no implicit sync with the legacy default stream.
pub const STREAM_NON_BLOCKING: c_uint = 0x01;
/// `cudaHostAllocPortable`: pinned allocation usable from every device.
pub const HOST_ALLOC_PORTABLE: c_uint = 0x01;

type FnHostAlloc = unsafe extern "C" fn(*mut *mut c_void, usize, c_uint) -> CudaError;
type FnFreePtr = unsafe extern "C" fn(*mut c_void) -> CudaError;
type FnMemcpyAsync =
    unsafe extern "C" fn(*mut c_void, *const c_void, usize, c_int, *mut c_void) -> CudaError;
type FnStreamCreate = unsafe extern "C" fn(*mut *mut c_void, c_uint) -> CudaError;
type FnStreamWaitEvent = unsafe extern "C" fn(*mut c_void, *mut c_void, c_uint) -> CudaError;
type FnEventCreate = unsafe extern "C" fn(*mut *mut c_void, c_uint) -> CudaError;
type FnEventRecord = unsafe extern "C" fn(*mut c_void, *mut c_void) -> CudaError;
type FnHandleArg = unsafe extern "C" fn(*mut c_void) -> CudaError;
type FnSetDevice = unsafe extern "C" fn(c_int) -> CudaError;
type FnGetDevice = unsafe extern "C" fn(*mut c_int) -> CudaError;
type FnErrorString = unsafe extern "C" fn(CudaError) -> *const c_char;
type FnMalloc = unsafe extern "C" fn(*mut *mut c_void, usize) -> CudaError;
type FnMallocAsync = unsafe extern "C" fn(*mut *mut c_void, usize, *mut c_void) -> CudaError;
type FnFreeAsync = unsafe extern "C" fn(*mut c_void, *mut c_void) -> CudaError;
type FnGetDefaultMemPool = unsafe extern "C" fn(*mut *mut c_void, c_int) -> CudaError;
type FnMemPoolTrimTo = unsafe extern "C" fn(*mut c_void, usize) -> CudaError;
type FnMemPoolSetAttribute = unsafe extern "C" fn(*mut c_void, c_int, *mut c_void) -> CudaError;
type FnMemGetInfo = unsafe extern "C" fn(*mut usize, *mut usize) -> CudaError;

/// `cudaMemPoolAttrReleaseThreshold`.
const MEM_POOL_ATTR_RELEASE_THRESHOLD: c_int = 4;

/// Resolved CUDA runtime entry points.
pub struct CudaApi {
    host_alloc: FnHostAlloc,
    #[allow(dead_code)]
    free_host: FnFreePtr,
    malloc: FnMalloc,
    free: FnFreePtr,
    /// Stream-ordered allocator (CUDA >= 11.2); `None` on older runtimes,
    /// where [`alloc_device`](Self::alloc_device) falls back to `cudaMalloc`
    /// (synchronous, no pool — correct but slower to allocate/free).
    malloc_async: Option<FnMallocAsync>,
    free_async: Option<FnFreeAsync>,
    device_get_default_mem_pool: Option<FnGetDefaultMemPool>,
    mem_pool_trim_to: Option<FnMemPoolTrimTo>,
    mem_pool_set_attribute: Option<FnMemPoolSetAttribute>,
    memcpy_async: FnMemcpyAsync,
    stream_create_with_flags: FnStreamCreate,
    stream_wait_event: FnStreamWaitEvent,
    stream_synchronize: FnHandleArg,
    #[allow(dead_code)]
    stream_destroy: FnHandleArg,
    mem_get_info: FnMemGetInfo,
    event_create_with_flags: FnEventCreate,
    event_record: FnEventRecord,
    event_synchronize: FnHandleArg,
    event_destroy: FnHandleArg,
    set_device: FnSetDevice,
    get_device: FnGetDevice,
    get_error_string: FnErrorString,
}

impl CudaApi {
    fn check(&self, code: CudaError, what: &str) -> Result<(), String> {
        if code == 0 {
            return Ok(());
        }
        // SAFETY: cudaGetErrorString returns a static string for any code.
        let msg = unsafe {
            let p = (self.get_error_string)(code);
            if p.is_null() {
                "unknown error".to_string()
            } else {
                CStr::from_ptr(p).to_string_lossy().into_owned()
            }
        };
        Err(format!("{what} failed: {msg} ({code})"))
    }

    /// `cudaHostAlloc`: pinned host memory. Returns the host address.
    pub fn host_alloc(&self, len: usize, flags: c_uint) -> Result<usize, String> {
        let mut ptr: *mut c_void = std::ptr::null_mut();
        // SAFETY: FFI call with an out-pointer to a local.
        let code = unsafe { (self.host_alloc)(&mut ptr, len, flags) };
        self.check(code, "cudaHostAlloc")?;
        Ok(ptr as usize)
    }

    /// `cudaFreeHost`. Unused today only because the slab ring is
    /// process-lived by design; kept so a pool teardown can exist.
    #[allow(dead_code)]
    pub fn free_host(&self, ptr: usize) -> Result<(), String> {
        // SAFETY: `ptr` must come from `host_alloc`.
        let code = unsafe { (self.free_host)(ptr as *mut c_void) };
        self.check(code, "cudaFreeHost")
    }

    /// Stream-ordered device allocation on the current device:
    /// `cudaMallocAsync(len, stream)`, falling back to plain `cudaMalloc`
    /// on runtimes without the stream-ordered allocator (CUDA < 11.2).
    /// Returns the device address, usable in `stream` order.
    pub fn alloc_device(&self, len: usize, stream: Stream) -> Result<usize, String> {
        let mut ptr: *mut c_void = std::ptr::null_mut();
        if let Some(malloc_async) = self.malloc_async {
            // SAFETY: FFI call with an out-pointer to a local; `stream` is a
            // live stream on the current device.
            let code = unsafe { (malloc_async)(&mut ptr, len, stream as *mut c_void) };
            self.check(code, "cudaMallocAsync")?;
        } else {
            // SAFETY: FFI call with an out-pointer to a local.
            let code = unsafe { (self.malloc)(&mut ptr, len) };
            self.check(code, "cudaMalloc")?;
        }
        Ok(ptr as usize)
    }

    /// Free a [`alloc_device`](Self::alloc_device) pointer in `stream`
    /// order (`cudaFreeAsync`), or synchronously on old runtimes.
    pub fn free_device(&self, ptr: usize, stream: Stream) -> Result<(), String> {
        if let Some(free_async) = self.free_async {
            // SAFETY: `ptr` comes from `alloc_device` on this runtime.
            let code = unsafe { (free_async)(ptr as *mut c_void, stream as *mut c_void) };
            self.check(code, "cudaFreeAsync")
        } else {
            // SAFETY: `ptr` comes from `alloc_device`'s cudaMalloc branch.
            let code = unsafe { (self.free)(ptr as *mut c_void) };
            self.check(code, "cudaFree")
        }
    }

    fn default_pool(&self, device: c_int) -> Result<Option<*mut c_void>, String> {
        let Some(get_pool) = self.device_get_default_mem_pool else {
            return Ok(None);
        };
        let mut pool: *mut c_void = std::ptr::null_mut();
        // SAFETY: FFI call with an out-pointer to a local.
        let code = unsafe { (get_pool)(&mut pool, device) };
        self.check(code, "cudaDeviceGetDefaultMemPool")?;
        Ok(Some(pool))
    }

    /// Best-effort `cudaMemPoolTrimTo(default pool of device, 0)`: return
    /// the stream-ordered allocator's cached memory to the driver so other
    /// allocators (e.g. torch) can size against real free memory. No-op on
    /// runtimes without mempool symbols.
    pub fn trim_default_pool(&self, device: c_int) -> Result<(), String> {
        let (Some(pool), Some(trim)) = (self.default_pool(device)?, self.mem_pool_trim_to) else {
            return Ok(());
        };
        // SAFETY: `pool` is the live default pool handle.
        let code = unsafe { (trim)(pool, 0) };
        self.check(code, "cudaMemPoolTrimTo")
    }

    /// Best-effort: set the default pool's release threshold. `u64::MAX`
    /// keeps freed blocks cached across stream syncs (fast reallocation
    /// during a load); `0` restores release-at-sync. No-op on runtimes
    /// without mempool symbols.
    pub fn set_default_pool_release_threshold(
        &self,
        device: c_int,
        threshold: u64,
    ) -> Result<(), String> {
        let (Some(pool), Some(set_attr)) =
            (self.default_pool(device)?, self.mem_pool_set_attribute)
        else {
            return Ok(());
        };
        let mut value = threshold;
        // SAFETY: `pool` is live; the attribute value is a u64 read during
        // the call.
        let code = unsafe {
            (set_attr)(
                pool,
                MEM_POOL_ATTR_RELEASE_THRESHOLD,
                &mut value as *mut u64 as *mut c_void,
            )
        };
        self.check(code, "cudaMemPoolSetAttribute")
    }

    /// Async pinned-host → device copy on `stream`.
    ///
    /// # Safety contract (checked by callers)
    /// `dst` names `len` bytes of live device memory; `src` names `len`
    /// bytes of pinned host memory that stays valid until the copy's fence
    /// event is synchronized.
    pub fn memcpy_h2d_async(
        &self,
        dst: usize,
        src: usize,
        len: usize,
        stream: Stream,
    ) -> Result<(), String> {
        // SAFETY: per the documented caller contract above.
        let code = unsafe {
            (self.memcpy_async)(
                dst as *mut c_void,
                src as *const c_void,
                len,
                MEMCPY_HOST_TO_DEVICE,
                stream as *mut c_void,
            )
        };
        self.check(code, "cudaMemcpyAsync")
    }

    /// `cudaStreamCreateWithFlags` on the current device.
    pub fn stream_create(&self, flags: c_uint) -> Result<Stream, String> {
        let mut s: *mut c_void = std::ptr::null_mut();
        // SAFETY: FFI call with an out-pointer to a local.
        let code = unsafe { (self.stream_create_with_flags)(&mut s, flags) };
        self.check(code, "cudaStreamCreateWithFlags")?;
        Ok(s as Stream)
    }

    /// `cudaStreamWaitEvent`: make all future work on `stream` wait for
    /// `event` (as of its most recent record).
    pub fn stream_wait_event(&self, stream: Stream, event: Event) -> Result<(), String> {
        // SAFETY: both handles were created by this API and not destroyed.
        let code =
            unsafe { (self.stream_wait_event)(stream as *mut c_void, event as *mut c_void, 0) };
        self.check(code, "cudaStreamWaitEvent")
    }

    /// `cudaStreamSynchronize`: block the host until `stream` is empty.
    pub fn stream_synchronize(&self, stream: Stream) -> Result<(), String> {
        // SAFETY: `stream` was created by this API and not destroyed.
        let code = unsafe { (self.stream_synchronize)(stream as *mut c_void) };
        self.check(code, "cudaStreamSynchronize")
    }

    /// `cudaMemGetInfo` on the current device: `(free, total)` bytes.
    pub fn mem_get_info(&self) -> Result<(usize, usize), String> {
        let (mut free, mut total) = (0usize, 0usize);
        // SAFETY: FFI call with out-pointers to locals.
        let code = unsafe { (self.mem_get_info)(&mut free, &mut total) };
        self.check(code, "cudaMemGetInfo")?;
        Ok((free, total))
    }

    /// `cudaEventCreateWithFlags(cudaEventDisableTiming)` on the current device.
    pub fn event_create(&self) -> Result<Event, String> {
        let mut e: *mut c_void = std::ptr::null_mut();
        // SAFETY: FFI call with an out-pointer to a local.
        let code = unsafe { (self.event_create_with_flags)(&mut e, EVENT_DISABLE_TIMING) };
        self.check(code, "cudaEventCreateWithFlags")?;
        Ok(e as Event)
    }

    /// `cudaEventRecord`.
    pub fn event_record(&self, event: Event, stream: Stream) -> Result<(), String> {
        // SAFETY: both handles were created by this API and not destroyed.
        let code = unsafe { (self.event_record)(event as *mut c_void, stream as *mut c_void) };
        self.check(code, "cudaEventRecord")
    }

    /// `cudaEventSynchronize`: block the host until the event completes.
    pub fn event_synchronize(&self, event: Event) -> Result<(), String> {
        // SAFETY: `event` was created by this API and not destroyed.
        let code = unsafe { (self.event_synchronize)(event as *mut c_void) };
        self.check(code, "cudaEventSynchronize")
    }

    /// `cudaEventDestroy`.
    pub fn event_destroy(&self, event: Event) -> Result<(), String> {
        // SAFETY: `event` was created by this API and not destroyed.
        let code = unsafe { (self.event_destroy)(event as *mut c_void) };
        self.check(code, "cudaEventDestroy")
    }

    /// `cudaSetDevice` for the calling thread.
    pub fn set_device(&self, device: c_int) -> Result<(), String> {
        // SAFETY: plain FFI call.
        let code = unsafe { (self.set_device)(device) };
        self.check(code, "cudaSetDevice")
    }

    /// `cudaGetDevice` for the calling thread.
    pub fn get_device(&self) -> Result<c_int, String> {
        let mut d: c_int = 0;
        // SAFETY: FFI call with an out-pointer to a local.
        let code = unsafe { (self.get_device)(&mut d) };
        self.check(code, "cudaGetDevice")?;
        Ok(d)
    }

    /// Set `device` current for this thread, restoring the previous device
    /// when the guard drops.
    pub fn device_guard(&'static self, device: c_int) -> Result<DeviceGuard, String> {
        let prev = self.get_device()?;
        self.set_device(device)?;
        Ok(DeviceGuard {
            cuda: self,
            prev,
            changed: prev != device,
        })
    }
}

/// RAII restore of the calling thread's current CUDA device.
pub struct DeviceGuard {
    cuda: &'static CudaApi,
    prev: c_int,
    changed: bool,
}

impl Drop for DeviceGuard {
    fn drop(&mut self) {
        if self.changed {
            let _ = self.cuda.set_device(self.prev);
        }
    }
}

/// # Safety
///
/// `T` must be a `fn` pointer type; the symbol must match its ABI.
unsafe fn sym<T: Copy>(handle: *mut c_void, name: &str) -> Result<T, String> {
    let c_name = CString::new(name).map_err(|_| format!("bad symbol name {name}"))?;
    // SAFETY: `handle` is a live dlopen handle; dlsym is safe to call.
    let p = unsafe { libc::dlsym(handle, c_name.as_ptr()) };
    if p.is_null() {
        return Err(format!("symbol {name} not found in CUDA runtime"));
    }
    debug_assert_eq!(size_of::<T>(), size_of::<*mut c_void>());
    // SAFETY: caller contract; fn pointers are pointer-sized.
    Ok(unsafe { std::mem::transmute_copy::<*mut c_void, T>(&p) })
}

/// [`sym`] for optional entry points (newer-runtime features): `None` when
/// the symbol is absent instead of failing the whole binding.
///
/// # Safety
///
/// Same contract as [`sym`].
unsafe fn sym_opt<T: Copy>(handle: *mut c_void, name: &str) -> Option<T> {
    // SAFETY: forwarded caller contract.
    unsafe { sym::<T>(handle, name).ok() }
}

/// Find a `libcudart` already mapped into this process and reopen it with
/// `RTLD_NOLOAD` (never loads a second copy; just bumps the refcount).
#[cfg(target_os = "linux")]
fn find_loaded_cudart() -> *mut c_void {
    let Ok(maps) = std::fs::read_to_string("/proc/self/maps") else {
        return std::ptr::null_mut();
    };
    for line in maps.lines() {
        let Some(idx) = line.find('/') else { continue };
        let path = line[idx..].trim_end();
        if !path.contains("libcudart.so") {
            continue;
        }
        let Ok(c_path) = CString::new(path) else {
            continue;
        };
        // SAFETY: dlopen with a NUL-terminated path.
        let handle = unsafe {
            libc::dlopen(
                c_path.as_ptr(),
                libc::RTLD_NOLOAD | libc::RTLD_NOW | libc::RTLD_LOCAL,
            )
        };
        if !handle.is_null() {
            return handle;
        }
    }
    std::ptr::null_mut()
}

#[cfg(not(target_os = "linux"))]
fn find_loaded_cudart() -> *mut c_void {
    std::ptr::null_mut()
}

fn open_cudart() -> *mut c_void {
    let handle = find_loaded_cudart();
    if !handle.is_null() {
        return handle;
    }
    // Nothing loaded yet (e.g. torch not imported): try by name so the
    // dynamic linker search path decides.
    for name in [
        c"libcudart.so.13",
        c"libcudart.so.12",
        c"libcudart.so.11.0",
        c"libcudart.so",
    ] {
        // SAFETY: dlopen with a static NUL-terminated name.
        let handle = unsafe { libc::dlopen(name.as_ptr(), libc::RTLD_NOW | libc::RTLD_LOCAL) };
        if !handle.is_null() {
            return handle;
        }
    }
    std::ptr::null_mut()
}

fn load() -> Option<CudaApi> {
    let handle = open_cudart();
    if handle.is_null() {
        return None;
    }
    // SAFETY: each `T` matches the CUDA runtime prototype of the symbol.
    let api = unsafe {
        (|| -> Result<CudaApi, String> {
            Ok(CudaApi {
                host_alloc: sym::<FnHostAlloc>(handle, "cudaHostAlloc")?,
                free_host: sym::<FnFreePtr>(handle, "cudaFreeHost")?,
                malloc: sym::<FnMalloc>(handle, "cudaMalloc")?,
                free: sym::<FnFreePtr>(handle, "cudaFree")?,
                malloc_async: sym_opt::<FnMallocAsync>(handle, "cudaMallocAsync"),
                free_async: sym_opt::<FnFreeAsync>(handle, "cudaFreeAsync"),
                device_get_default_mem_pool: sym_opt::<FnGetDefaultMemPool>(
                    handle,
                    "cudaDeviceGetDefaultMemPool",
                ),
                mem_pool_trim_to: sym_opt::<FnMemPoolTrimTo>(handle, "cudaMemPoolTrimTo"),
                mem_pool_set_attribute: sym_opt::<FnMemPoolSetAttribute>(
                    handle,
                    "cudaMemPoolSetAttribute",
                ),
                memcpy_async: sym::<FnMemcpyAsync>(handle, "cudaMemcpyAsync")?,
                stream_create_with_flags: sym::<FnStreamCreate>(
                    handle,
                    "cudaStreamCreateWithFlags",
                )?,
                stream_wait_event: sym::<FnStreamWaitEvent>(handle, "cudaStreamWaitEvent")?,
                stream_synchronize: sym::<FnHandleArg>(handle, "cudaStreamSynchronize")?,
                stream_destroy: sym::<FnHandleArg>(handle, "cudaStreamDestroy")?,
                mem_get_info: sym::<FnMemGetInfo>(handle, "cudaMemGetInfo")?,
                event_create_with_flags: sym::<FnEventCreate>(handle, "cudaEventCreateWithFlags")?,
                event_record: sym::<FnEventRecord>(handle, "cudaEventRecord")?,
                event_synchronize: sym::<FnHandleArg>(handle, "cudaEventSynchronize")?,
                event_destroy: sym::<FnHandleArg>(handle, "cudaEventDestroy")?,
                set_device: sym::<FnSetDevice>(handle, "cudaSetDevice")?,
                get_device: sym::<FnGetDevice>(handle, "cudaGetDevice")?,
                get_error_string: sym::<FnErrorString>(handle, "cudaGetErrorString")?,
            })
        })()
    };
    api.ok()
}

static API: OnceLock<Option<CudaApi>> = OnceLock::new();

/// The process-wide CUDA runtime binding, or `None` when no usable
/// `libcudart` exists in (or can be loaded into) this process.
pub fn api() -> Option<&'static CudaApi> {
    API.get_or_init(load).as_ref()
}
