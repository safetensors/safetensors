use std::{
    ffi::{c_char, c_int, c_uint, c_void, CStr, CString},
    sync::OnceLock,
};

pub type Stream = *mut c_void;
pub type Event = *mut c_void;
pub type MemPool = *mut c_void;
type Err = c_int;

const CUDA_MEMCPY_HOST_TO_DEVICE: c_int = 1;
const CUDA_EVENT_DISABLE_TIMING: c_uint = 0x2;
const CUDA_STREAM_NON_BLOCKING: c_uint = 0x1;
const CUDA_MEMPOOL_ATTR_RELEASE_THRESHOLD: c_int = 4;

macro_rules! cuda_fns {
    ($( $name:ident : fn( $($arg:ty),*) -> Err),+ $(,)?) => {
        #[allow(non_snake_case)]
        struct Fns { $( $name: unsafe extern "C" fn($($arg),*) -> Err, )+ }
        impl Fns {
            unsafe fn load(h: *mut c_void) -> Option<Self> {
                Some(Self {
                    $( $name: {
                        let sym = CString::new(stringify!($name)).unwrap();
                        let p = libc::dlsym(h, sym.as_ptr());
                        if p.is_null() { return None; }
                        std::mem::transmute::<*mut c_void, unsafe extern "C" fn($($arg),*) -> Err>(p)
                    },)+
                })
            }
        }
    };
}

cuda_fns! {
    cudaHostAlloc:               fn(*mut *mut c_void, usize, c_uint) -> Err,
    cudaFreeHost:                fn(*mut c_void) -> Err,
    cudaMallocAsync:             fn(*mut *mut c_void, usize, Stream) -> Err,
    cudaFreeAsync:               fn(*mut c_void, Stream) -> Err,
    cudaMemcpyAsync:             fn(*mut c_void, *const c_void, usize, c_int, Stream) -> Err,
    cudaStreamCreateWithFlags:   fn(*mut Stream, c_uint) -> Err,
    cudaStreamDestroy:           fn(Stream) -> Err,
    cudaStreamWaitEvent:         fn(Stream, Event, c_uint) -> Err,
    cudaEventCreateWithFlags:    fn(*mut Event, c_uint) -> Err,
    cudaEventRecord:             fn(Event, Stream) -> Err,
    cudaEventQuery:              fn(Event) -> Err,
    cudaEventSynchronize:        fn(Event) -> Err,
    cudaEventDestroy:            fn(Event) -> Err,
    cudaSetDevice:               fn(c_int) -> Err,
    cudaGetDevice:               fn(*mut c_int) -> Err,
    cudaMemGetInfo:              fn(*mut usize, *mut usize) -> Err,
    cudaDeviceGetDefaultMemPool: fn(*mut MemPool, c_int) -> Err,
    cudaMemPoolSetAttribute:     fn(MemPool, c_int, *mut c_void) -> Err,
    cudaMemPoolTrimTo:           fn(MemPool, usize) -> Err,
}

type GetErrorString = unsafe extern "C" fn(Err) -> *const c_char;

#[derive(Debug)]
pub struct CudaError {
    pub code: i32,
    pub msg: String,
}

impl std::fmt::Display for CudaError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "cuda error {}: {}", self.code, self.msg)
    }
}

impl std::error::Error for CudaError {}

pub struct CudaApi {
    f: Fns,
    err_str: GetErrorString,
}

// SAFETY: immutable fn pointers + CUDA runtime API is thread safe
unsafe impl Send for CudaApi {}
unsafe impl Sync for CudaApi {}

impl CudaApi {
    #[inline]
    fn check(&self, code: Err) -> Result<(), CudaError> {
        if code == 0 {
            return Ok(());
        }
        let msg = unsafe { CStr::from_ptr((self.err_str)(code)) }
            .to_string_lossy()
            .into_owned();
        Err(CudaError { code, msg })
    }
}

impl CudaApi {
    pub fn host_alloc(&self, len: usize) -> Result<*mut u8, CudaError> {
        let mut ptr = std::ptr::null_mut();
        self.check(unsafe { (self.f.cudaHostAlloc)(&mut ptr, len, 0) })?;
        Ok(ptr as *mut u8)
    }

    pub fn free_host(&self, ptr: *mut u8) -> Result<(), CudaError> {
        self.check(unsafe { (self.f.cudaFreeHost)(ptr as *mut c_void) })
    }

    pub fn malloc_async(&self, len: usize, s: Stream) -> Result<u64, CudaError> {
        let mut ptr = std::ptr::null_mut();
        self.check(unsafe { (self.f.cudaMallocAsync)(&mut ptr, len, s) })?;
        Ok(ptr as u64)
    }

    pub fn free_async(&self, ptr: u64, s: Stream) -> Result<(), CudaError> {
        self.check(unsafe { (self.f.cudaFreeAsync)(ptr as *mut c_void, s) })
    }

    pub fn memcpy_h2d_async(
        &self,
        dst: u64,
        src: *const u8,
        len: usize,
        s: Stream,
    ) -> Result<(), CudaError> {
        self.check(unsafe {
            (self.f.cudaMemcpyAsync)(
                dst as *mut c_void,
                src as *const c_void,
                len,
                CUDA_MEMCPY_HOST_TO_DEVICE,
                s,
            )
        })
    }

    pub fn stream_create(&self) -> Result<Stream, CudaError> {
        let mut s = std::ptr::null_mut();
        self.check(unsafe {
            (self.f.cudaStreamCreateWithFlags)(&mut s, CUDA_STREAM_NON_BLOCKING)
        })?;
        Ok(s)
    }

    pub fn stream_destroy(&self, s: Stream) -> Result<(), CudaError> {
        self.check(unsafe { (self.f.cudaStreamDestroy)(s) })
    }

    pub fn event_create(&self) -> Result<Event, CudaError> {
        let mut e = std::ptr::null_mut();
        self.check(unsafe {
            (self.f.cudaEventCreateWithFlags)(&mut e, CUDA_EVENT_DISABLE_TIMING)
        })?;
        Ok(e)
    }

    pub fn event_record(&self, e: Event, s: Stream) -> Result<(), CudaError> {
        self.check(unsafe { (self.f.cudaEventRecord)(e, s) })
    }

    pub fn event_query(&self, e: Event) -> Result<bool, CudaError> {
        match unsafe { (self.f.cudaEventQuery)(e) } {
            0 => Ok(true),
            600 => Ok(false),
            c => {
                self.check(c)?;
                unreachable!()
            }
        }
    }

    pub fn event_sync(&self, e: Event) -> Result<(), CudaError> {
        self.check(unsafe { (self.f.cudaEventSynchronize)(e) })
    }

    pub fn event_destroy(&self, e: Event) -> Result<(), CudaError> {
        self.check(unsafe { (self.f.cudaEventDestroy)(e) })
    }

    pub fn stream_wait_event(&self, s: Stream, e: Event) -> Result<(), CudaError> {
        self.check(unsafe { (self.f.cudaStreamWaitEvent)(s, e, 0) })
    }

    pub fn set_device(&self, d: i32) -> Result<(), CudaError> {
        self.check(unsafe { (self.f.cudaSetDevice)(d) })
    }

    pub fn device_guard(&'static self, d: i32) -> Result<DeviceGuard, CudaError> {
        let mut prev = 0;
        self.check(unsafe { (self.f.cudaGetDevice)(&mut prev) })?;
        self.set_device(d)?;
        Ok(DeviceGuard { api: self, prev })
    }

    pub fn mem_get_info(&self) -> Result<(usize, usize), CudaError> {
        let mut free = 0;
        let mut total = 0;
        self.check(unsafe { (self.f.cudaMemGetInfo)(&mut free, &mut total) })?;
        Ok((free, total))
    }

    pub fn pool_set_release_threshold(&self, device: i32, bytes: u64) -> Result<(), CudaError> {
        let mut pool = std::ptr::null_mut();
        self.check(unsafe { (self.f.cudaDeviceGetDefaultMemPool)(&mut pool, device) })?;
        let mut v = bytes;
        self.check(unsafe {
            (self.f.cudaMemPoolSetAttribute)(
                pool,
                CUDA_MEMPOOL_ATTR_RELEASE_THRESHOLD,
                &mut v as *mut u64 as *mut c_void,
            )
        })
    }

    pub fn pool_trim(&self, device: i32) -> Result<(), CudaError> {
        let mut pool = std::ptr::null_mut();
        self.check(unsafe { (self.f.cudaDeviceGetDefaultMemPool)(&mut pool, device) })?;
        self.check(unsafe { (self.f.cudaMemPoolTrimTo)(pool, 0) })
    }
}

pub struct DeviceGuard {
    api: &'static CudaApi,
    prev: c_int,
}

impl Drop for DeviceGuard {
    fn drop(&mut self) {
        let _ = self.api.set_device(self.prev);
    }
}

#[cfg(target_os = "linux")]
fn find_loaded_cudart() -> Option<CString> {
    let maps = std::fs::read_to_string("/proc/self/maps").ok()?;
    maps.lines()
        .filter_map(|l| l.split_whitespace().last())
        .find(|p| p.contains("libcudart.so"))
        .and_then(|p| CString::new(p).ok())
}

#[cfg(not(target_os = "linux"))]
fn find_loaded_cudart() -> Option<CString> {
    None
}

fn open_cudart() -> *mut c_void {
    unsafe {
        if let Some(path) = find_loaded_cudart() {
            let h = libc::dlopen(path.as_ptr(), libc::RTLD_NOW | libc::RTLD_NOLOAD);
            if !h.is_null() {
                return h;
            }
        }
        for name in ["libcudart.so.13", "libcudart.so.12", "libcudart.so"] {
            let c = CString::new(name).unwrap();
            let h = libc::dlopen(c.as_ptr(), libc::RTLD_NOW);
            if !h.is_null() {
                return h;
            }
        }
        std::ptr::null_mut()
    }
}

pub fn api() -> Option<&'static CudaApi> {
    static API: OnceLock<Option<CudaApi>> = OnceLock::new();
    API.get_or_init(|| unsafe {
        let h = open_cudart();
        if h.is_null() {
            return None;
        }
        let f = Fns::load(h)?;
        let sym = CString::new("cudaGetErrorString").unwrap();
        let p = libc::dlsym(h, sym.as_ptr());
        if p.is_null() {
            return None;
        }
        Some(CudaApi {
            f,
            err_str: std::mem::transmute::<*mut c_void, GetErrorString>(p),
        })
    })
    .as_ref()
}

#[cfg(test)]
mod test {
    use super::{api, CudaApi};

    fn cuda_or_skip() -> Option<&'static CudaApi> {
        match api() {
            Some(api) => Some(api),
            None => {
                if std::env::var_os("ENFORCE_CUDA_TEST").is_some() {
                    panic!("ENFORCE_CUDA_TEST is set but could not load CUDA runtime");
                }
                eprintln!("skipped: could not load CUDA runtime");
                None
            }
        }
    }

    #[test]
    fn test_idempotent_api_load() {
        let a = api();
        let b = api();
        assert_eq!(a.is_some(), b.is_some())
    }

    #[test]
    fn test_gpu_roundtrip() {
        let Some(api) = cuda_or_skip() else { return };
        let _g = api.device_guard(0).unwrap();
        let s = api.stream_create().unwrap();
        let (free0, _) = api.mem_get_info().unwrap();

        let host = api.host_alloc(4096).unwrap();
        unsafe { std::slice::from_raw_parts_mut(host, 4096) }.fill(0xAB);
        let dev = api.malloc_async(4096, s).unwrap();
        api.memcpy_h2d_async(dev, host, 4096, s).unwrap();
        let e = api.event_create().unwrap();
        api.event_record(e, s).unwrap();
        api.event_sync(e).unwrap();
        assert!(api.event_query(e).unwrap());

        api.free_async(dev, s).unwrap();
        api.event_record(e, s).unwrap();
        api.event_sync(e).unwrap();
        api.pool_trim(0).unwrap();

        let (free1, _) = api.mem_get_info().unwrap();
        assert!(
            free1 + (16 << 20) >= free0,
            "pool retained memory after trim"
        );

        api.stream_wait_event(s, e).unwrap();
        api.free_host(host).unwrap();
    }
}
