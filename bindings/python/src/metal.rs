//! Minimal Metal allocator for host-shared MTLBuffers (Apple silicon UMA).
//!
//! Shared-mode buffers are CPU- and GPU-visible: writing through
//! [`MTLBuffer::contents_ptr`] is observable from the GPU after a normal
//! command-buffer commit (no `didModifyRange:` needed on Apple silicon).
//!
//! Note: this module's `MTLBuffer` is the Rust-side wrapper. The underlying
//! Objective-C protocol from `objc2-metal` is aliased as `RawMTLBuffer`
//! to keep both names visible and disambiguate raw vs. managed flavor.

use std::error::Error;
use std::ffi::c_void;
use std::fmt;
use std::sync::OnceLock;

use objc2::rc::Retained;
use objc2::runtime::ProtocolObject;
use objc2_metal::{
    MTLBuffer as RawMTLBuffer, MTLCreateSystemDefaultDevice, MTLDevice, MTLResourceOptions,
};

#[derive(Debug)]
pub enum MetalError {
    /// `MTLDevice::newBufferWithLength` returned nil for a request of N bytes.
    Allocation(usize),
}

impl fmt::Display for MetalError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            MetalError::Allocation(nbytes) => {
                write!(
                    f,
                    "MTLDevice::newBufferWithLength failed for {nbytes} bytes"
                )
            }
        }
    }
}

impl Error for MetalError {}

static DEVICE: OnceLock<DeviceHandle> = OnceLock::new();

struct DeviceHandle(Retained<ProtocolObject<dyn MTLDevice>>);

// MTLDevice is documented thread-safe.
unsafe impl Send for DeviceHandle {}
unsafe impl Sync for DeviceHandle {}

fn device() -> &'static ProtocolObject<dyn MTLDevice> {
    &DEVICE
        .get_or_init(|| {
            let dev =
                MTLCreateSystemDefaultDevice().expect("MTLCreateSystemDefaultDevice returned nil");
            DeviceHandle(dev)
        })
        .0
}

pub struct MTLBuffer {
    buf: Retained<ProtocolObject<dyn RawMTLBuffer>>,
    nbytes: usize,
    contents: *mut c_void,
}

// MTLBuffer is documented thread-safe; disjoint writes through the host
// alias are sound from multiple threads.
unsafe impl Send for MTLBuffer {}
unsafe impl Sync for MTLBuffer {}

impl MTLBuffer {
    /// Allocates a Shared-mode buffer of `nbytes`. A zero-byte request is
    /// clamped to a 1-byte allocation so it still yields a valid `MTLBuffer`
    /// to hand off; the wrapper keeps the original `nbytes` (0), and the
    /// DLPack shape carries the zero dim, so `numel` stays 0 for the consumer.
    pub fn alloc_shared(nbytes: usize) -> Result<Self, MetalError> {
        let dev = device();
        let len = nbytes.max(1);
        let buf = dev
            .newBufferWithLength_options(len, MTLResourceOptions::StorageModeShared)
            .ok_or(MetalError::Allocation(nbytes))?;
        let contents = buf.contents().as_ptr();
        Ok(Self {
            buf,
            nbytes,
            contents,
        })
    }

    pub fn contents_ptr(&self) -> *mut c_void {
        self.contents
    }

    /// Host-visible bytes as a mutable slice, for filling the buffer before
    /// it is handed to a framework. `&mut self` rules out aliasing access.
    pub fn as_mut_slice(&mut self) -> &mut [u8] {
        // SAFETY: `contents` is the host-coherent pointer of a Shared-mode
        // MTLBuffer allocated for at least `nbytes` (clamped to 1 internally),
        // valid for the buffer's lifetime.
        unsafe { std::slice::from_raw_parts_mut(self.contents as *mut u8, self.nbytes) }
    }

    /// The Objective-C buffer pointer (`id<MTLBuffer>`). PyTorch's MPS
    /// `from_dlpack` expects this (not the `contents` pointer) in
    /// `DLTensor.data`, since the MPS allocator tracks buffers by ID.
    pub fn as_metal_id_ptr(&self) -> *mut c_void {
        &*self.buf as *const ProtocolObject<dyn RawMTLBuffer> as *mut c_void
    }

    #[allow(dead_code)]
    pub fn nbytes(&self) -> usize {
        self.nbytes
    }
}
