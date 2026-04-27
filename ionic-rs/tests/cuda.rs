//! End-to-end tests for the CUDA driver wrapper layer.
//!
//! Skips gracefully when `libcuda.so.1` isn't loadable (CPU-only boxes,
//! containers without driver mounts) — same behavior the pytest version
//! had, just done in pure Rust.

use ionic_rs::cuda;

/// Returns `Some(count)` when the CUDA driver is loadable AND visible
/// devices > 0, else `None` to signal "skip this test".
fn cuda_available_count() -> Option<i32> {
    match cuda::CuDevice::count() {
        Ok(n) if n > 0 => Some(n),
        _ => None,
    }
}

#[test]
fn enumerate_devices_returns_metadata() {
    let Some(count) = cuda_available_count() else {
        eprintln!("skipping: libcuda.so.1 not loadable or no devices");
        return;
    };
    for ordinal in 0..count {
        let dev = cuda::CuDevice::get(ordinal).expect("CuDevice::get");
        let name = dev.name().expect("name");
        let bdf = dev.pci_bus_id().expect("pci_bus_id");
        assert!(!name.is_empty(), "device name should not be empty");
        // Driver-returned BDF, lowercased by our wrapper for sysfs use.
        assert!(
            bdf.chars().all(|c| c.is_ascii_hexdigit() || c == ':' || c == '.'),
            "unexpected pci_bus_id: {bdf}"
        );
        assert_eq!(bdf, bdf.to_lowercase(), "pci_bus_id must be lowercase");
    }
}

#[test]
fn ctx_smoke_roundtrip() {
    // Retain primary context, push, allocate pinned + device, H2D copy on
    // a non-blocking stream, record + sync event, drop everything in order.
    // The whole contract of the wrapper layer in one test.
    let Some(_) = cuda_available_count() else {
        eprintln!("skipping: libcuda.so.1 not loadable or no devices");
        return;
    };
    const BYTES: usize = 4096;

    let dev = cuda::CuDevice::get(0).expect("CuDevice::get");
    let ctx = cuda::CuContext::primary_retain(dev).expect("primary_retain");
    ctx.with_current(|| {
        let stream = cuda::CuStream::new()?;
        let event = cuda::CuEvent::new()?;
        let mut pinned = cuda::PinnedBuf::alloc(BYTES)?;
        let dev_buf = cuda::DeviceBuf::alloc(BYTES)?;
        for (i, b) in pinned.as_mut_slice().iter_mut().enumerate() {
            *b = (i & 0xFF) as u8;
        }
        cuda::memcpy_h2d_async(dev_buf.as_device_ptr(), pinned.as_ptr(), BYTES, &stream)?;
        event.record(&stream)?;
        event.synchronize()?;
        Ok(())
    })
    .expect("ctx work");
}

#[test]
fn bad_ordinal_errors() {
    let Some(count) = cuda_available_count() else {
        eprintln!("skipping: libcuda.so.1 not loadable or no devices");
        return;
    };
    let bad = count + 10;
    assert!(
        cuda::CuDevice::get(bad).is_err(),
        "expected error for ordinal {bad}"
    );
}
