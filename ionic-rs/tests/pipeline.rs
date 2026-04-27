//! End-to-end CUDA pipeline test: write a deterministic file, run it
//! through `CudaPipeline` into a device buffer, copy back via D2H, compare.
//!
//! Skips when `libcuda.so.1` isn't loadable (CPU-only boxes).

use std::io::Write;

use ionic_rs::cuda::{self, CuContext, CuDevice, DeviceBuf};
use ionic_rs::iouring::LogicalSegment;
use ionic_rs::pipeline::CudaPipeline;
use tempfile::NamedTempFile;

const FOUR_MB: usize = 4 * 1024 * 1024;

fn cuda_available() -> bool {
    matches!(cuda::CuDevice::count(), Ok(n) if n > 0)
}

fn write_pattern_file(bytes: usize) -> (NamedTempFile, Vec<u8>) {
    let mut data = vec![0u8; bytes];
    let mut state: u32 = 0;
    for byte in data.iter_mut() {
        state = state.wrapping_mul(1103515245).wrapping_add(12345);
        *byte = (state & 0xFF) as u8;
    }
    let mut f = NamedTempFile::new().expect("tmpfile");
    f.write_all(&data).expect("write");
    f.flush().expect("flush");
    (f, data)
}

#[test]
fn full_file_through_pipeline_matches_source() {
    if !cuda_available() {
        eprintln!("skipping: libcuda.so.1 not loadable or no devices");
        return;
    }
    let (f, expected) = write_pattern_file(FOUR_MB);

    let dev = CuDevice::get(0).expect("CuDevice::get");
    let ctx = CuContext::primary_retain(dev).expect("primary_retain");
    let mut pipeline = CudaPipeline::new(0, 64, 128 * 1024).expect("CudaPipeline::new");
    let fd_idx = pipeline.register_file(f.path()).expect("register_file");

    // Allocate a single contiguous device buffer to receive the whole file.
    // The pipeline's `run` interprets `dst_offset` as a CUdeviceptr — so we
    // hand it the device buffer's base address as the segment's destination.
    let dev_buf = ctx
        .with_current(|| DeviceBuf::alloc(FOUR_MB))
        .expect("DeviceBuf::alloc");

    let segments = vec![LogicalSegment {
        fd_idx,
        from: 0,
        to: FOUR_MB as u64,
        dst_offset: dev_buf.as_device_ptr() as usize,
        user_data: 0,
    }];
    pipeline.run(&segments).expect("pipeline.run");

    // Copy back and verify.
    let mut got = vec![0u8; FOUR_MB];
    ctx.with_current(|| {
        // SAFETY: got is a mutable Vec sized to FOUR_MB, dev_buf has at least
        // FOUR_MB allocated.
        unsafe { cuda::memcpy_d2h(got.as_mut_ptr(), dev_buf.as_device_ptr(), FOUR_MB) }
    })
    .expect("memcpy_d2h");

    assert_eq!(got, expected);
}

#[test]
fn multiple_segments_into_separate_device_regions() {
    if !cuda_available() {
        eprintln!("skipping: libcuda.so.1 not loadable or no devices");
        return;
    }
    let (f, expected) = write_pattern_file(FOUR_MB);

    let dev = CuDevice::get(0).expect("CuDevice::get");
    let ctx = CuContext::primary_retain(dev).expect("primary_retain");
    let mut pipeline = CudaPipeline::new(0, 32, 128 * 1024).expect("CudaPipeline::new");
    let fd_idx = pipeline.register_file(f.path()).expect("register_file");

    // Three separate device allocations — emulates per-tensor torch.empty.
    // Pipeline must write each segment to its own destination, not into a
    // common buffer.
    let lengths = [4096usize, 65536, 1_000_000];
    let offsets = [100u64, 50_000, 1_500_000];
    let bufs: Vec<DeviceBuf> = ctx
        .with_current(|| {
            lengths
                .iter()
                .map(|&n| DeviceBuf::alloc(n))
                .collect::<ionic_rs::Result<Vec<_>>>()
        })
        .expect("alloc");

    let segments: Vec<LogicalSegment> = lengths
        .iter()
        .zip(offsets.iter())
        .zip(bufs.iter())
        .enumerate()
        .map(|(i, ((&len, &off), buf))| LogicalSegment {
            fd_idx,
            from: off,
            to: off + len as u64,
            dst_offset: buf.as_device_ptr() as usize,
            user_data: i as u64,
        })
        .collect();
    pipeline.run(&segments).expect("pipeline.run");

    // Verify each region.
    for ((&len, &off), buf) in lengths.iter().zip(offsets.iter()).zip(bufs.iter()) {
        let mut got = vec![0u8; len];
        ctx.with_current(|| unsafe {
            cuda::memcpy_d2h(got.as_mut_ptr(), buf.as_device_ptr(), len)
        })
        .expect("memcpy_d2h");
        assert_eq!(
            got,
            expected[off as usize..off as usize + len],
            "region at offset {off} length {len} mismatch"
        );
    }
}

#[test]
fn cross_chunk_boundary_segment() {
    if !cuda_available() {
        eprintln!("skipping: libcuda.so.1 not loadable or no devices");
        return;
    }
    let (f, expected) = write_pattern_file(FOUR_MB);

    let dev = CuDevice::get(0).expect("CuDevice::get");
    let ctx = CuContext::primary_retain(dev).expect("primary_retain");
    // Force lots of chunking with a tiny chunk size — exercises coalescer +
    // multi-chunk scatter into a single device region.
    let mut pipeline = CudaPipeline::new(0, 16, 4096).expect("CudaPipeline::new");
    let fd_idx = pipeline.register_file(f.path()).expect("register_file");

    let len = 100_000usize;
    let off = 5000u64;
    let buf = ctx
        .with_current(|| DeviceBuf::alloc(len))
        .expect("DeviceBuf::alloc");

    let segments = vec![LogicalSegment {
        fd_idx,
        from: off,
        to: off + len as u64,
        dst_offset: buf.as_device_ptr() as usize,
        user_data: 0,
    }];
    pipeline.run(&segments).expect("pipeline.run");

    let mut got = vec![0u8; len];
    ctx.with_current(|| unsafe { cuda::memcpy_d2h(got.as_mut_ptr(), buf.as_device_ptr(), len) })
        .expect("memcpy_d2h");
    assert_eq!(got, expected[off as usize..off as usize + len]);
}
