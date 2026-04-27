//! End-to-end tests for the io_uring read engine.
//!
//! Each test writes a deterministic byte pattern to a temp file, reads
//! it back via `ReadEngine`, and compares against the expected bytes.
//! The previous incarnation of these tests went through pyo3 + pytest;
//! they're pure Rust now.

use std::io::Write;

use ionic_rs::iouring::{coalesce_chunks, LogicalSegment, ReadEngine};
use tempfile::NamedTempFile;

const FIVE_MB: usize = 5 * 1024 * 1024;

/// Build a deterministic 5 MB pattern: byte i = LCG(i) & 0xFF. Big enough
/// to span multiple 128 KiB chunks; deterministic so tests can compare
/// without holding the full file in RAM more than once.
fn write_pattern_file() -> (NamedTempFile, Vec<u8>) {
    let mut data = vec![0u8; FIVE_MB];
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

/// Read `ranges` from `path` via `ReadEngine` and return the concatenated
/// bytes (in input order, length = sum of range lengths).
fn read_via_engine(
    path: &std::path::Path,
    ranges: &[(u64, u64)],
    chunk_bytes: usize,
    qd: u32,
) -> Vec<u8> {
    let mut engine = ReadEngine::new(qd, chunk_bytes).expect("ReadEngine::new");
    let fd_idx = engine.register_file(path).expect("register_file");

    let mut total: usize = 0;
    let segments: Vec<LogicalSegment> = ranges
        .iter()
        .enumerate()
        .map(|(i, &(off, len))| {
            let seg = LogicalSegment {
                fd_idx,
                from: off,
                to: off + len,
                dst_offset: total,
                user_data: i as u64,
            };
            total += len as usize;
            seg
        })
        .collect();

    let chunks = coalesce_chunks(&segments, chunk_bytes);
    let mut dst = vec![0u8; total];
    engine
        .fetch(&chunks, |buf, scatter| {
            for s in scatter {
                dst[s.dst_offset..s.dst_offset + s.len]
                    .copy_from_slice(&buf[s.staging_offset..s.staging_offset + s.len]);
            }
            Ok(())
        })
        .expect("fetch");
    dst
}

#[test]
fn reads_full_file_match_reference() {
    let (f, expected) = write_pattern_file();
    let out = read_via_engine(f.path(), &[(0, expected.len() as u64)], 128 * 1024, 64);
    assert_eq!(out, expected);
}

#[test]
fn partial_read_one_chunk() {
    let (f, expected) = write_pattern_file();
    let out = read_via_engine(f.path(), &[(100, 1000)], 128 * 1024, 64);
    assert_eq!(out, expected[100..1100]);
}

#[test]
fn range_spans_two_chunks() {
    let (f, expected) = write_pattern_file();
    let chunk = 128 * 1024;
    let start = chunk - 500;
    let length = 1000;
    let out = read_via_engine(f.path(), &[(start as u64, length)], 128 * 1024, 64);
    assert_eq!(out, expected[start..start + length as usize]);
}

#[test]
fn multiple_ranges_concatenated() {
    let (f, expected) = write_pattern_file();
    let ranges = [(0u64, 100u64), (50_000, 200), (1_000_000, 4096)];
    let out = read_via_engine(f.path(), &ranges, 128 * 1024, 64);
    let mut want = Vec::new();
    for &(off, len) in &ranges {
        want.extend_from_slice(&expected[off as usize..(off + len) as usize]);
    }
    assert_eq!(out, want);
}

#[test]
fn overlapping_ranges_dedup_chunks() {
    // Two ranges within the same 128 KiB chunk — coalescer should issue
    // one read but scatter both portions.
    let (f, expected) = write_pattern_file();
    let ranges = [(100, 1000), (5000, 1000)];
    let out = read_via_engine(f.path(), &ranges, 128 * 1024, 64);
    let want = [&expected[100..1100], &expected[5000..6000]].concat();
    assert_eq!(out, want);
}

#[test]
fn small_chunk_size_forces_qd_exhaustion() {
    // 4 KiB chunks, QD of 4 — reading 5 MB requires 1280 chunks across
    // only 4 slots. Heavily exercises slot recycling.
    let (f, expected) = write_pattern_file();
    let out = read_via_engine(f.path(), &[(0, expected.len() as u64)], 4096, 4);
    assert_eq!(out, expected);
}

#[test]
fn short_read_at_eof() {
    // Range that ends past EOF — kernel returns a short read; engine
    // should trim scatter entries to what was actually delivered, leaving
    // unfilled tail bytes as zero-init.
    let (f, expected) = write_pattern_file();
    let eof_minus_50 = expected.len() - 50;
    let out = read_via_engine(f.path(), &[(eof_minus_50 as u64, 200)], 128 * 1024, 64);
    assert_eq!(&out[..50], &expected[eof_minus_50..]);
    assert_eq!(&out[50..], &vec![0u8; 150][..]);
}
