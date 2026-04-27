//! Offset-based read coalescing — pure logic, no I/O. Port of ionic's
//! `get_seq_chunks` from `~/ionic/lib/platform/linux/iouring.c:168`.
//!
//! Caller hands us a list of logical reads `(fd_idx, file_from..file_to,
//! dst_offset, user_data)`. We:
//!
//!   1. Round each read to chunk-aligned boundaries.
//!   2. Dedupe across reads that share aligned chunks (one read fills many
//!      tensors; one chunk can be reused without re-issuing the syscall).
//!   3. For each unique aligned chunk, build the scatter list of byte ranges
//!      from each contributing logical read.
//!
//! Output is a sorted list of `ChunkRead`s ready to submit to io_uring. The
//! caller iterates and either copies portions out (P3 standalone) or hands
//! them to the DMA worker (P4 integration).
//!
//! ionic uses `qsort` over `(path, offset)` pairs; we use `BTreeMap` keyed on
//! `(fd_idx, offset)` which dedupes and sorts in one pass.

use std::collections::BTreeMap;

/// One logical read the caller wants performed: bytes `[from, to)` in the
/// registered file `fd_idx`, deposited at `dst_offset` in the eventual
/// destination buffer. `user_data` rides through unchanged so callers can
/// attribute completed scatter entries back to their tensor / parameter.
#[derive(Debug, Clone, Copy)]
pub struct LogicalSegment {
    pub fd_idx: u32,
    pub from: u64,
    pub to: u64,
    pub dst_offset: usize,
    pub user_data: u64,
}

/// One chunk-aligned read to issue against io_uring. `len == chunk_bytes`
/// always (the coalescer pads to chunk boundaries; trailing bytes past EOF
/// are handled by the kernel returning a short read, which the engine then
/// trims via the actual byte count in the scatter entries).
#[derive(Debug)]
pub struct ChunkRead {
    pub fd_idx: u32,
    pub offset: u64,
    pub len: usize,
    pub scatter: Vec<ScatterEntry>,
}

/// One byte range to lift out of a chunk's staging buffer once the chunk is
/// resident. `staging_offset` is into the chunk; `dst_offset` is into the
/// caller's destination buffer.
#[derive(Debug, Clone, Copy)]
pub struct ScatterEntry {
    pub staging_offset: usize,
    pub len: usize,
    pub dst_offset: usize,
    pub user_data: u64,
}

/// Coalesce logical reads into chunk-aligned reads.
///
/// `chunk_bytes` controls both alignment and per-read size — every emitted
/// `ChunkRead.len` equals it. ionic uses 128 KiB by default; that's a sweet
/// spot between syscall amortization and read-amplification.
///
/// Empty input or `chunk_bytes == 0` returns an empty plan.
pub fn coalesce_chunks(segments: &[LogicalSegment], chunk_bytes: usize) -> Vec<ChunkRead> {
    if segments.is_empty() || chunk_bytes == 0 {
        return Vec::new();
    }

    let chunk = chunk_bytes as u64;

    // BTreeMap keyed on (fd_idx, aligned_offset). Sorting + dedup come for
    // free; we just append scatter entries as we encounter overlaps.
    let mut chunks: BTreeMap<(u32, u64), Vec<ScatterEntry>> = BTreeMap::new();

    for seg in segments {
        if seg.from >= seg.to {
            continue;
        }
        let aligned_from = (seg.from / chunk) * chunk;
        // align_up: divides + ceils.
        let aligned_to = seg.to.div_ceil(chunk) * chunk;

        let mut offset = aligned_from;
        while offset < aligned_to {
            let chunk_end = offset + chunk;
            let overlap_start = seg.from.max(offset);
            let overlap_end = seg.to.min(chunk_end);
            // A chunk that lies entirely past `seg.to` cannot occur given the
            // bounds, but defensive check is cheap.
            if overlap_start < overlap_end {
                let entry = ScatterEntry {
                    staging_offset: (overlap_start - offset) as usize,
                    len: (overlap_end - overlap_start) as usize,
                    dst_offset: seg.dst_offset + (overlap_start - seg.from) as usize,
                    user_data: seg.user_data,
                };
                chunks.entry((seg.fd_idx, offset)).or_default().push(entry);
            }
            offset += chunk;
        }
    }

    chunks
        .into_iter()
        .map(|((fd_idx, offset), scatter)| ChunkRead {
            fd_idx,
            offset,
            len: chunk_bytes,
            scatter,
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn seg(fd_idx: u32, from: u64, to: u64, dst_offset: usize, user_data: u64) -> LogicalSegment {
        LogicalSegment {
            fd_idx,
            from,
            to,
            dst_offset,
            user_data,
        }
    }

    #[test]
    fn empty_in_empty_out() {
        assert!(coalesce_chunks(&[], 4096).is_empty());
    }

    #[test]
    fn zero_chunk_bytes_returns_empty() {
        let segs = [seg(0, 0, 100, 0, 0)];
        assert!(coalesce_chunks(&segs, 0).is_empty());
    }

    #[test]
    fn single_segment_one_chunk() {
        // segment fits entirely in one chunk
        let segs = [seg(0, 100, 200, 0, 42)];
        let plan = coalesce_chunks(&segs, 4096);
        assert_eq!(plan.len(), 1);
        assert_eq!(plan[0].fd_idx, 0);
        assert_eq!(plan[0].offset, 0);
        assert_eq!(plan[0].scatter.len(), 1);
        let s = &plan[0].scatter[0];
        assert_eq!(s.staging_offset, 100);
        assert_eq!(s.len, 100);
        assert_eq!(s.dst_offset, 0);
        assert_eq!(s.user_data, 42);
    }

    #[test]
    fn segment_spans_two_chunks() {
        // 4000..5000 spans chunk 0..4096 and chunk 4096..8192
        let segs = [seg(0, 4000, 5000, 0, 7)];
        let plan = coalesce_chunks(&segs, 4096);
        assert_eq!(plan.len(), 2);
        // Sorted by (fd_idx, offset), so chunk 0 first.
        assert_eq!(plan[0].offset, 0);
        assert_eq!(plan[1].offset, 4096);
        // Tail of segment in chunk 0: 4000..4096 = 96 bytes at staging[4000..]
        assert_eq!(plan[0].scatter[0].staging_offset, 4000);
        assert_eq!(plan[0].scatter[0].len, 96);
        assert_eq!(plan[0].scatter[0].dst_offset, 0);
        // Head of segment in chunk 1: 4096..5000 = 904 bytes at staging[0..]
        assert_eq!(plan[1].scatter[0].staging_offset, 0);
        assert_eq!(plan[1].scatter[0].len, 904);
        assert_eq!(plan[1].scatter[0].dst_offset, 96);
    }

    #[test]
    fn two_segments_share_chunk_dedupe() {
        // Two tensors both reading from chunk 0..4096
        let segs = [
            seg(0, 0, 100, 0, 1),     // tensor 1
            seg(0, 200, 300, 100, 2), // tensor 2
        ];
        let plan = coalesce_chunks(&segs, 4096);
        assert_eq!(plan.len(), 1, "should dedupe to one chunk read");
        assert_eq!(plan[0].scatter.len(), 2);
        // Both scatter entries refer to chunk @ 0
        let users: Vec<u64> = plan[0].scatter.iter().map(|s| s.user_data).collect();
        assert!(users.contains(&1) && users.contains(&2));
    }

    #[test]
    fn segments_in_different_files_dont_dedupe() {
        let segs = [seg(0, 0, 100, 0, 1), seg(1, 0, 100, 100, 2)];
        let plan = coalesce_chunks(&segs, 4096);
        assert_eq!(plan.len(), 2);
        assert_eq!(plan[0].fd_idx, 0);
        assert_eq!(plan[1].fd_idx, 1);
    }

    #[test]
    fn empty_segment_dropped() {
        let segs = [seg(0, 100, 100, 0, 0), seg(0, 100, 200, 0, 1)];
        let plan = coalesce_chunks(&segs, 4096);
        assert_eq!(plan.len(), 1);
        assert_eq!(plan[0].scatter.len(), 1);
        assert_eq!(plan[0].scatter[0].user_data, 1);
    }

    #[test]
    fn segment_aligned_to_chunk_boundary() {
        // 0..4096 fits exactly in chunk 0
        let segs = [seg(0, 0, 4096, 0, 9)];
        let plan = coalesce_chunks(&segs, 4096);
        assert_eq!(plan.len(), 1);
        assert_eq!(plan[0].scatter[0].staging_offset, 0);
        assert_eq!(plan[0].scatter[0].len, 4096);
    }

    #[test]
    fn segment_crossing_three_chunks() {
        // 100..9000 spans chunks 0/4096/8192
        let segs = [seg(0, 100, 9000, 0, 5)];
        let plan = coalesce_chunks(&segs, 4096);
        assert_eq!(plan.len(), 3);
        let total_len: usize = plan
            .iter()
            .flat_map(|c| c.scatter.iter())
            .map(|s| s.len)
            .sum();
        assert_eq!(total_len, 9000 - 100);
    }
}
