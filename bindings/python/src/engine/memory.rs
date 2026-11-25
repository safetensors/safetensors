//! The geometry of a load and the CUDA sink its chunks drain to. Two units,
//! computed once from the spans the consumer requests, sorted by start and disjoint:
//!
//! ```text
//! tensors  |t0 |  t1  |     t2     |  t3  |       t4        | t5 | t6  |
//! spans    |S0#|..|S1#|S2#|...............|#######S3########|#S4#|#S5##|
//! chunks   |c0 |  |  c1   |               |   c2   |   c3   |   c4   |c|
//! ```
//!
//! - spans [`LoadPlan::spans`]: a byte interval of the data section, copied to
//!   its destination, or, with a [`Gather`], packed into it by the gather's
//!   strided copies. The delivery unit: `take` hands out the span's owner.
//! - chunks ([`LoadPlan::chunks`]): at most [`CHUNK_SIZE`] of adjacent spans,
//!   never across a gap. The I/O unit: one `pread`, one slab, one event, one
//!   `memcpy` per run of spans whose destinations are adjacent. `span_chunks`
//!   and `chunk_spans` map one to the other.

use std::{
    num::NonZeroUsize,
    ops::Range,
    sync::{
        atomic::{AtomicBool, AtomicU64, AtomicUsize, Ordering},
        Arc, Condvar, Mutex,
    },
    time::Duration,
};

use super::{
    cuda_memory::device_stream,
    loader::LoaderError,
    transfer::{Error, Sink, Staged},
};
use crate::engine::cuda::{CudaApi, CudaError, Event, Stream};
use safetensors::slice::{Gather, StridedCopy};

/// A byte interval `start..end` of the file's data section
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Span {
    pub start: usize,
    pub end: usize,
}

/// Geometry of one load. Every interval here is a byte range within the file's
/// data section; device addresses never appear in the plan.
pub struct LoadPlan {
    pub(super) spans: Box<[Span]>,
    /// per span, the copies that pack its bytes into its destination; `None` for
    /// one plain copy
    gathers: Box<[Option<Gather>]>,
    /// per span, the `chunks` that carry its bytes; empty for a zero-size span
    span_chunks: Box<[Range<usize>]>,
    /// data section byte interval of each chunk: at most [`CHUNK_SIZE`] of
    /// adjacent spans, never across a gap
    pub(super) chunks: Box<[Range<usize>]>,
    /// per chunk, the spans it carries bytes of
    chunk_spans: Box<[Range<usize>]>,
    in_file_offset: usize,
}

/// The part of `copy` whose source bytes lie in `window`, sources relative to
/// `window.start`: the rows inside it as one copy, and the rows its edges cut (at
/// most two) as single-row copies of their part inside it.
fn clip(copy: &StridedCopy, window: &Range<usize>) -> [Option<StridedCopy>; 3] {
    let row = |k: usize| copy.src + k * copy.src_pitch;
    let part = |k: usize| {
        let (start, end) = (
            row(k).max(window.start),
            (row(k) + copy.width).min(window.end),
        );
        (start < end).then(|| StridedCopy {
            src: start - window.start,
            dst: copy.dst + k * copy.dst_pitch + (start - row(k)),
            width: end - start,
            height: 1,
            src_pitch: 0,
            dst_pitch: 0,
        })
    };
    if copy.height == 1 {
        return [part(0), None, None];
    }
    if window.end <= copy.src {
        return [None; 3];
    }
    // rows are disjoint and in order (src_pitch >= width): the ones meeting the
    // window are consecutive
    let first = window
        .start
        .checked_sub(copy.src + copy.width)
        .map_or(0, |past| past / copy.src_pitch + 1);
    let last = ((window.end - copy.src - 1) / copy.src_pitch).min(copy.height - 1);
    if first > last {
        return [None; 3];
    }
    let whole = |k: usize| row(k) >= window.start && row(k) + copy.width <= window.end;
    let (mut lo, mut hi) = (first, last);
    let head = if whole(lo) {
        None
    } else {
        lo += 1;
        part(lo - 1)
    };
    let tail = if lo <= hi && !whole(hi) {
        hi -= 1;
        part(hi + 1)
    } else {
        None
    };
    let rows = (lo <= hi).then(|| StridedCopy {
        src: row(lo) - window.start,
        dst: copy.dst + lo * copy.dst_pitch,
        height: hi - lo + 1,
        ..*copy
    });
    [head, rows, tail]
}

impl LoadPlan {
    pub fn new(
        spans: Vec<Span>,
        gathers: Vec<Option<Gather>>,
        in_file_offset: usize,
        chunk_size: NonZeroUsize,
    ) -> Self {
        let chunk_size = chunk_size.get();
        assert!(
            spans.iter().all(|s| s.start <= s.end)
                && spans.windows(2).all(|w| w[0].end <= w[1].start),
            "spans must be sorted by start and disjoint, each with start <= end"
        );
        assert_eq!(gathers.len(), spans.len(), "one gather entry per span");

        let mut chunks: Vec<Range<usize>> = Vec::new();
        let mut chunk_spans: Vec<Range<usize>> = Vec::new();
        let mut span_chunks = vec![0..0; spans.len()];
        for (i, span) in spans.iter().enumerate() {
            let mut start = span.start;
            let mut first = chunks.len();
            while start < span.end {
                let n = chunks.len();
                match chunks.last_mut() {
                    // the previous span's last chunk ends here and has room left
                    Some(chunk) if chunk.end == start && chunk.len() < chunk_size => {
                        chunk.end = (chunk.start + chunk_size).min(span.end);
                        chunk_spans.last_mut().unwrap().end = i + 1;
                        first = first.min(n - 1);
                        start = chunk.end;
                    }
                    _ => {
                        let end = (start + chunk_size).min(span.end);
                        chunks.push(start..end);
                        chunk_spans.push(i..i + 1);
                        start = end;
                    }
                }
            }
            span_chunks[i] = first..chunks.len();
        }

        Self {
            spans: spans.into_boxed_slice(),
            gathers: gathers.into_boxed_slice(),
            span_chunks: span_chunks.into_boxed_slice(),
            chunks: chunks.into_boxed_slice(),
            chunk_spans: chunk_spans.into_boxed_slice(),
            in_file_offset,
        }
    }

    /// The chunks' byte ranges in the file
    pub(super) fn file_chunks(&self) -> Box<[Range<u64>]> {
        let at = |n: usize| (self.in_file_offset + n) as u64;
        self.chunks.iter().map(|c| at(c.start)..at(c.end)).collect()
    }

    /// The index of the chunk at `file` in the file, see [`Self::file_chunks`]
    fn chunk_at(&self, file: &Range<u64>) -> usize {
        let start = file.start as usize - self.in_file_offset;
        let c = self.chunks.partition_point(|c| c.start < start);
        debug_assert_eq!(self.chunks.get(c).map(|c| c.start), Some(start));
        c
    }

    /// Issues the copies of chunk `c` from its slab, sources as offsets in the slab
    /// and destinations as addresses: `dst(s)` is span `s`'s destination and
    /// `allocation_end(d)` the end of the allocation holding `d`, if known.
    fn chunk_copies<E>(
        &self,
        c: usize,
        dst: impl Fn(usize) -> u64,
        mut allocation_end: impl FnMut(u64) -> Option<u64>,
        mut copy: impl FnMut(StridedCopy) -> Result<(), E>,
    ) -> Result<(), E> {
        let chunk = self.chunks[c].clone();
        let plain = |(dst, at, len, _): (u64, usize, usize, u64)| StridedCopy {
            src: at - chunk.start,
            dst: dst as usize,
            width: len,
            height: 1,
            src_pitch: 0,
            dst_pitch: 0,
        };
        // one copy per run of pieces adjacent in the file and on the device,
        // within one allocation: (device address, data section offset, length,
        // end of the allocation, looked up once the run grows)
        let mut run: Option<(u64, usize, usize, u64)> = None;
        for s in self.chunk_spans[c].clone() {
            let span = self.spans[s];
            let (start, end) = (span.start.max(chunk.start), span.end.min(chunk.end));
            if start >= end {
                continue;
            }
            if let Some(gather) = &self.gathers[s] {
                if let Some(run) = run.take() {
                    copy(plain(run))?;
                }
                let window = start - span.start..end - span.start;
                for piece in gather
                    .copies
                    .iter()
                    .flat_map(|g| clip(g, &window))
                    .flatten()
                {
                    copy(StridedCopy {
                        src: start - chunk.start + piece.src,
                        dst: dst(s) as usize + piece.dst,
                        ..piece
                    })?;
                }
                continue;
            }
            let dst = dst(s) + (start - span.start) as u64;
            if let Some((d0, at, len, limit)) = run.as_mut() {
                if *at + *len == start && *d0 + *len as u64 == dst {
                    if *limit == 0 {
                        *limit = allocation_end(*d0).unwrap_or(dst);
                    }
                    if dst + (end - start) as u64 <= *limit {
                        *len += end - start;
                        continue;
                    }
                }
            }
            if let Some(run) = run.replace((dst, start, end - start, 0)) {
                copy(plain(run))?;
            }
        }
        if let Some(run) = run {
            copy(plain(run))?;
        }
        Ok(())
    }
}

pub(super) struct CudaSink<D> {
    api: &'static CudaApi,
    device: i32,
    stream: Stream,
    plan: Arc<LoadPlan>,
    /// per span, the device address its bytes are copied to; set by `provide`
    dsts: Box<[AtomicU64]>,
    /// per span, the owner of its destination; `None` until provided, and once
    /// taken or released
    owners: Box<[Mutex<Option<D>>]>,
    /// per span, the event of the batch that provided it, recorded on the
    /// allocating stream: the destination is free to write once it completes
    ready: Box<[AtomicU64]>,
    /// the spans with a destination: a prefix of the plan
    pub(super) provided: AtomicUsize,
    /// guards `provided` advancing and `released` turning on, so neither is missed
    /// by a waiter on `provision`
    provision_lock: Mutex<()>,
    provision: Condvar,
    batch_events: Mutex<Vec<Event>>,
    chunk_completion_events: Box<[AtomicU64]>,
    closed: AtomicBool,
    released: AtomicBool,
}

unsafe impl<D: Send> Send for CudaSink<D> {}
unsafe impl<D: Send> Sync for CudaSink<D> {}

impl<D> CudaSink<D> {
    pub(super) fn new(
        api: &'static CudaApi,
        plan: Arc<LoadPlan>,
        device: i32,
    ) -> Result<Self, LoaderError> {
        let stream = device_stream(api, device)?;
        let n = plan.spans.len();
        Ok(Self {
            api,
            device,
            stream,
            dsts: (0..n).map(|_| AtomicU64::new(0)).collect(),
            owners: (0..n).map(|_| Mutex::new(None)).collect(),
            ready: (0..n).map(|_| AtomicU64::new(0)).collect(),
            provided: AtomicUsize::new(0),
            provision_lock: Mutex::new(()),
            provision: Condvar::new(),
            batch_events: Mutex::new(Vec::new()),
            chunk_completion_events: (0..plan.chunks.len()).map(|_| AtomicU64::new(0)).collect(),
            closed: AtomicBool::new(false),
            released: AtomicBool::new(false),
            plan,
        })
    }

    /// Gives the next `dsts.len()` spans their destinations, allocated on
    /// `alloc_stream`. Hands `dsts` back if the sink is released or the event fails.
    pub(super) fn provide(
        &self,
        dsts: Vec<(u64, D)>,
        alloc_stream: Stream,
    ) -> Result<(), (LoaderError, Vec<(u64, D)>)> {
        let _g = self.provision_lock.lock().unwrap();
        if self.released.load(Ordering::Acquire) {
            return Err((LoaderError::Closed, dsts));
        }
        let first = self.provided.load(Ordering::Acquire);
        assert!(
            first + dsts.len() <= self.plan.spans.len(),
            "more destinations than spans"
        );
        // the caller's allocator may hand out memory still in use by work queued
        // on the stream it allocated on: workers wait for that work before copying
        let ready = self.api.with_device(self.device, |d| {
            let e = d.event_create()?;
            if let Err(err) = self.api.event_record(e, alloc_stream) {
                let _ = self.api.event_destroy(e);
                return Err(err);
            }
            Ok(e)
        });
        let ready = match ready {
            Ok(e) => e,
            Err(err) => return Err((err.into(), dsts)),
        };
        self.batch_events.lock().unwrap().push(ready);
        let n = dsts.len();
        for (i, (ptr, owner)) in dsts.into_iter().enumerate() {
            self.dsts[first + i].store(ptr, Ordering::Relaxed);
            self.ready[first + i].store(ready as u64, Ordering::Relaxed);
            *self.owners[first + i].lock().unwrap() = Some(owner);
        }
        self.provided.store(first + n, Ordering::Release);
        self.provision.notify_all();
        Ok(())
    }

    /// Waits until the first `n` spans have a destination.
    fn wait_provided(&self, n: usize) -> Result<(), Error> {
        if self.provided.load(Ordering::Acquire) >= n {
            return Ok(());
        }
        let mut g = self.provision_lock.lock().unwrap();
        loop {
            if self.provided.load(Ordering::Acquire) >= n {
                return Ok(());
            }
            if self.closed.load(Ordering::Acquire) {
                return Err(Error::Stopped);
            }
            g = self.provision.wait(g).unwrap();
        }
    }

    /// Waits until span `span`'s bytes landed.
    pub(super) fn wait_landed(&self, span: usize) -> Result<(), LoaderError> {
        // a zero-size span has no chunk: it is ready once provided
        self.wait_provided(span + 1)
            .map_err(|_| LoaderError::Closed)?;
        for chunk in self.plan.span_chunks[span].clone() {
            let mut i = 0;
            let e = loop {
                match self.chunk_completion_events[chunk].load(Ordering::Acquire) {
                    0 => {
                        if self.closed.load(Ordering::Acquire) {
                            return Err(LoaderError::Closed);
                        }
                        if i < 64 {
                            std::thread::yield_now();
                            i += 1;
                        } else {
                            std::thread::sleep(Duration::from_micros(200));
                        }
                    }
                    e => break e as Event,
                }
            };
            self.api.event_sync(e)?;
        }
        Ok(())
    }

    pub(super) fn take(&self, idx: usize) -> Result<D, LoaderError> {
        match self.owners[idx].lock().unwrap().take() {
            Some(owner) => Ok(owner),
            None if self.released.load(Ordering::Acquire) => Err(LoaderError::Closed),
            None => Err(LoaderError::AlreadyDelivered),
        }
    }

    /// Called once the workers are joined: waits for the copies in flight, then
    /// returns the owners not taken, or leaks them if the wait fails.
    pub(super) fn close(&self) -> Vec<D> {
        {
            let _g = self.provision_lock.lock().unwrap();
            self.closed.store(true, Ordering::Release);
            self.released.store(true, Ordering::Release);
            self.provision.notify_all();
        }
        let synced: Result<(), CudaError> = self.api.with_device(self.device, |d| {
            let e = d.event_create()?;
            let res = self
                .api
                .event_record(e, self.stream)
                .and_then(|()| self.api.event_sync(e));
            let _ = self.api.event_destroy(e);
            res
        });
        let owners: Vec<D> = self
            .owners
            .iter()
            .filter_map(|o| o.lock().unwrap().take())
            .collect();
        if synced.is_err() {
            // copies may still be queued into them: leak them rather than free them
            std::mem::forget(owners);
            return Vec::new();
        }
        owners
    }

    pub(super) fn signal_stop(&self) {
        let _g = self.provision_lock.lock().unwrap();
        self.closed.store(true, Ordering::Release);
        self.provision.notify_all();
    }

    pub(super) fn fully_scheduled(&self, span: usize) -> bool {
        self.provided.load(Ordering::Acquire) > span
            && self.plan.span_chunks[span]
                .clone()
                .all(|c| self.chunk_completion_events[c].load(Ordering::Acquire) != 0)
    }

    /// How many spans the claimed chunks reach
    pub(super) fn scheduled(&self, claimed: usize) -> usize {
        match claimed.min(self.plan.chunks.len()) {
            n if n == self.plan.chunks.len() => self.plan.spans.len(),
            0 => 0,
            n => self.plan.chunk_spans[n - 1].end,
        }
    }
}

impl<D: Send> Sink for CudaSink<D> {
    fn wait_ready(&self, file: &Range<u64>) -> Result<(), Error> {
        let spans = self.plan.chunk_spans[self.plan.chunk_at(file)].clone();
        self.wait_provided(spans.end)?;
        // host-side: a pending copy would hold its slab and stall other loaders
        let mut last = 0;
        for s in spans {
            let e = self.ready[s].load(Ordering::Relaxed);
            if e != last {
                self.api.event_sync(e as Event)?;
                last = e;
            }
        }
        Ok(())
    }

    fn drain(&self, staged: Staged) -> Result<(), Error> {
        let chunk_idx = self.plan.chunk_at(staged.file());
        let src = staged.bytes().as_ptr();
        let e = self.api.with_device(self.device, |d| {
            self.plan.chunk_copies(
                chunk_idx,
                |s| self.dsts[s].load(Ordering::Relaxed),
                |d0| self.api.allocation_end(d0),
                |c| {
                    let src = src.wrapping_add(c.src);
                    match c.height {
                        1 => self
                            .api
                            .memcpy_h2d_async(c.dst as u64, src, c.width, self.stream),
                        _ => self.api.memcpy2d_h2d_async(
                            c.dst as u64,
                            c.dst_pitch,
                            src,
                            c.src_pitch,
                            c.width,
                            c.height,
                            self.stream,
                        ),
                    }
                },
            )?;
            staged.into_slab().release(d, self.stream)?;
            let e = d.event_create()?;
            if let Err(err) = self.api.event_record(e, self.stream) {
                let _ = self.api.event_destroy(e);
                return Err(err);
            }
            Ok(e)
        })?;
        self.chunk_completion_events[chunk_idx].store(e as u64, Ordering::Release);
        Ok(())
    }

    fn stop(&self) {
        self.signal_stop();
    }
}

impl<D> Drop for CudaSink<D> {
    fn drop(&mut self) {
        self.close();
        for e in self.batch_events.get_mut().unwrap().drain(..) {
            let _ = self.api.event_destroy(e);
        }
        for e in self.chunk_completion_events.iter() {
            let e = e.swap(0, Ordering::AcqRel);
            if e != 0 {
                let _ = self.api.event_destroy(e as Event);
            }
        }
    }
}

#[cfg(test)]
#[allow(clippy::single_range_in_vec_init)] // one range per dimension
mod tests {
    use std::{num::NonZeroUsize, ops::Range};

    use super::{LoadPlan, Span};
    #[cfg(target_os = "linux")]
    use crate::engine::file::{
        tests::{evict, random_file},
        DirectFile,
    };
    use crate::engine::{
        file::tests::{slab_buf, Rng},
        pool::SLAB_SIZE,
    };
    use safetensors::slice::{slice_layout, Gather, StridedCopy, TensorIndexer};
    use safetensors::Dtype;

    fn sp(start: usize, end: usize) -> Span {
        Span { start, end }
    }

    fn plan(spans: Vec<Span>, chunk_size: usize) -> LoadPlan {
        let gathers = vec![None; spans.len()];
        let plan = LoadPlan::new(spans, gathers, 0, NonZeroUsize::new(chunk_size).unwrap());
        check_plan(&plan, chunk_size);
        plan
    }

    /// A device of `len` bytes of 0xAA after the plan's copies of `file`, span `s`
    /// landing at `dsts[s]`, all in one allocation
    fn land(plan: &LoadPlan, file: &[u8], dsts: &[usize], len: usize) -> Vec<u8> {
        land_read(plan, dsts, len, |slab, offset, n| {
            let at = offset as usize - plan.in_file_offset;
            slab[..n].copy_from_slice(&file[at..at + n]);
            0
        })
    }

    /// [`land`] with each chunk read as the sink reads it: `read(slab, offset,
    /// len)` fills the slab and returns where the chunk starts in it
    fn land_read(
        plan: &LoadPlan,
        dsts: &[usize],
        len: usize,
        mut read: impl FnMut(&mut [u8], u64, usize) -> usize,
    ) -> Vec<u8> {
        let mut device = vec![0xAA; len];
        let (mut buf, at) = slab_buf(SLAB_SIZE, 0);
        for c in 0..plan.chunks.len() {
            let chunk = plan.chunks[c].clone();
            let offset = (plan.in_file_offset + chunk.start) as u64;
            let head = read(&mut buf[at.clone()], offset, chunk.len());
            let slab = &buf[at.start + head..];
            let copy = |cp: StridedCopy| -> Result<(), ()> {
                for r in 0..cp.height {
                    let (src, dst) = (cp.src + r * cp.src_pitch, cp.dst + r * cp.dst_pitch);
                    device[dst..dst + cp.width].copy_from_slice(&slab[src..src + cp.width]);
                }
                Ok(())
            };
            plan.chunk_copies(c, |s| dsts[s] as u64, |_| Some(len as u64), copy)
                .unwrap();
        }
        device
    }

    /// The slice's bytes by brute force over element indices, from a tensor starting at byte 0 of `bytes`.
    fn slice_bytes(
        bytes: &[u8],
        shape: &[usize],
        elem: usize,
        intervals: &[Vec<Range<usize>>],
    ) -> Vec<u8> {
        fn rec(
            d: usize,
            at: usize,
            strides: &[usize],
            ivs: &[Vec<Range<usize>>],
            elem: usize,
            b: &[u8],
            out: &mut Vec<u8>,
        ) {
            if d == strides.len() {
                out.extend_from_slice(&b[at..at + elem]);
                return;
            }
            for r in &ivs[d] {
                for i in r.clone() {
                    rec(d + 1, at + i * strides[d], strides, ivs, elem, b, out);
                }
            }
        }
        let mut strides = vec![elem; shape.len()];
        for d in (0..shape.len().saturating_sub(1)).rev() {
            strides[d] = strides[d + 1] * shape[d + 1];
        }
        let mut out = Vec::new();
        rec(0, 0, &strides, intervals, elem, bytes, &mut out);
        out
    }

    #[test]
    fn test_random_slices_land_exactly() {
        let mut rng = Rng(7);
        for trial in 0..2000 {
            // a few tensors laid out back to back, as in a safetensors data section
            let n_tensors = 1 + rng.below(4);
            let mut layout = Vec::new(); // (start, shape, elem)
            let mut cursor = 0;
            for _ in 0..n_tensors {
                let shape: Vec<usize> = (0..1 + rng.below(4)).map(|_| 1 + rng.below(6)).collect();
                let elem = [1, 2, 4][rng.below(3)];
                layout.push((cursor, shape.clone(), elem));
                cursor += shape.iter().product::<usize>() * elem;
            }
            let file: Vec<u8> = (0..cursor).map(|_| rng.below(0xAA) as u8).collect();

            let (mut spans, mut gathers, mut dsts) = (Vec::new(), Vec::new(), Vec::new());
            // destinations back to back or a few bytes apart
            let mut expected = Vec::new();
            for (start, shape, elem) in &layout {
                if rng.below(4) == 0 {
                    continue; // left out of the plan
                }
                let intervals: Vec<Vec<Range<usize>>> = shape
                    .iter()
                    .map(|&n| match rng.below(3) {
                        0 => vec![0..n],
                        _ => {
                            let mut cuts: Vec<usize> = (0..2 * (1 + rng.below(2)))
                                .map(|_| rng.below(n + 1))
                                .collect();
                            cuts.sort_unstable();
                            cuts.dedup();
                            cuts.chunks(2)
                                .filter(|c| c.len() == 2)
                                .map(|c| c[0]..c[1])
                                .collect()
                        }
                    })
                    .collect();
                let dtype = [Dtype::U8, Dtype::U16, Dtype::U32]
                    [[1, 2, 4].iter().position(|e| e == elem).unwrap()];
                let indexers: Vec<Vec<TensorIndexer>> = intervals
                    .iter()
                    .map(|d| d.iter().map(|r| TensorIndexer::from(r.clone())).collect())
                    .collect();
                let r = slice_layout(dtype, shape, &indexers).unwrap();
                let nbytes = shape.iter().product::<usize>() * elem;
                let bytes = slice_bytes(&file[*start..*start + nbytes], shape, *elem, &intervals);
                expected.resize(expected.len() + [0, 3][rng.below(2)], 0xAA);
                dsts.push(expected.len());
                expected.extend(bytes);
                spans.push(sp(start + r.read.start, start + r.read.end));
                gathers.push(r.gather);
            }
            let chunk = 1 + rng.below(48);
            let plan = LoadPlan::new(spans, gathers, 0, NonZeroUsize::new(chunk).unwrap());
            assert_eq!(
                land(&plan, &file, &dsts, expected.len()),
                expected,
                "trial {trial}, chunk {chunk}"
            );
        }
    }

    #[test]
    fn test_gather_rows_cut_by_chunk_edges() {
        // columns 2..5 of a 4 x 10 tensor between two plain spans, adjacent in the
        // file and on the device: 7-byte chunks cut its rows, and the plain spans
        // don't join it
        let g = Gather {
            copies: vec![StridedCopy {
                src: 0,
                dst: 0,
                width: 3,
                height: 4,
                src_pitch: 10,
                dst_pitch: 3,
            }],
            len: 12,
        };
        let spans = vec![sp(0, 7), sp(7, 40), sp(40, 45)];
        let plan = LoadPlan::new(
            spans,
            vec![None, Some(g), None],
            0,
            NonZeroUsize::new(7).unwrap(),
        );
        let file: Vec<u8> = (0..45).collect();
        let rows = (0..4).flat_map(|r| 7 + r * 10..10 + r * 10);
        let expected: Vec<u8> = (0..7).chain(rows).chain(40..45).collect();
        assert_eq!(land(&plan, &file, &[0, 7, 19], 24), expected);
    }

    fn check_plan(plan: &LoadPlan, chunk_size: usize) {
        let covered = |at: usize| plan.spans.iter().any(|s| s.start <= at && at < s.end);
        // chunks: sorted, disjoint, <= chunk_size, cut only at gaps and full chunks
        for w in plan.chunks.windows(2) {
            assert!(
                w[0].end <= w[1].start,
                "chunks overlap or are unsorted: {w:?}"
            );
            if w[0].end == w[1].start {
                assert_eq!(
                    w[0].len(),
                    chunk_size,
                    "undersized cut without a gap: {w:?}"
                );
            }
        }
        for c in plan.chunks.iter() {
            assert!(!c.is_empty() && c.len() <= chunk_size, "bad chunk {c:?}");
            assert!(c.clone().all(covered), "chunk {c:?} reads a gap");
        }
        // every span byte is in a chunk
        for s in plan.spans.iter() {
            assert!(
                (s.start..s.end).all(|at| plan.chunks.iter().any(|c| c.contains(&at))),
                "span {s:?} not covered"
            );
        }
        // span_chunks and chunk_spans == the chunks and spans that intersect
        let meet = |s: &Span, c: &std::ops::Range<usize>| s.start < c.end && c.start < s.end;
        for (i, s) in plan.spans.iter().enumerate() {
            for (j, c) in plan.chunks.iter().enumerate() {
                assert_eq!(
                    plan.span_chunks[i].contains(&j),
                    meet(s, c),
                    "span {s:?} vs chunk {c:?}"
                );
                if meet(s, c) {
                    assert!(
                        plan.chunk_spans[j].contains(&i),
                        "chunk {c:?} misses span {s:?}"
                    );
                }
            }
        }
        for (j, spans) in plan.chunk_spans.iter().enumerate() {
            for i in spans.clone() {
                let s = &plan.spans[i];
                assert!(
                    s.start == s.end || meet(s, &plan.chunks[j]),
                    "chunk {j} lists span {s:?}"
                );
            }
        }
    }

    #[test]
    fn test_chunks_cross_adjacent_spans() {
        let p = plan(vec![sp(0, 5), sp(5, 12), sp(12, 18)], 5);
        assert_eq!(
            p.chunks,
            vec![0..5, 5..10, 10..15, 15..18].into_boxed_slice()
        );
        assert_eq!(p.chunk_spans[2], 1..3);
    }

    #[test]
    fn test_chunk_len_exact_multiple_of_chunk_size() {
        let p = plan(vec![sp(0, 10)], 5);
        assert_eq!(p.chunks.len(), 2);
    }

    #[test]
    fn test_multi_chunk_span() {
        let p = plan(vec![sp(0, 42)], 5);
        assert_eq!(p.span_chunks[0], 0..9);
    }

    #[test]
    fn test_many_spans_in_single_chunk() {
        let p = plan(
            vec![
                sp(0, 1),
                sp(1, 2),
                sp(2, 3),
                sp(3, 4),
                sp(4, 5),
                sp(5, 6),
                sp(6, 7),
                sp(7, 8),
                sp(8, 11),
                sp(11, 17),
                sp(17, 20),
            ],
            20,
        );
        assert_eq!((p.chunks.len(), p.chunk_spans[0].clone()), (1, 0..11));
    }

    #[test]
    fn test_chunk_size_larger_than_data() {
        let p = plan(vec![sp(0, 5), sp(5, 8)], 4242);
        assert_eq!(p.chunks.len(), 1);
    }

    #[test]
    fn test_empty_spans() {
        let p = plan(vec![sp(0, 0), sp(0, 5), sp(5, 5), sp(5, 8), sp(8, 8)], 5);
        assert!(p.span_chunks[0].is_empty() && p.span_chunks[2].is_empty());
    }

    #[test]
    fn test_spans_with_a_gap() {
        // no chunk covers the gap
        let p = plan(vec![sp(0, 5), sp(12, 18)], 5);
        assert_eq!(p.chunks, vec![0..5, 12..17, 17..18].into_boxed_slice());
        assert_eq!(p.span_chunks[0], 0..1);
        assert_eq!(p.span_chunks[1], 1..3);
    }

    #[test]
    fn test_single_sub_span() {
        let p = plan(vec![sp(14, 17)], 5);
        assert_eq!((p.chunks.len(), p.chunks[0].clone()), (1, 14..17));
    }

    #[test]
    #[should_panic(expected = "sorted by start and disjoint")]
    fn test_unsorted_spans_are_rejected() {
        plan(vec![sp(12, 18), sp(0, 5)], 5);
    }

    #[test]
    #[should_panic(expected = "sorted by start and disjoint")]
    fn test_overlapping_spans_are_rejected() {
        plan(vec![sp(0, 5), sp(3, 8)], 5);
    }

    #[cfg(target_os = "linux")]
    #[test]
    fn test_sliced_plan_reads_cold_as_cached() {
        // rows and columns of a 64 x 300 U16 tensor, then a whole tensor, in a data
        // section at an odd offset, read in chunks that start mid-block
        let (offset, tensor) = (4099, 64 * 600);
        let (path, file, data) = random_file("sliced", offset + tensor + 777);
        let intervals = [vec![3..60], vec![10..50, 200..260]];
        let indexers: Vec<Vec<TensorIndexer>> = intervals
            .iter()
            .map(|d| d.iter().map(|r| TensorIndexer::from(r.clone())).collect())
            .collect();
        let r = slice_layout(Dtype::U16, &[64, 300], &indexers).unwrap();
        let packed = r.gather.as_ref().unwrap().len;
        let plan = LoadPlan::new(
            vec![sp(r.read.start, r.read.end), sp(tensor, tensor + 777)],
            vec![r.gather, None],
            offset,
            NonZeroUsize::new(5000).unwrap(),
        );
        let (dsts, len) = ([0, packed], packed + 777);
        let mut expected = slice_bytes(&data[offset..], &[64, 300], 2, &intervals);
        expected.extend_from_slice(&data[offset + tensor..]);
        for (direct, cold) in [(Some(true), true), (Some(false), false)] {
            let source = DirectFile::new(file.clone(), direct, true);
            if cold {
                evict(&file);
            }
            let got = land_read(&plan, &dsts, len, |slab, at, n| {
                source.read(slab, at, n).unwrap()
            });
            assert_eq!(got, expected, "{direct:?}");
        }
        std::fs::remove_file(path).unwrap();
    }
}
