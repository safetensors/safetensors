//! Memory that ranges of a file map to, as a [`Sink`] that scatters chunks into
//! it. Two units, computed once from the extents, sorted by start and disjoint:
//!
//! ```text
//! tensors  |t0 |  t1  |     t2     |  t3  |       t4        | t5 | t6  |
//! spans    |S0#|..|S1#|S2#|...............|#######S3########|#S4#|#S5##|
//! chunks   |c0 |  |  c1   |               |   c2   |   c3   |   c4   |c|
//! ```
//!
//! - spans [`LoadPlan::spans`]: a byte interval of the data section, copied to
//!   its destination, or, with a [`slice::Gather`], packed into it by the gather's
//!   strided copies.
//! - chunks ([`LoadPlan::chunks`]): at most [`CHUNK_SIZE`](super::transfer::CHUNK_SIZE) of adjacent spans,
//!   never across a gap. The I/O unit: one read, one slab, one copy per run of
//!   spans whose destinations are adjacent. `span_chunks` and `chunk_spans` map
//!   one to the other.
//!
//! The memory's [`Scatter`] makes the copies: host memory, or a device's.

use std::{
    num::NonZeroUsize,
    ops::Range,
    sync::{
        atomic::{AtomicBool, AtomicUsize, Ordering},
        Arc, Condvar, Mutex, OnceLock,
    },
    time::Duration,
};

use super::{
    pool::Slab,
    transfer::{Error, Fence, Sink, Source, Staged},
};
use safetensors::slice::{self, StridedCopy};

/// A byte interval `start..end` of the file's data section
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Span {
    pub start: usize,
    pub end: usize,
}

/// Geometry of one load. Every interval here is a byte range within the file's
/// data section; device addresses never appear in the plan.
pub struct LoadPlan {
    spans: Box<[Span]>,
    /// per span, the copies that pack its bytes into its destination; `None` for
    /// one plain copy
    gathers: Box<[Option<slice::Gather>]>,
    /// per span, the `chunks` that carry its bytes; empty for a zero-size span
    span_chunks: Box<[Range<usize>]>,
    /// data section byte interval of each chunk: at most [`CHUNK_SIZE`](super::transfer::CHUNK_SIZE) of
    /// adjacent spans, never across a gap
    chunks: Box<[Range<usize>]>,
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
        gathers: Vec<Option<slice::Gather>>,
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
    fn file_chunks(&self) -> Box<[Range<u64>]> {
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

    /// The spans that meet `file`, a byte range of the file
    fn spans_in(&self, file: &Range<u64>) -> Range<usize> {
        let (start, end) = (
            file.start as usize - self.in_file_offset,
            file.end as usize - self.in_file_offset,
        );
        let first = self.spans.partition_point(|s| s.end <= start);
        first..first + self.spans[first..].partition_point(|s| s.start < end)
    }

    /// The copy of the bytes of span `s` in `file` from `addr`, where span `s`
    /// is, to their offset in `file`
    fn copy_from(&self, s: usize, file: &Range<u64>, addr: u64) -> Option<StridedCopy> {
        assert!(
            self.gathers[s].is_none(),
            "a gather is copied to memory only"
        );
        let (span, at) = (self.spans[s], file.start as usize - self.in_file_offset);
        let (start, end) = (
            span.start.max(at),
            span.end.min(file.end as usize - self.in_file_offset),
        );
        (start < end).then(|| StridedCopy {
            src: addr as usize + (start - span.start),
            dst: start - at,
            width: end - start,
            height: 1,
            src_pitch: 0,
            dst_pitch: 0,
        })
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

/// Copies chunks into the memory a [`Memory`]'s destinations are in: host
/// memory, or a device's
pub trait Scatter: Send + Sync {
    /// Copies the staged bytes to their destinations as `copies` say: sources
    /// are offsets in [`Staged::bytes`], destinations addresses in this space.
    /// Keeps `staged` until the copies read it, and returns a fence when they
    /// complete after the call. On an error, no copy is left in flight.
    fn scatter(
        &self,
        staged: Staged,
        copies: &[StridedCopy],
    ) -> Result<Option<Arc<dyn Fence>>, Error>;

    /// The end of the allocation holding `addr`, when known: adjacent
    /// destinations are copied at once only within one allocation.
    fn allocation_end(&self, _addr: u64) -> Option<u64> {
        None
    }
}

/// Copies chunks out of the memory a [`Memory`]'s destinations are in
pub trait Gather: Send + Sync {
    /// Copies bytes to `dst` as `copies` say: sources are addresses in this
    /// memory, destinations offsets in `dst`. Complete on return.
    fn gather(&self, dst: &mut [u8], copies: &[StridedCopy]) -> Result<(), Error>;
}

/// Host memory, copied by the worker
#[cfg(all(test, target_os = "linux"))]
pub struct Host;

#[cfg(all(test, target_os = "linux"))]
impl Gather for Host {
    fn gather(&self, dst: &mut [u8], copies: &[StridedCopy]) -> Result<(), Error> {
        for c in copies {
            debug_assert_eq!(c.height, 1);
            // SAFETY: `Memory::provide` requires the address readable for its
            // span's bytes, which `c` is within
            let src = unsafe { std::slice::from_raw_parts(c.src as *const u8, c.width) };
            dst[c.dst..c.dst + c.width].copy_from_slice(src);
        }
        Ok(())
    }
}

/// A chunk's copies once issued: their fence, `None` when they completed
/// in [`Scatter::scatter`]
type Issued = Option<Arc<dyn Fence>>;

/// Where a span's bytes go, once `ready` completed
struct Destination {
    addr: u64,
    ready: Option<Arc<dyn Fence>>,
}

/// The destinations of a [`LoadPlan`]'s spans in some memory. They arrive in
/// plan order as the transfer advances, see [`Self::provide`]: draining a
/// chunk waits for those of its spans.
pub struct Memory<S> {
    plan: LoadPlan,
    space: S,
    /// per span, set by `provide`
    dests: Box<[OnceLock<Destination>]>,
    /// the spans with a destination: a prefix of the plan
    provided: AtomicUsize,
    /// guards `provided` advancing and `stopped` turning on, so neither is
    /// missed by a waiter on `provision`
    provision_lock: Mutex<()>,
    provision: Condvar,
    stopped: AtomicBool,
    /// per chunk, set once its copies are issued
    landed: Box<[OnceLock<Issued>]>,
}

impl<S> Memory<S> {
    pub fn new(plan: LoadPlan, space: S) -> Self {
        Self {
            dests: (0..plan.spans.len()).map(|_| OnceLock::new()).collect(),
            provided: AtomicUsize::new(0),
            provision_lock: Mutex::new(()),
            provision: Condvar::new(),
            stopped: AtomicBool::new(false),
            landed: (0..plan.chunks.len()).map(|_| OnceLock::new()).collect(),
            plan,
            space,
        }
    }

    /// How many spans the plan has
    pub fn spans(&self) -> usize {
        self.plan.spans.len()
    }

    /// The plan's chunks, as byte ranges of the file
    pub fn chunks(&self) -> Box<[Range<u64>]> {
        self.plan.file_chunks()
    }

    /// How many spans have a destination
    pub fn provided(&self) -> usize {
        self.provided.load(Ordering::Acquire)
    }

    /// Gives the next spans, in plan order, their destination addresses. The
    /// copies to them wait for `ready`. One caller at a time.
    ///
    /// # Safety
    ///
    /// Each address is valid for writes of its span's bytes (or its gather's
    /// [`slice::Gather::len`]) in this memory, until
    /// every copy to it completed: [`Self::wait_landed`] returned for its
    /// span, or [`Self::wait_copies`] once no worker is left. When the memory
    /// is a [`Source`], each address is valid for reads of its span's bytes
    /// until the transfer returns.
    pub unsafe fn provide(
        &self,
        addrs: impl ExactSizeIterator<Item = u64>,
        ready: Option<Arc<dyn Fence>>,
    ) {
        let _g = self.provision_lock.lock().unwrap();
        let first = self.provided.load(Ordering::Acquire);
        assert!(
            first + addrs.len() <= self.dests.len(),
            "more destinations than spans"
        );
        let n = addrs.len();
        for (dest, addr) in self.dests[first..].iter().zip(addrs) {
            let ready = ready.clone();
            assert!(dest.set(Destination { addr, ready }).is_ok());
        }
        self.provided.store(first + n, Ordering::Release);
        self.provision.notify_all();
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
            if self.stopped.load(Ordering::Acquire) {
                return Err(Error::Stopped);
            }
            g = self.provision.wait(g).unwrap();
        }
    }

    fn dest(&self, span: usize) -> &Destination {
        self.dests[span].get().expect("span provided")
    }

    /// Waits until span `span`'s bytes landed.
    pub fn wait_landed(&self, span: usize) -> Result<(), Error> {
        // a zero-size span has no chunk: it is ready once provided
        self.wait_provided(span + 1)?;
        for chunk in self.plan.span_chunks[span].clone() {
            let mut i = 0;
            let fence = loop {
                match self.landed[chunk].get() {
                    Some(fence) => break fence,
                    None if self.stopped.load(Ordering::Acquire) => return Err(Error::Stopped),
                    None if i < 64 => {
                        std::thread::yield_now();
                        i += 1;
                    }
                    None => std::thread::sleep(Duration::from_micros(200)),
                }
            };
            if let Some(fence) = fence {
                fence.wait()?;
            }
        }
        Ok(())
    }

    /// Whether span `span`'s copies are all issued
    pub fn landed(&self, span: usize) -> bool {
        self.provided.load(Ordering::Acquire) > span
            && self.plan.span_chunks[span]
                .clone()
                .all(|c| self.landed[c].get().is_some())
    }

    /// How many spans the first `claimed` chunks reach
    pub fn scheduled(&self, claimed: usize) -> usize {
        match claimed.min(self.plan.chunks.len()) {
            n if n == self.plan.chunks.len() => self.plan.spans.len(),
            0 => 0,
            n => self.plan.chunk_spans[n - 1].end,
        }
    }

    /// Waits for every copy issued. Call once no worker is left.
    pub fn wait_copies(&self) -> Result<(), Error> {
        self.landed
            .iter()
            .filter_map(|l| l.get().and_then(Option::as_ref))
            .try_for_each(|fence| fence.wait())
    }
}

impl<S: Gather> Source for Memory<S> {
    fn fill(&self, file: Range<u64>, mut slab: Slab) -> Result<Staged, Error> {
        let len = (file.end - file.start) as usize;
        let spans = self.plan.spans_in(&file);
        self.wait_provided(spans.end)?;
        let copies: Vec<StridedCopy> = spans
            .filter_map(|s| self.plan.copy_from(s, &file, self.dest(s).addr))
            .collect();
        self.space.gather(&mut slab[..len], &copies)?;
        Ok(Staged::new(slab, 0, file))
    }
}

impl<S: Scatter> Sink for Memory<S> {
    fn wait_ready(&self, file: &Range<u64>) -> Result<(), Error> {
        let spans = self.plan.chunk_spans[self.plan.chunk_at(file)].clone();
        self.wait_provided(spans.end)?;
        // host-side: a pending copy would hold its slab and stall other loaders
        let mut last: Option<&Arc<dyn Fence>> = None;
        for s in spans {
            if let Some(ready) = &self.dest(s).ready {
                if !last.is_some_and(|l| Arc::ptr_eq(l, ready)) {
                    ready.wait()?;
                    last = Some(ready);
                }
            }
        }
        Ok(())
    }

    fn drain(&self, staged: Staged) -> Result<(), Error> {
        let chunk = self.plan.chunk_at(staged.file());
        let mut copies = Vec::new();
        self.plan.chunk_copies(
            chunk,
            |s| self.dest(s).addr,
            |addr| self.space.allocation_end(addr),
            |copy| {
                copies.push(copy);
                Ok::<_, Error>(())
            },
        )?;
        let fence = self.space.scatter(staged, &copies)?;
        let _ = self.landed[chunk].set(fence);
        Ok(())
    }

    fn stop(&self) {
        let _g = self.provision_lock.lock().unwrap();
        self.stopped.store(true, Ordering::Release);
        self.provision.notify_all();
    }
}

#[cfg(test)]
#[allow(clippy::single_range_in_vec_init)] // one range per dimension
mod tests {
    use std::{
        num::NonZeroUsize,
        ops::Range,
        sync::{Arc, Mutex},
        time::Duration,
    };

    use super::{LoadPlan, Memory, Scatter, Span};
    #[cfg(target_os = "linux")]
    use crate::engine::file::{
        tests::{evict, random_file},
        DirectFile,
    };
    use crate::engine::{
        file::tests::{slab_buf, Rng},
        pool::SLAB_SIZE,
    };
    use crate::engine::{
        pool::{tests::heap_pool, Slab},
        transfer::{work, Error, Fence, Plan, Source, Staged},
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

    /// A file's bytes in memory
    struct Bytes(Vec<u8>);

    impl Source for Bytes {
        fn fill(&self, file: Range<u64>, mut slab: Slab) -> Result<Staged, Error> {
            let (start, end) = (file.start as usize, file.end as usize);
            slab[..end - start].copy_from_slice(&self.0[start..end]);
            Ok(Staged::new(slab, 0, file))
        }
    }

    /// A space whose addresses are offsets in a buffer; every copy fails with `fail`
    struct Buffer {
        bytes: Mutex<Vec<u8>>,
        fail: bool,
    }

    impl Scatter for Buffer {
        fn scatter(
            &self,
            staged: Staged,
            copies: &[StridedCopy],
        ) -> Result<Option<Arc<dyn Fence>>, Error> {
            if self.fail {
                return Err(Error::Other("copy failed".into()));
            }
            let (src, mut dst) = (staged.bytes(), self.bytes.lock().unwrap());
            for c in copies {
                for r in 0..c.height {
                    let (from, to) = (c.src + r * c.src_pitch, c.dst + r * c.dst_pitch);
                    dst[to..to + c.width].copy_from_slice(&src[from..from + c.width]);
                }
            }
            Ok(None)
        }
    }

    #[test]
    fn test_memory_takes_destinations_as_they_arrive() {
        // a span, a gap, columns of a 4 x 10 tensor, then a span: 7-byte chunks
        // wait for the second batch of destinations
        let gather = Gather {
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
        let plan = LoadPlan::new(
            vec![sp(0, 7), sp(12, 45), sp(50, 60)],
            vec![None, Some(gather), None],
            0,
            NonZeroUsize::new(7).unwrap(),
        );
        let bytes = Mutex::new(vec![0xAA; 29]);
        let memory = Memory::new(plan, Buffer { bytes, fail: false });
        let (transfer, pool) = (Plan::new(memory.chunks()), heap_pool(2, 7));
        let file = Bytes((0..60).collect());
        std::thread::scope(|s| {
            let worker = s.spawn(|| work(&transfer, &file, &memory, &pool));
            // SAFETY: offsets in the buffer, which `memory` holds
            unsafe { memory.provide([0].into_iter(), None) };
            std::thread::sleep(Duration::from_millis(20));
            // SAFETY: as above
            unsafe { memory.provide([7, 19].into_iter(), None) };
            worker.join().unwrap();
        });
        assert!(transfer.error().is_none());
        (0..3).for_each(|span| memory.wait_landed(span).unwrap());
        let rows = (0..4).flat_map(|r| 12 + r * 10..15 + r * 10);
        let expected: Vec<u8> = (0..7).chain(rows).chain(50..60).collect();
        assert_eq!(*memory.space.bytes.lock().unwrap(), expected);
    }

    #[test]
    fn test_failed_copy_stops_the_transfer() {
        let plan = LoadPlan::new(
            vec![sp(0, 40)],
            vec![None],
            0,
            NonZeroUsize::new(8).unwrap(),
        );
        let bytes = Mutex::new(vec![0; 40]);
        let memory = Memory::new(plan, Buffer { bytes, fail: true });
        let (transfer, pool) = (Plan::new(memory.chunks()), heap_pool(2, 8));
        let file = Bytes(vec![1; 40]);
        // SAFETY: an offset in the buffer, which `memory` holds
        unsafe { memory.provide([0].into_iter(), None) };
        std::thread::scope(|s| {
            for _ in 0..2 {
                s.spawn(|| work(&transfer, &file, &memory, &pool));
            }
        });
        assert_eq!(transfer.error().unwrap().to_string(), "copy failed");
        assert!(matches!(memory.wait_landed(0), Err(Error::Stopped)));
    }

    #[cfg(target_os = "linux")]
    #[test]
    fn test_memory_gathers_any_window() {
        // three tensors back to back, the middle one empty, read in windows that
        // cut them
        let tensors: [Vec<u8>; 3] = [(0..5).collect(), vec![], (5..12).collect()];
        let plan = LoadPlan::new(
            vec![sp(0, 5), sp(5, 5), sp(5, 12)],
            vec![None; 3],
            0,
            NonZeroUsize::new(4).unwrap(),
        );
        let memory = Memory::new(plan, super::Host);
        // SAFETY: each tensor outlives `memory`
        unsafe { memory.provide(tensors.iter().map(|t| t.as_ptr() as u64), None) };
        let pool = heap_pool(1, 16);
        for window in [0..12, 0..3, 3..9, 5..6, 11..12] {
            let staged = memory
                .fill(window.clone(), pool.acquire().unwrap())
                .unwrap();
            let expected: Vec<u8> = (window.start as u8..window.end as u8).collect();
            assert_eq!(staged.bytes(), expected, "{window:?}");
        }
    }
}
