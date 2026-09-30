//! Module handling lazy loading via iterating on slices on the original buffer.
use crate::lib::Vec;
use crate::tensor::{Dtype, TensorView};
use core::fmt::Display;
use core::num::NonZeroUsize;
use core::ops::{
    Bound, Range, RangeBounds, RangeFrom, RangeFull, RangeInclusive, RangeTo, RangeToInclusive,
};

/// Error representing invalid slicing attempt
#[derive(Debug, PartialEq, Eq)]
pub enum InvalidSlice {
    /// When the client asked for more slices than the tensors has dimensions
    TooManySlices,
    /// When the client asked for a slice that exceeds the allowed bounds
    SliceOutOfRange {
        /// The rank of the dimension that has the out of bounds
        dim_index: usize,
        /// The problematic value
        asked: usize,
        /// The dimension size we shouldn't go over.
        dim_size: usize,
    },
    /// For smaller than 1 byte dtypes, some slices will happen outside of the byte boundary, some special care has to be taken
    /// and standard functions will fail
    MisalignedSlice,
    /// The ranges given for one dimension of a region overlap or are not in increasing order
    UnorderedRanges {
        /// The rank of the dimension
        dim_index: usize,
    },
    /// The region would take more than [`MAX_REGION_COPIES`] strided copies to gather
    TooFragmented {
        /// The number of copies it would take
        copies: usize,
    },
}

impl Display for InvalidSlice {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match *self {
            InvalidSlice::TooManySlices => {
                write!(f, "more slicing indexes than dimensions in tensor")
            }
            InvalidSlice::SliceOutOfRange {
                dim_index,
                asked,
                dim_size,
            } => {
                write!(f, "index {asked} out of bounds for tensor dimension #{dim_index} of size {dim_size}")
            }
            InvalidSlice::MisalignedSlice => {
                write!(f, "The slice is slicing for subbytes dtypes, and the slice does not end up at a byte boundary, this is invalid.")
            }
            InvalidSlice::UnorderedRanges { dim_index } => {
                write!(f, "the ranges of tensor dimension #{dim_index} overlap or are not in increasing order")
            }
            InvalidSlice::TooFragmented { copies } => {
                write!(f, "the region would take {copies} strided copies to gather, more than {MAX_REGION_COPIES}")
            }
        }
    }
}

#[cfg(feature = "std")]
impl std::error::Error for InvalidSlice {}

#[cfg(not(feature = "std"))]
impl core::error::Error for InvalidSlice {}

#[derive(Debug, Clone)]
/// Generic structure used to index a slice of the tensor
pub enum TensorIndexer {
    /// This is selecting an entire dimension
    Select(usize),
    /// A slice `start:stop:step`. `step` is always >= 1; a contiguous slice
    /// has `step == 1`.
    Narrow(Bound<usize>, Bound<usize>, NonZeroUsize),
}

fn display_bound(bound: &Bound<usize>) -> &dyn Display {
    match bound {
        Bound::Unbounded => &"",
        Bound::Excluded(n) => n,
        Bound::Included(n) => n,
    }
}

/// Intended for Python users mostly or at least for its conventions
impl Display for TensorIndexer {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            TensorIndexer::Select(n) => {
                write!(f, "{n}")
            }
            TensorIndexer::Narrow(left, right, step) => {
                if step.get() == 1 {
                    write!(f, "{}:{}", display_bound(left), display_bound(right))
                } else {
                    write!(f, "{}:{}:{step}", display_bound(left), display_bound(right))
                }
            }
        }
    }
}

impl From<usize> for TensorIndexer {
    fn from(index: usize) -> Self {
        TensorIndexer::Select(index)
    }
}

// impl From<&[usize]> for TensorIndexer {
//     fn from(index: &[usize]) -> Self {
//         let tensor = index.into();
//         TensorIndexer::IndexSelect(tensor)
//     }
// }
//
// impl From<Vec<usize>> for TensorIndexer {
//     fn from(index: Vec<usize>) -> Self {
//         let tensor = Tensor::of_slice(&index);
//         TensorIndexer::IndexSelect(tensor)
//     }
// }

macro_rules! impl_from_range {
    ($range_type:ty) => {
        impl From<$range_type> for TensorIndexer {
            fn from(range: $range_type) -> Self {
                use core::ops::Bound::*;

                let start = match range.start_bound() {
                    Included(idx) => Included(*idx),
                    Excluded(idx) => Excluded(*idx),
                    Unbounded => Unbounded,
                };

                let end = match range.end_bound() {
                    Included(idx) => Included(*idx),
                    Excluded(idx) => Excluded(*idx),
                    Unbounded => Unbounded,
                };

                TensorIndexer::Narrow(start, end, NonZeroUsize::MIN)
            }
        }
    };
}

impl_from_range!(Range<usize>);
impl_from_range!(RangeFrom<usize>);
impl_from_range!(RangeFull);
impl_from_range!(RangeInclusive<usize>);
impl_from_range!(RangeTo<usize>);
impl_from_range!(RangeToInclusive<usize>);

/// Trait used to implement multiple signatures for ease of use of the slicing
/// of a tensor
pub trait IndexOp<'data, T> {
    /// Returns a slicing iterator which are the chunks of data necessary to
    /// reconstruct the desired tensor.
    fn slice(&'data self, index: T) -> Result<SliceIterator<'data>, InvalidSlice>;
}

impl<'data, A> IndexOp<'data, A> for TensorView<'data>
where
    A: Into<TensorIndexer>,
{
    fn slice(&'data self, index: A) -> Result<SliceIterator<'data>, InvalidSlice> {
        self.sliced_data(&[index.into()])
    }
}

impl<'data, A> IndexOp<'data, (A,)> for TensorView<'data>
where
    A: Into<TensorIndexer>,
{
    fn slice(&'data self, index: (A,)) -> Result<SliceIterator<'data>, InvalidSlice> {
        let idx_a = index.0.into();
        self.sliced_data(&[idx_a])
    }
}

impl<'data, A, B> IndexOp<'data, (A, B)> for TensorView<'data>
where
    A: Into<TensorIndexer>,
    B: Into<TensorIndexer>,
{
    fn slice(&'data self, index: (A, B)) -> Result<SliceIterator<'data>, InvalidSlice> {
        let idx_a = index.0.into();
        let idx_b = index.1.into();
        self.sliced_data(&[idx_a, idx_b])
    }
}

impl<'data, A, B, C> IndexOp<'data, (A, B, C)> for TensorView<'data>
where
    A: Into<TensorIndexer>,
    B: Into<TensorIndexer>,
    C: Into<TensorIndexer>,
{
    fn slice(&'data self, index: (A, B, C)) -> Result<SliceIterator<'data>, InvalidSlice> {
        let idx_a = index.0.into();
        let idx_b = index.1.into();
        let idx_c = index.2.into();
        self.sliced_data(&[idx_a, idx_b, idx_c])
    }
}

// impl<A, B, C, D> IndexOp<(A, B, C, D)> for TensorView<'data>
// where
//     A: Into<TensorIndexer>,
//     B: Into<TensorIndexer>,
//     C: Into<TensorIndexer>,
//     D: Into<TensorIndexer>,
// {
//     fn slice(&self, index: (A, B, C, D)) -> TensorView<'data> {
//         let idx_a = index.0.into();
//         let idx_b = index.1.into();
//         let idx_c = index.2.into();
//         let idx_d = index.3.into();
//         self.sliced_data(&[idx_a, idx_b, idx_c, idx_d])
//     }
// }
//
// impl<A, B, C, D, E> IndexOp<(A, B, C, D, E)> for TensorView<'data>
// where
//     A: Into<TensorIndexer>,
//     B: Into<TensorIndexer>,
//     C: Into<TensorIndexer>,
//     D: Into<TensorIndexer>,
//     E: Into<TensorIndexer>,
// {
//     fn slice(&self, index: (A, B, C, D, E)) -> TensorView<'data> {
//         let idx_a = index.0.into();
//         let idx_b = index.1.into();
//         let idx_c = index.2.into();
//         let idx_d = index.3.into();
//         let idx_e = index.4.into();
//         self.sliced_data(&[idx_a, idx_b, idx_c, idx_d, idx_e])
//     }
// }
//
// impl<A, B, C, D, E, F> IndexOp<(A, B, C, D, E, F)> for TensorView<'data>
// where
//     A: Into<TensorIndexer>,
//     B: Into<TensorIndexer>,
//     C: Into<TensorIndexer>,
//     D: Into<TensorIndexer>,
//     E: Into<TensorIndexer>,
//     F: Into<TensorIndexer>,
// {
//     fn slice(&self, index: (A, B, C, D, E, F)) -> TensorView<'data> {
//         let idx_a = index.0.into();
//         let idx_b = index.1.into();
//         let idx_c = index.2.into();
//         let idx_d = index.3.into();
//         let idx_e = index.4.into();
//         let idx_f = index.5.into();
//         self.sliced_data(&[idx_a, idx_b, idx_c, idx_d, idx_e, idx_f])
//     }
// }
//
// impl<A, B, C, D, E, F, G> IndexOp<(A, B, C, D, E, F, G)> for TensorView<'data>
// where
//     A: Into<TensorIndexer>,
//     B: Into<TensorIndexer>,
//     C: Into<TensorIndexer>,
//     D: Into<TensorIndexer>,
//     E: Into<TensorIndexer>,
//     F: Into<TensorIndexer>,
//     G: Into<TensorIndexer>,
// {
//     fn slice(&self, index: (A, B, C, D, E, F, G)) -> TensorView<'data> {
//         let idx_a = index.0.into();
//         let idx_b = index.1.into();
//         let idx_c = index.2.into();
//         let idx_d = index.3.into();
//         let idx_e = index.4.into();
//         let idx_f = index.5.into();
//         let idx_g = index.6.into();
//         self.sliced_data(&[idx_a, idx_b, idx_c, idx_d, idx_e, idx_f, idx_g])
//     }
// }

/// Iterator used to return the bits of the overall tensor buffer
/// when client asks for a slice of the original tensor.
#[cfg_attr(test, derive(Debug, Eq, PartialEq))]
pub struct SliceIterator<'data> {
    view: &'data TensorView<'data>,
    indices: Vec<(usize, usize)>,
    newshape: Vec<usize>,
}

impl<'data> SliceIterator<'data> {
    pub(crate) fn new(
        view: &'data TensorView<'data>,
        slices: &[TensorIndexer],
    ) -> Result<Self, InvalidSlice> {
        let (indices, newshape) = slice_byte_ranges(view.dtype(), view.shape(), slices)?;
        // Reversing so we can pop faster while iterating on the slice
        let indices = indices.into_iter().rev().collect();
        Ok(Self {
            view,
            indices,
            newshape,
        })
    }

    /// Gives back the amount of bytes still being in the iterator
    pub fn remaining_byte_len(&self) -> usize {
        self.indices.iter().map(|(start, stop)| stop - start).sum()
    }

    /// Gives back the amount of bytes still being in the iterator
    pub fn newshape(&self) -> Vec<usize> {
        self.newshape.clone()
    }
}

/// Byte ranges into a tensor's data section in iteration order;
/// concatenating them yields the dense destination layout.
pub type SliceByteRanges = Vec<(usize, usize)>;

/// Post-slice tensor shape (element counts per dim).
pub type SlicedShape = Vec<usize>;

/// Resolve a `(start, stop)` half-open element range from slice bounds,
/// defaulting unbounded ends to `0` and `dim`.
fn narrow_bounds(left: &Bound<usize>, right: &Bound<usize>, dim: usize) -> (usize, usize) {
    let start = match left {
        Bound::Unbounded => 0,
        Bound::Included(s) => *s,
        Bound::Excluded(s) => *s + 1,
    };
    let stop = match right {
        Bound::Unbounded => dim,
        Bound::Included(s) => *s + 1,
        Bound::Excluded(s) => *s,
    };
    (start, stop)
}

/// Compute the byte ranges and post-slice shape for a slicing operation
/// without requiring the underlying data buffer.
///
/// The returned [`SliceByteRanges`] is in source iteration order; callers
/// may reverse it for pop-based iteration.
pub fn slice_byte_ranges(
    dtype: Dtype,
    shape: &[usize],
    slices: &[TensorIndexer],
) -> Result<(SliceByteRanges, SlicedShape), InvalidSlice> {
    let n_slice = slices.len();
    let n_shape = shape.len();
    if n_slice > n_shape {
        return Err(InvalidSlice::TooManySlices);
    }
    let mut newshape = Vec::with_capacity(n_shape);

    // Minimum span is the span of 1 item;
    let mut span = dtype.bitsize();
    let mut indices: Vec<(usize, usize)> = vec![];
    // Everything is row major.
    for (i, &dim) in shape.iter().enumerate().rev() {
        if i >= slices.len() {
            // We are not slicing yet, just increase the local span
            newshape.push(dim);
        } else {
            let slice = &slices[i];
            let (start, stop, step) = match slice {
                TensorIndexer::Select(s) => (*s, *s + 1, 1),
                TensorIndexer::Narrow(left, right, step) => {
                    let (start, stop) = narrow_bounds(left, right, dim);
                    (start, stop, step.get())
                }
            };
            if start >= dim || stop > dim {
                let asked = if start >= dim {
                    start
                } else {
                    stop.saturating_sub(1)
                };
                return Err(InvalidSlice::SliceOutOfRange {
                    dim_index: i,
                    asked,
                    dim_size: dim,
                });
            }
            if !matches!(slice, TensorIndexer::Select(_)) {
                newshape.push((stop - start).div_ceil(step));
            }
            if indices.is_empty() {
                if step == 1 && start == 0 && stop == dim {
                    // Full range, nothing sliced yet; just grow the span.
                } else if step == 1 {
                    if start * span % 8 != 0 {
                        return Err(InvalidSlice::MisalignedSlice);
                    }
                    let offset = (start * span) / 8;
                    if stop * span % 8 != 0 {
                        return Err(InvalidSlice::MisalignedSlice);
                    }
                    let small_span = (stop * span) / 8 - offset;
                    indices.push((offset, offset + small_span));
                } else {
                    // Strided innermost dim: each kept element is its own run.
                    for n in (start..stop).step_by(step) {
                        if n * span % 8 != 0 || (n + 1) * span % 8 != 0 {
                            return Err(InvalidSlice::MisalignedSlice);
                        }
                        indices.push(((n * span) / 8, ((n + 1) * span) / 8));
                    }
                }
            } else {
                let capacity = (stop - start).div_ceil(step) * indices.len();
                let mut newindices = Vec::with_capacity(capacity);
                for n in (start..stop).step_by(step) {
                    if n * span % 8 != 0 {
                        return Err(InvalidSlice::MisalignedSlice);
                    }
                    let offset = (n * span) / 8;
                    for (old_start, old_stop) in &indices {
                        newindices.push((old_start + offset, old_stop + offset));
                    }
                }
                indices = newindices;
            }
        }
        span *= dim;
    }
    if indices.is_empty() {
        // Empty `slices` (or all unbounded full-range slices): no slicing
        // happened, the whole tensor is the result. `span` ended as
        // bitsize * product(shape).
        let total_bits = span;
        if total_bits % 8 != 0 {
            return Err(InvalidSlice::MisalignedSlice);
        }
        indices.push((0, total_bits / 8));
    }
    let newshape = newshape.into_iter().rev().collect();
    Ok((indices, newshape))
}

/// `height` runs of `width` bytes, `src_pitch` apart in the source (offsets from the first byte of
/// [`Region::read`]) and `dst_pitch` apart in the compact destination.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct StridedCopy {
    /// Offset of the first run in the source
    pub src: usize,
    /// Offset of the first run in the destination
    pub dst: usize,
    /// Bytes per run
    pub width: usize,
    /// Number of runs
    pub height: usize,
    /// Bytes between the starts of two runs in the source
    pub src_pitch: usize,
    /// Bytes between the starts of two runs in the destination
    pub dst_pitch: usize,
}

/// The copies that turn the bytes read for a [`Region`] into its compact result, `len` bytes long.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Gather {
    /// The strided copies, which together fill the result exactly once
    pub copies: Vec<StridedCopy>,
    /// The result's size in bytes
    pub len: usize,
}

/// Where a region of a tensor lives in its bytes.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Region {
    /// The bytes to read, relative to the tensor's first byte
    pub read: Range<usize>,
    /// The result's shape (dimensions indexed with [`TensorIndexer::Select`] dropped)
    pub shape: Vec<usize>,
    /// `None` when the bytes read are the result as is
    pub gather: Option<Gather>,
}

/// A region taking more strided copies than this is refused ([`InvalidSlice::TooFragmented`]).
pub const MAX_REGION_COPIES: usize = 1 << 16;

/// Plans a region of a row-major tensor: the bytes to read and, when they aren't the result as is, the strided
/// copies that gather the result compactly.
///
/// `slices` holds one list of indexers per leading dimension (dimensions left out are kept whole). A list with
/// several entries keeps all of them, in order, along that dimension; a single [`TensorIndexer::Select`] drops the
/// dimension from the result, as it does when slicing. Full trailing dimensions fold into the element and a
/// dimension whose inner neighbour is kept whole merges with it, so a column slice of any rank is one strided copy.
#[allow(clippy::single_range_in_vec_init)] // one range per dimension
pub fn slice_region(
    dtype: Dtype,
    shape: &[usize],
    slices: &[Vec<TensorIndexer>],
) -> Result<Region, InvalidSlice> {
    if slices.len() > shape.len() {
        return Err(InvalidSlice::TooManySlices);
    }
    // one sorted, disjoint list of element ranges per dimension, and whether the dimension stays in the result
    let mut ranges: Vec<Vec<Range<usize>>> = Vec::with_capacity(shape.len());
    let mut result_shape = Vec::with_capacity(shape.len());
    for (i, &dim) in shape.iter().enumerate() {
        let Some(indexers) = slices.get(i) else {
            ranges.push(vec![0..dim]);
            result_shape.push(dim);
            continue;
        };
        let out_of_range = |asked| InvalidSlice::SliceOutOfRange {
            dim_index: i,
            asked,
            dim_size: dim,
        };
        let mut kept: Vec<Range<usize>> = Vec::new();
        for indexer in indexers {
            match indexer {
                TensorIndexer::Select(s) => {
                    if *s >= dim {
                        return Err(out_of_range(*s));
                    }
                    kept.push(*s..*s + 1);
                }
                TensorIndexer::Narrow(left, right, step) => {
                    let (start, stop) = narrow_bounds(left, right, dim);
                    if stop > dim {
                        return Err(out_of_range(stop.saturating_sub(1)));
                    }
                    if start >= stop {
                        continue; // empty
                    }
                    if step.get() == 1 {
                        kept.push(start..stop);
                    } else {
                        kept.extend((start..stop).step_by(step.get()).map(|n| n..n + 1));
                    }
                }
            }
        }
        if kept.windows(2).any(|w| w[0].end > w[1].start) {
            return Err(InvalidSlice::UnorderedRanges { dim_index: i });
        }
        // touching ranges are one: fewer copies, and a whole dimension stays whole
        let mut merged: Vec<Range<usize>> = Vec::with_capacity(kept.len());
        for r in kept {
            match merged.last_mut() {
                Some(last) if last.end == r.start => last.end = r.end,
                _ => merged.push(r),
            }
        }
        if !matches!(indexers.as_slice(), [TensorIndexer::Select(_)]) {
            result_shape.push(merged.iter().map(|r| r.len()).sum());
        }
        ranges.push(merged);
    }
    if ranges.iter().any(|d| d.is_empty()) {
        return Ok(Region {
            read: 0..0,
            shape: result_shape,
            gather: None,
        });
    }

    let whole = |d: usize| ranges[d].len() == 1 && ranges[d][0] == (0..shape[d]);
    // (size, ranges) from the outermost dimension in; full trailing dimensions fold into the element
    let mut elem_bits = dtype.bitsize();
    let mut innermost_partial = shape.len();
    while innermost_partial > 0 && whole(innermost_partial - 1) {
        innermost_partial -= 1;
        elem_bits *= shape[innermost_partial];
    }
    let mut dims: Vec<(usize, Vec<Range<usize>>)> = (0..innermost_partial)
        .map(|d| (shape[d], ranges[d].clone()))
        .collect();
    // a dimension whose inner neighbour is kept whole takes it in: `[E, rows, cols[a..b]]` with every row kept
    // becomes `[E * rows, cols[a..b]]`
    let mut i = dims.len().saturating_sub(1);
    while i > 0 {
        if dims[i].1.len() == 1 && dims[i].1[0] == (0..dims[i].0) {
            let (inner, _) = dims.remove(i);
            let (size, rs) = &mut dims[i - 1];
            *size *= inner;
            for r in rs.iter_mut() {
                *r = r.start * inner..r.end * inner;
            }
        }
        i -= 1;
    }
    let byte = |bits: usize| -> Result<usize, InvalidSlice> {
        if bits % 8 == 0 {
            Ok(bits / 8)
        } else {
            Err(InvalidSlice::MisalignedSlice)
        }
    };
    if dims.is_empty() {
        return Ok(Region {
            read: 0..byte(elem_bits)?,
            shape: result_shape,
            gather: None,
        });
    }

    // strides in bits, source layout and compact result layout
    let n = dims.len();
    let mut stride = vec![elem_bits; n];
    let mut compact = vec![elem_bits; n];
    for d in (0..n - 1).rev() {
        stride[d] = stride[d + 1] * dims[d + 1].0;
        compact[d] = compact[d + 1] * dims[d + 1].1.iter().map(|r| r.len()).sum::<usize>();
    }
    // one contiguous run: the bytes read are the result
    if n == 1 && dims[0].1.len() == 1 {
        let r = &dims[0].1[0];
        return Ok(Region {
            read: byte(r.start * stride[0])?..byte(r.end * stride[0])?,
            shape: result_shape,
            gather: None,
        });
    }

    let k = n - 1; // runs are along the innermost dimension
    let h = k.checked_sub(1); // copies stride along the one outside it
    let outer = h.unwrap_or(0); // dimensions outside that are enumerated
    let mut combos = 1usize;
    for (_, rs) in &dims[..outer] {
        combos = combos.saturating_mul(rs.iter().map(|r| r.len()).sum());
    }
    let copies_needed = combos.saturating_mul(dims[k].1.len() * h.map_or(1, |h| dims[h].1.len()));
    if copies_needed > MAX_REGION_COPIES {
        return Err(InvalidSlice::TooFragmented {
            copies: copies_needed,
        });
    }

    // (source, destination) offsets in bits of every index combination of the enumerated dimensions
    let mut bases = vec![(0usize, 0usize)];
    for (d, (_, rs)) in dims[..outer].iter().enumerate() {
        let mut next = Vec::with_capacity(bases.len() * rs.iter().map(|r| r.len()).sum::<usize>());
        for &(src, dst) in &bases {
            let mut at = 0;
            for r in rs {
                for idx in r.clone() {
                    next.push((src + idx * stride[d], dst + at * compact[d]));
                    at += 1;
                }
            }
        }
        bases = next;
    }
    // the runs of the copies' height dimension, each with where it lands in the result
    let rows: Vec<(Range<usize>, usize)> = match h {
        Some(h) => {
            let mut at = 0;
            dims[h]
                .1
                .iter()
                .map(|r| {
                    let landed = (r.clone(), at);
                    at += r.len();
                    landed
                })
                .collect()
        }
        None => vec![(0..1, 0)],
    };
    let mut copies = Vec::with_capacity(copies_needed);
    for &(src_base, dst_base) in &bases {
        for (row_range, row_at) in &rows {
            let (src_row, dst_row) = match h {
                Some(h) => (row_range.start * stride[h], row_at * compact[h]),
                None => (0, 0),
            };
            let mut at = 0;
            for r in &dims[k].1 {
                copies.push(StridedCopy {
                    src: byte(src_base + src_row + r.start * stride[k])?,
                    dst: byte(dst_base + dst_row + at * compact[k])?,
                    width: byte(r.len() * stride[k])?,
                    height: row_range.len(),
                    src_pitch: byte(h.map_or(0, |h| stride[h]))?,
                    dst_pitch: byte(h.map_or(0, |h| compact[h]))?,
                });
                at += r.len();
            }
        }
    }

    let start = copies.iter().map(|c| c.src).min().unwrap_or(0);
    let end = copies
        .iter()
        .map(|c| c.src + (c.height - 1) * c.src_pitch + c.width)
        .max()
        .unwrap_or(0);
    for c in &mut copies {
        c.src -= start;
    }
    let len = byte(compact[0] * dims[0].1.iter().map(|r| r.len()).sum::<usize>())?;
    Ok(Region {
        read: start..end,
        shape: result_shape,
        gather: Some(Gather { copies, len }),
    })
}

impl<'data> Iterator for SliceIterator<'data> {
    type Item = &'data [u8];

    fn next(&mut self) -> Option<Self::Item> {
        // TODO We might want to move the logic from `new`
        // here actually to remove the need to get all the indices
        // upfront.
        let (start, stop) = self.indices.pop()?;
        Some(&self.view.data()[start..stop])
    }
}

#[cfg(test)]
#[allow(clippy::single_range_in_vec_init)] // one range per dimension
mod region_tests {
    use super::*;

    /// The region applied to a buffer holding byte `i` at offset `i`, as a device gather would.
    fn gathered(nbytes: usize, region: &Region) -> Vec<u8> {
        let source: Vec<u8> = (0..nbytes).map(|i| i as u8).collect();
        let read = &source[region.read.clone()];
        match &region.gather {
            None => read.to_vec(),
            Some(g) => {
                let mut out = vec![0u8; g.len];
                for c in &g.copies {
                    for row in 0..c.height {
                        let (s, d) = (c.src + row * c.src_pitch, c.dst + row * c.dst_pitch);
                        out[d..d + c.width].copy_from_slice(&read[s..s + c.width]);
                    }
                }
                out
            }
        }
    }

    /// The same region by brute force over element indices (1-byte elements).
    fn expected(shape: &[usize], ranges: &[Vec<Range<usize>>]) -> Vec<u8> {
        fn rec(
            d: usize,
            at: usize,
            strides: &[usize],
            ranges: &[Vec<Range<usize>>],
            out: &mut Vec<u8>,
        ) {
            if d == strides.len() {
                out.push(at as u8);
                return;
            }
            for r in &ranges[d] {
                for i in r.clone() {
                    rec(d + 1, at + i * strides[d], strides, ranges, out);
                }
            }
        }
        let mut strides = vec![1; shape.len()];
        for d in (0..shape.len().saturating_sub(1)).rev() {
            strides[d] = strides[d + 1] * shape[d + 1];
        }
        let mut out = Vec::new();
        rec(0, 0, &strides, ranges, &mut out);
        out
    }

    fn indexers(ranges: &[Vec<Range<usize>>]) -> Vec<Vec<TensorIndexer>> {
        ranges
            .iter()
            .map(|d| d.iter().map(|r| r.clone().into()).collect())
            .collect()
    }

    fn check(shape: &[usize], ranges: Vec<Vec<Range<usize>>>) -> Region {
        let region = slice_region(Dtype::U8, shape, &indexers(&ranges)).unwrap();
        assert_eq!(
            gathered(shape.iter().product(), &region),
            expected(shape, &ranges),
            "{shape:?} {ranges:?}"
        );
        let sizes: Vec<usize> = ranges
            .iter()
            .map(|d| d.iter().map(|r| r.len()).sum())
            .collect();
        assert_eq!(region.shape, sizes);
        region
    }

    #[test]
    fn whole_tensor_and_row_slices_need_no_gather() {
        assert_eq!(check(&[6, 5], vec![vec![0..6], vec![0..5]]).gather, None);
        let rows = check(&[6, 5], vec![vec![2..4], vec![0..5]]);
        assert_eq!((rows.read, rows.gather), (10..20, None));
        assert_eq!(
            check(&[4, 3, 2], vec![vec![1..3], vec![0..3], vec![0..2]]).gather,
            None
        );
        // dimensions left out are kept whole
        let r = slice_region(Dtype::U8, &[6, 5], &[vec![(2..4).into()]]).unwrap();
        assert_eq!((r.read, r.shape, r.gather), (10..20, vec![2, 5], None));
    }

    #[test]
    fn a_column_slice_is_one_strided_copy() {
        let r = check(&[6, 8], vec![vec![0..6], vec![2..5]]);
        assert_eq!(r.read, 2..(5 * 8 + 5));
        let g = r.gather.unwrap();
        assert_eq!(g.copies.len(), 1);
        assert_eq!(
            (g.copies[0].height, g.copies[0].width, g.copies[0].src_pitch),
            (6, 3, 8)
        );
    }

    #[test]
    fn full_middle_dims_merge_into_one_copy() {
        let g = check(&[4, 3, 8], vec![vec![0..4], vec![0..3], vec![1..3]])
            .gather
            .unwrap();
        assert_eq!((g.copies.len(), g.copies[0].height), (1, 12));
    }

    #[test]
    fn interleaved_and_mixed_regions() {
        check(&[8, 4], vec![vec![0..2, 4..6], vec![0..4]]);
        check(&[8, 6], vec![vec![1..3, 5..8], vec![0..2, 4..6]]);
        check(&[3, 4, 5], vec![vec![0..3], vec![1..3], vec![0..5]]);
        check(&[3, 4, 5], vec![vec![1..2], vec![0..4], vec![2..4]]);
        check(
            &[3, 4, 5],
            vec![vec![0..1, 2..3], vec![1..4], vec![0..2, 3..5]],
        );
        check(
            &[2, 3, 4, 5],
            vec![vec![0..2], vec![1..2], vec![0..4], vec![1..4]],
        );
        check(&[7], vec![vec![1..3, 4..6]]);
        // touching ranges merge: a whole dimension stays whole
        assert_eq!(
            check(&[6, 5], vec![vec![0..2, 2..6], vec![0..5]]).gather,
            None
        );
    }

    #[test]
    fn steps_and_selects() {
        // a stepped slice keeps every step-th element
        let step = TensorIndexer::Narrow(
            Bound::Included(1),
            Bound::Excluded(7),
            NonZeroUsize::new(2).unwrap(),
        );
        let r = slice_region(Dtype::U8, &[3, 8], &[vec![(0..3).into()], vec![step]]).unwrap();
        assert_eq!(r.shape, vec![3, 3]);
        assert_eq!(
            gathered(24, &r),
            expected(&[3, 8], &[vec![0..3], vec![1..2, 3..4, 5..6]])
        );
        // a lone select drops its dimension; inside a list it keeps one element and the dimension
        let r = slice_region(Dtype::U8, &[4, 5], &[vec![TensorIndexer::Select(2)]]).unwrap();
        assert_eq!((r.read, r.shape), (10..15, vec![5]));
        let r = slice_region(
            Dtype::U8,
            &[4, 5],
            &[vec![TensorIndexer::Select(0), TensorIndexer::Select(3)]],
        )
        .unwrap();
        assert_eq!(r.shape, vec![2, 5]);
        assert_eq!(
            gathered(20, &r),
            expected(&[4, 5], &[vec![0..1, 3..4], vec![0..5]])
        );
    }

    #[test]
    fn empty_regions_read_nothing() {
        let r = check(&[4, 4], vec![vec![1..1], vec![0..4]]);
        assert_eq!((r.read, r.shape), (0..0, vec![0, 4]));
    }

    #[test]
    fn invalid_regions_are_refused() {
        let u8 = Dtype::U8;
        assert_eq!(
            slice_region(u8, &[4], &[vec![], vec![]]),
            Err(InvalidSlice::TooManySlices)
        );
        assert!(matches!(
            slice_region(u8, &[4, 5], &[vec![], vec![(0..6).into()]]),
            Err(InvalidSlice::SliceOutOfRange { dim_index: 1, .. })
        ));
        assert!(matches!(
            slice_region(u8, &[4], &[vec![TensorIndexer::Select(4)]]),
            Err(InvalidSlice::SliceOutOfRange {
                dim_index: 0,
                asked: 4,
                ..
            })
        ));
        assert_eq!(
            slice_region(u8, &[8], &[vec![(3..5).into(), (0..2).into()]]),
            Err(InvalidSlice::UnorderedRanges { dim_index: 0 })
        );
        assert_eq!(
            slice_region(u8, &[8], &[vec![(0..4).into(), (2..6).into()]]),
            Err(InvalidSlice::UnorderedRanges { dim_index: 0 })
        );
        // 4-bit elements: two per byte
        assert!(slice_region(
            Dtype::F4,
            &[4, 8],
            &[vec![(0..4).into()], vec![(2..6).into()]]
        )
        .is_ok());
        assert_eq!(
            slice_region(
                Dtype::F4,
                &[4, 8],
                &[vec![(0..4).into()], vec![(1..5).into()]]
            ),
            Err(InvalidSlice::MisalignedSlice)
        );
        let every_other: Vec<TensorIndexer> = (0..600).map(|i| (2 * i..2 * i + 1).into()).collect();
        assert!(matches!(
            slice_region(u8, &[1200, 1200], &[every_other.clone(), every_other]),
            Err(InvalidSlice::TooFragmented { .. })
        ));
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tensor::{Dtype, TensorView};

    #[test]
    fn test_helpers() {
        let data: Vec<u8> = vec![0.0f32, 1.0, 2.0, 3.0, 4.0, 5.0]
            .into_iter()
            .flat_map(|f| f.to_le_bytes())
            .collect();

        let attn_0 = TensorView::new(Dtype::F32, vec![1, 2, 3], &data).unwrap();

        let iterator = SliceIterator::new(
            &attn_0,
            &[TensorIndexer::Narrow(
                Bound::Unbounded,
                Bound::Unbounded,
                NonZeroUsize::MIN,
            )],
        )
        .unwrap();
        assert_eq!(iterator.remaining_byte_len(), 24);
        assert_eq!(iterator.newshape(), vec![1, 2, 3]);

        let iterator = SliceIterator::new(
            &attn_0,
            &[
                TensorIndexer::Narrow(Bound::Unbounded, Bound::Unbounded, NonZeroUsize::MIN),
                TensorIndexer::Narrow(Bound::Included(0), Bound::Excluded(1), NonZeroUsize::MIN),
            ],
        )
        .unwrap();
        assert_eq!(iterator.remaining_byte_len(), 12);
        assert_eq!(iterator.newshape(), vec![1, 1, 3]);
    }

    #[test]
    fn test_fp4_simple() {
        let data: Vec<u8> = vec![0u8, 1u8];

        let attn_0 = TensorView::new(Dtype::F4, vec![1, 2, 2], &data).unwrap();

        let iterator = SliceIterator::new(
            &attn_0,
            &[TensorIndexer::Narrow(
                Bound::Unbounded,
                Bound::Unbounded,
                NonZeroUsize::MIN,
            )],
        )
        .unwrap();
        assert_eq!(iterator.remaining_byte_len(), 2);
        assert_eq!(iterator.newshape(), vec![1, 2, 2]);

        let iterator = SliceIterator::new(
            &attn_0,
            &[
                TensorIndexer::Narrow(Bound::Unbounded, Bound::Unbounded, NonZeroUsize::MIN),
                TensorIndexer::Narrow(Bound::Included(0), Bound::Excluded(1), NonZeroUsize::MIN),
            ],
        )
        .unwrap();
        assert_eq!(iterator.remaining_byte_len(), 1);
        assert_eq!(iterator.newshape(), vec![1, 1, 2]);
    }

    #[test]
    fn test_fp4_misaligned() {
        let data: Vec<u8> = vec![0u8];

        let attn_0 = TensorView::new(Dtype::F4, vec![1, 2], &data).unwrap();

        let iterator = SliceIterator::new(
            &attn_0,
            &[TensorIndexer::Narrow(
                Bound::Unbounded,
                Bound::Unbounded,
                NonZeroUsize::MIN,
            )],
        )
        .unwrap();
        assert_eq!(iterator.remaining_byte_len(), 1);
        assert_eq!(iterator.newshape(), vec![1, 2]);

        let iterator = SliceIterator::new(
            &attn_0,
            &[
                TensorIndexer::Narrow(Bound::Unbounded, Bound::Unbounded, NonZeroUsize::MIN),
                TensorIndexer::Narrow(Bound::Included(0), Bound::Excluded(1), NonZeroUsize::MIN),
            ],
        );

        assert_eq!(iterator, Err(InvalidSlice::MisalignedSlice));
    }

    #[test]
    fn test_dummy() {
        let data: Vec<u8> = vec![0.0f32, 1.0, 2.0, 3.0, 4.0, 5.0]
            .into_iter()
            .flat_map(|f| f.to_le_bytes())
            .collect();

        let attn_0 = TensorView::new(Dtype::F32, vec![1, 2, 3], &data).unwrap();

        let mut iterator = SliceIterator::new(
            &attn_0,
            &[TensorIndexer::Narrow(
                Bound::Unbounded,
                Bound::Unbounded,
                NonZeroUsize::MIN,
            )],
        )
        .unwrap();
        assert_eq!(iterator.next(), Some(&data[0..24]));
        assert_eq!(iterator.next(), None);

        let mut iterator = SliceIterator::new(
            &attn_0,
            &[
                TensorIndexer::Narrow(Bound::Unbounded, Bound::Unbounded, NonZeroUsize::MIN),
                TensorIndexer::Narrow(Bound::Unbounded, Bound::Unbounded, NonZeroUsize::MIN),
            ],
        )
        .unwrap();
        assert_eq!(iterator.next(), Some(&data[0..24]));
        assert_eq!(iterator.next(), None);

        let mut iterator = SliceIterator::new(
            &attn_0,
            &[
                TensorIndexer::Narrow(Bound::Unbounded, Bound::Unbounded, NonZeroUsize::MIN),
                TensorIndexer::Narrow(Bound::Unbounded, Bound::Unbounded, NonZeroUsize::MIN),
            ],
        )
        .unwrap();
        assert_eq!(iterator.next(), Some(&data[0..24]));
        assert_eq!(iterator.next(), None);

        let mut iterator = SliceIterator::new(
            &attn_0,
            &[
                TensorIndexer::Narrow(Bound::Unbounded, Bound::Unbounded, NonZeroUsize::MIN),
                TensorIndexer::Narrow(Bound::Unbounded, Bound::Unbounded, NonZeroUsize::MIN),
                TensorIndexer::Narrow(Bound::Unbounded, Bound::Unbounded, NonZeroUsize::MIN),
            ],
        )
        .unwrap();
        assert_eq!(iterator.next(), Some(&data[0..24]));
        assert_eq!(iterator.next(), None);

        assert!(SliceIterator::new(
            &attn_0,
            &[
                TensorIndexer::Narrow(Bound::Unbounded, Bound::Unbounded, NonZeroUsize::MIN),
                TensorIndexer::Narrow(Bound::Unbounded, Bound::Unbounded, NonZeroUsize::MIN),
                TensorIndexer::Narrow(Bound::Unbounded, Bound::Unbounded, NonZeroUsize::MIN),
                TensorIndexer::Narrow(Bound::Unbounded, Bound::Unbounded, NonZeroUsize::MIN),
            ],
        )
        .is_err(),);
    }

    #[test]
    fn test_slice_variety() {
        let data: Vec<u8> = vec![0.0f32, 1.0, 2.0, 3.0, 4.0, 5.0]
            .into_iter()
            .flat_map(|f| f.to_le_bytes())
            .collect();

        let attn_0 = TensorView::new(Dtype::F32, vec![1, 2, 3], &data).unwrap();

        let mut iterator = SliceIterator::new(
            &attn_0,
            &[TensorIndexer::Narrow(
                Bound::Included(0),
                Bound::Excluded(1),
                NonZeroUsize::MIN,
            )],
        )
        .unwrap();
        assert_eq!(iterator.next(), Some(&data[0..24]));
        assert_eq!(iterator.next(), None);

        let mut iterator = SliceIterator::new(
            &attn_0,
            &[
                TensorIndexer::Narrow(Bound::Unbounded, Bound::Unbounded, NonZeroUsize::MIN),
                TensorIndexer::Narrow(Bound::Included(0), Bound::Excluded(1), NonZeroUsize::MIN),
            ],
        )
        .unwrap();
        assert_eq!(iterator.next(), Some(&data[0..12]));
        assert_eq!(iterator.next(), None);

        let mut iterator = SliceIterator::new(
            &attn_0,
            &[
                TensorIndexer::Narrow(Bound::Unbounded, Bound::Unbounded, NonZeroUsize::MIN),
                TensorIndexer::Narrow(Bound::Unbounded, Bound::Unbounded, NonZeroUsize::MIN),
                TensorIndexer::Narrow(Bound::Included(0), Bound::Excluded(1), NonZeroUsize::MIN),
            ],
        )
        .unwrap();
        assert_eq!(iterator.next(), Some(&data[0..4]));
        assert_eq!(iterator.next(), Some(&data[12..16]));
        assert_eq!(iterator.next(), None);

        let mut iterator = SliceIterator::new(
            &attn_0,
            &[
                TensorIndexer::Narrow(Bound::Unbounded, Bound::Unbounded, NonZeroUsize::MIN),
                TensorIndexer::Narrow(Bound::Included(1), Bound::Excluded(2), NonZeroUsize::MIN),
                TensorIndexer::Narrow(Bound::Included(0), Bound::Excluded(1), NonZeroUsize::MIN),
            ],
        )
        .unwrap();
        assert_eq!(iterator.next(), Some(&data[12..16]));
        assert_eq!(iterator.next(), None);
    }

    #[test]
    fn test_slice_variety2() {
        let data: Vec<u8> = vec![0.0f32, 1.0, 2.0, 3.0, 4.0, 5.0]
            .into_iter()
            .flat_map(|f| f.to_le_bytes())
            .collect();

        let attn_0 = TensorView::new(Dtype::F32, vec![2, 3], &data).unwrap();

        let mut iterator = SliceIterator::new(
            &attn_0,
            &[
                TensorIndexer::Narrow(Bound::Unbounded, Bound::Unbounded, NonZeroUsize::MIN),
                TensorIndexer::Narrow(Bound::Included(1), Bound::Excluded(3), NonZeroUsize::MIN),
            ],
        )
        .unwrap();
        assert_eq!(iterator.next(), Some(&data[4..12]));
        assert_eq!(iterator.next(), Some(&data[16..24]));
        assert_eq!(iterator.next(), None);
    }

    #[test]
    fn test_slice_select() {
        let data: Vec<u8> = vec![0.0f32, 1.0, 2.0, 3.0, 4.0, 5.0]
            .into_iter()
            .flat_map(|f| f.to_le_bytes())
            .collect();

        let attn_0 = TensorView::new(Dtype::F32, vec![2, 3], &data).unwrap();

        let mut iterator = SliceIterator::new(
            &attn_0,
            &[
                TensorIndexer::Select(1),
                TensorIndexer::Narrow(Bound::Included(1), Bound::Excluded(3), NonZeroUsize::MIN),
            ],
        )
        .unwrap();
        assert_eq!(iterator.next(), Some(&data[16..24]));
        assert_eq!(iterator.next(), None);

        let mut iterator = SliceIterator::new(
            &attn_0,
            &[
                TensorIndexer::Select(0),
                TensorIndexer::Narrow(Bound::Included(1), Bound::Excluded(3), NonZeroUsize::MIN),
            ],
        )
        .unwrap();
        assert_eq!(iterator.next(), Some(&data[4..12]));
        assert_eq!(iterator.next(), None);

        let mut iterator = SliceIterator::new(
            &attn_0,
            &[
                TensorIndexer::Narrow(Bound::Included(1), Bound::Excluded(2), NonZeroUsize::MIN),
                TensorIndexer::Select(0),
            ],
        )
        .unwrap();
        assert_eq!(iterator.next(), Some(&data[12..16]));
        assert_eq!(iterator.next(), None);
    }

    #[test]
    fn test_invalid_range() {
        let data: Vec<u8> = vec![0.0f32, 1.0, 2.0, 3.0, 4.0, 5.0]
            .into_iter()
            .flat_map(|f| f.to_le_bytes())
            .collect();

        let attn_0 = TensorView::new(Dtype::F32, vec![2, 3], &data).unwrap();

        assert_eq!(
            SliceIterator::new(
                &attn_0,
                &[
                    TensorIndexer::Select(1),
                    TensorIndexer::Narrow(
                        Bound::Included(1),
                        Bound::Excluded(4),
                        NonZeroUsize::MIN
                    ),
                ],
            ),
            Err(InvalidSlice::SliceOutOfRange {
                asked: 3,
                dim_index: 1,
                dim_size: 3,
            })
        );
        assert_eq!(
            SliceIterator::new(
                &attn_0,
                &[
                    TensorIndexer::Select(1),
                    TensorIndexer::Narrow(
                        Bound::Included(3),
                        Bound::Excluded(2),
                        NonZeroUsize::MIN
                    ),
                ],
            ),
            Err(InvalidSlice::SliceOutOfRange {
                asked: 3,
                dim_index: 1,
                dim_size: 3,
            })
        );
        assert_eq!(
            SliceIterator::new(
                &attn_0,
                &[
                    TensorIndexer::Select(1),
                    TensorIndexer::Select(1),
                    TensorIndexer::Select(1),
                ],
            ),
            Err(InvalidSlice::TooManySlices)
        );
    }
}
