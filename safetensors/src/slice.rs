//! Module handling lazy loading via iterating on slices on the original buffer.
use crate::lib::Vec;
use crate::tensor::{Dtype, TensorView};
use core::fmt::Display;
use core::num::NonZeroUsize;
use core::ops::{
    Bound, Range, RangeBounds, RangeFrom, RangeFull, RangeInclusive, RangeTo, RangeToInclusive,
};

/// Error representing invalid slicing attempt
#[derive(Debug)]
#[cfg_attr(test, derive(Eq, PartialEq))]
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
    /// The indexers given to [`slice_layout`] for a dimension overlap or are out of order
    UnorderedRanges {
        /// The rank of the dimension
        dim_index: usize,
    },
    /// [`slice_layout`] would need more than [`MAX_SLICE_COPIES`] strided copies
    TooFragmented {
        /// The number of copies needed
        copies: usize,
    },
    /// [`slice_layout`] was given a [`TensorIndexer::Narrow`] with a step other than 1
    SteppedSlice {
        /// The rank of the dimension
        dim_index: usize,
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
                write!(
                    f,
                    "ranges of tensor dimension #{dim_index} overlap or are out of order"
                )
            }
            InvalidSlice::TooFragmented { copies } => {
                write!(f, "the slice would take {copies} strided copies to gather, more than {MAX_SLICE_COPIES}")
            }
            InvalidSlice::SteppedSlice { dim_index } => {
                write!(f, "the slice of tensor dimension #{dim_index} has a step, which slice layouts do not support")
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
            let is_select = matches!(slice, TensorIndexer::Select(_));
            // A narrow may start at `dim` and select nothing, e.g. `[0:0]` on an empty dim.
            let start_oob = if is_select { start >= dim } else { start > dim };
            if start_oob || stop > dim {
                let asked = if start_oob {
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
            // Reversed bounds select nothing, as in Python.
            let stop = stop.max(start);
            if !is_select {
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
    if newshape.contains(&0) {
        indices.clear();
    } else if indices.is_empty() {
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

/// `height` rows of `width` bytes, `src_pitch` bytes apart in [`SliceLayout::read`] and
/// `dst_pitch` bytes apart in the result.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct StridedCopy {
    /// Offset of the first row in [`SliceLayout::read`]
    pub src: usize,
    /// Offset of the first row in the result
    pub dst: usize,
    /// Bytes per row
    pub width: usize,
    /// Number of rows
    pub height: usize,
    /// Bytes between the starts of two rows in [`SliceLayout::read`]
    pub src_pitch: usize,
    /// Bytes between the starts of two rows in the result
    pub dst_pitch: usize,
}

/// The [`StridedCopy`]s that fill a `len`-byte result from [`SliceLayout::read`], each
/// result byte exactly once.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Gather {
    /// The copies
    pub copies: Vec<StridedCopy>,
    /// The result's size in bytes
    pub len: usize,
}

/// The bytes of a tensor that a slice reads, and how to arrange them; see [`slice_layout`].
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct SliceLayout {
    /// The bytes to read, relative to the tensor's first byte
    pub read: Range<usize>,
    /// The result's shape
    pub shape: Vec<usize>,
    /// `None` when the bytes read are the result
    pub gather: Option<Gather>,
}

/// The most [`StridedCopy`]s a [`Gather`] holds; more is [`InvalidSlice::TooFragmented`].
pub const MAX_SLICE_COPIES: usize = 16_384;

/// Computes the [`SliceLayout`] of a slice of a row-major tensor.
///
/// `slices` holds a list of indexers per leading dimension; dimensions left out are kept
/// whole. The indexers of a list must be disjoint and in increasing order, and their
/// elements are kept in that order. A list holding a single [`TensorIndexer::Select`]
/// drops its dimension from the result shape. Bounds are checked as in
/// [`slice_byte_ranges`] and reversed bounds select nothing. A [`TensorIndexer::Narrow`]
/// with a step other than 1 is refused.
///
/// Dimensions kept whole merge into their outer neighbour, so the copies stride along the
/// second innermost remaining dimension and are repeated for every index of the ones
/// outside it.
pub fn slice_layout(
    dtype: Dtype,
    shape: &[usize],
    slices: &[Vec<TensorIndexer>],
) -> Result<SliceLayout, InvalidSlice> {
    if slices.len() > shape.len() {
        return Err(InvalidSlice::TooManySlices);
    }
    // (size, sorted disjoint non-touching kept ranges) per dimension
    let mut dims: Vec<(usize, Vec<Range<usize>>)> = Vec::with_capacity(shape.len());
    let mut result_shape = Vec::with_capacity(shape.len());
    for (i, &dim) in shape.iter().enumerate() {
        let Some(indexers) = slices.get(i) else {
            dims.push((dim, core::iter::once(0..dim).collect()));
            result_shape.push(dim);
            continue;
        };
        let mut kept: Vec<Range<usize>> = Vec::with_capacity(indexers.len());
        for indexer in indexers {
            let (start, stop) = match indexer {
                TensorIndexer::Select(s) => (*s, s.saturating_add(1)),
                TensorIndexer::Narrow(left, right, step) => {
                    if step.get() != 1 {
                        return Err(InvalidSlice::SteppedSlice { dim_index: i });
                    }
                    narrow_bounds(left, right, dim)
                }
            };
            if start >= dim || stop > dim {
                return Err(InvalidSlice::SliceOutOfRange {
                    dim_index: i,
                    asked: if start >= dim { start } else { stop - 1 },
                    dim_size: dim,
                });
            }
            if start >= stop {
                continue;
            }
            match kept.last_mut() {
                Some(last) if last.end > start => {
                    return Err(InvalidSlice::UnorderedRanges { dim_index: i });
                }
                Some(last) if last.end == start => last.end = stop,
                _ => kept.push(start..stop),
            }
        }
        if !matches!(indexers.as_slice(), [TensorIndexer::Select(_)]) {
            result_shape.push(kept.iter().map(Range::len).sum());
        }
        dims.push((dim, kept));
    }
    if dims.iter().any(|(_, rs)| rs.iter().all(Range::is_empty)) {
        return Ok(SliceLayout {
            read: 0..0,
            shape: result_shape,
            gather: None,
        });
    }

    for i in (1..dims.len()).rev() {
        if matches!(dims[i].1.as_slice(), [r] if r.start == 0 && r.end == dims[i].0) {
            let (inner, _) = dims.remove(i);
            let (size, rs) = &mut dims[i - 1];
            *size *= inner;
            for r in rs.iter_mut() {
                *r = r.start * inner..r.end * inner;
            }
        }
    }
    let byte = |bits: usize| {
        if bits % 8 == 0 {
            Ok(bits / 8)
        } else {
            Err(InvalidSlice::MisalignedSlice)
        }
    };
    let elem_bits = dtype.bitsize();
    let count = |rs: &[Range<usize>]| rs.iter().map(Range::len).sum::<usize>();
    match dims.as_slice() {
        [] => {
            return Ok(SliceLayout {
                read: 0..byte(elem_bits)?,
                shape: result_shape,
                gather: None,
            })
        }
        [(_, rs)] if rs.len() == 1 => {
            return Ok(SliceLayout {
                read: byte(rs[0].start * elem_bits)?..byte(rs[0].end * elem_bits)?,
                shape: result_shape,
                gather: None,
            })
        }
        _ => {}
    }

    // strides in bits, in the source and in the compact result
    let n = dims.len();
    let mut stride = vec![elem_bits; n];
    let mut compact = vec![elem_bits; n];
    for d in (0..n - 1).rev() {
        stride[d] = stride[d + 1] * dims[d + 1].0;
        compact[d] = compact[d + 1] * count(&dims[d + 1].1);
    }
    // copies are runs of the innermost dimension `k`, strided along `h` when there is one
    let k = n - 1;
    let h = k.checked_sub(1);
    let enumerated = &dims[..h.unwrap_or(0)];
    let combos = enumerated
        .iter()
        .fold(1usize, |acc, (_, rs)| acc.saturating_mul(count(rs)));
    let copies_needed = combos
        .saturating_mul(dims[k].1.len())
        .saturating_mul(h.map_or(1, |h| dims[h].1.len()));
    if copies_needed > MAX_SLICE_COPIES {
        return Err(InvalidSlice::TooFragmented {
            copies: copies_needed,
        });
    }

    // (source, result) offsets in bits of every index combination of the enumerated dimensions
    let mut bases = vec![(0usize, 0usize)];
    for (d, (_, rs)) in enumerated.iter().enumerate() {
        let mut next = Vec::with_capacity(bases.len() * count(rs));
        for &(src, dst) in &bases {
            for (at, idx) in rs.iter().cloned().flatten().enumerate() {
                next.push((src + idx * stride[d], dst + at * compact[d]));
            }
        }
        bases = next;
    }
    // (rows, index of the first one in the result)
    let rows: Vec<(Range<usize>, usize)> = match h {
        Some(h) => dims[h]
            .1
            .iter()
            .scan(0, |at, r| {
                *at += r.len();
                Some((r.clone(), *at - r.len()))
            })
            .collect(),
        None => vec![(0..1, 0)],
    };
    let (row_stride, row_compact) = h.map_or((0, 0), |h| (stride[h], compact[h]));
    let mut copies = Vec::with_capacity(copies_needed);
    for &(src_base, dst_base) in &bases {
        for (row, row_at) in &rows {
            let pitch = |bits| if row.len() > 1 { byte(bits) } else { Ok(0) };
            let mut at = 0;
            for w in &dims[k].1 {
                copies.push(StridedCopy {
                    src: byte(src_base + row.start * row_stride + w.start * elem_bits)?,
                    dst: byte(dst_base + row_at * row_compact + at * elem_bits)?,
                    width: byte(w.len() * elem_bits)?,
                    height: row.len(),
                    src_pitch: pitch(row_stride)?,
                    dst_pitch: pitch(row_compact)?,
                });
                at += w.len();
            }
        }
    }

    let start = copies.iter().map(|c| c.src).min().unwrap_or(0);
    let end = copies
        .iter()
        .map(|c| c.src + (c.height - 1) * c.src_pitch + c.width)
        .max()
        .unwrap_or(0);
    if let [c] = copies.as_slice() {
        if c.height == 1 {
            return Ok(SliceLayout {
                read: start..end,
                shape: result_shape,
                gather: None,
            });
        }
    }
    for c in &mut copies {
        c.src -= start;
    }
    let len = byte(compact[0] * count(&dims[0].1))?;
    Ok(SliceLayout {
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
mod layout_tests {
    use super::*;
    use proptest::prelude::*;

    /// Applies `layout` to a source whose byte `i` is `i`.
    fn gathered(layout: &SliceLayout) -> Vec<usize> {
        let read: Vec<usize> = layout.read.clone().collect();
        match &layout.gather {
            None => read,
            Some(g) => {
                let mut out = vec![usize::MAX; g.len];
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

    /// The source bit offsets of the slice's elements, in result order, by enumerating
    /// element indices.
    fn expected_bits(dtype: Dtype, shape: &[usize], slices: &[Vec<TensorIndexer>]) -> Vec<usize> {
        let kept: Vec<Vec<usize>> = shape
            .iter()
            .enumerate()
            .map(|(i, &dim)| match slices.get(i) {
                None => (0..dim).collect(),
                Some(list) => list
                    .iter()
                    .flat_map(|ix| {
                        let (start, stop) = match ix {
                            TensorIndexer::Select(s) => (*s, *s + 1),
                            TensorIndexer::Narrow(l, r, _) => narrow_bounds(l, r, dim),
                        };
                        start..stop.max(start)
                    })
                    .collect(),
            })
            .collect();
        let bits = dtype.bitsize();
        let mut elements = vec![0usize];
        for (d, idx) in kept.iter().enumerate() {
            let stride: usize = shape[d + 1..].iter().product();
            elements = elements
                .iter()
                .flat_map(|&e| idx.iter().map(move |&i| e + i * stride))
                .collect();
        }
        elements
            .into_iter()
            .flat_map(|e| e * bits..(e + 1) * bits)
            .collect()
    }

    fn bits_of(bytes: &[usize]) -> Vec<usize> {
        bytes.iter().flat_map(|&b| b * 8..b * 8 + 8).collect()
    }

    fn check(shape: &[usize], ranges: Vec<Vec<Range<usize>>>) -> SliceLayout {
        let slices: Vec<Vec<TensorIndexer>> = ranges
            .iter()
            .map(|d| d.iter().map(|r| r.clone().into()).collect())
            .collect();
        let layout = slice_layout(Dtype::U8, shape, &slices).unwrap();
        assert_eq!(
            bits_of(&gathered(&layout)),
            expected_bits(Dtype::U8, shape, &slices),
            "{shape:?} {ranges:?}"
        );
        let sizes: Vec<usize> = ranges
            .iter()
            .map(|d| d.iter().map(|r| r.len()).sum())
            .collect();
        assert_eq!(layout.shape, sizes);
        layout
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
        let part_of_a_row = check(&[6, 5], vec![vec![2..3], vec![1..4]]);
        assert_eq!((part_of_a_row.read, part_of_a_row.gather), (11..14, None));
        let r = slice_layout(Dtype::U8, &[6, 5], &[vec![(2..4).into()]]).unwrap();
        assert_eq!((r.read, r.shape, r.gather), (10..20, vec![2, 5], None));
        let r = slice_layout(Dtype::F32, &[], &[]).unwrap();
        assert_eq!((r.read, r.shape, r.gather), (0..4, vec![], None));
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
    fn interleaved_and_mixed_slices() {
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
        // touching ranges merge, so a whole dimension stays whole
        assert_eq!(
            check(&[6, 5], vec![vec![0..2, 2..6], vec![0..5]]).gather,
            None
        );
    }

    #[test]
    fn selects() {
        let r = slice_layout(Dtype::U8, &[4, 5], &[vec![TensorIndexer::Select(2)]]).unwrap();
        assert_eq!((r.read, r.shape), (10..15, vec![5]));
        let slices = [vec![TensorIndexer::Select(0), TensorIndexer::Select(3)]];
        let r = slice_layout(Dtype::U8, &[4, 5], &slices).unwrap();
        assert_eq!(r.shape, vec![2, 5]);
        assert_eq!(
            bits_of(&gathered(&r)),
            expected_bits(Dtype::U8, &[4, 5], &slices)
        );
    }

    #[test]
    fn empty_slices_read_nothing() {
        let r = check(&[4, 4], vec![vec![1..1], vec![0..4]]);
        assert_eq!((r.read, r.shape), (0..0, vec![0, 4]));
        let r = slice_layout(Dtype::U8, &[5, 0], &[vec![(0..1).into(), (3..4).into()]]).unwrap();
        assert_eq!((r.read, r.shape, r.gather), (0..0, vec![2, 0], None));
        let reversed =
            TensorIndexer::Narrow(Bound::Included(3), Bound::Excluded(1), NonZeroUsize::MIN);
        let r = slice_layout(Dtype::U8, &[8, 2], &[vec![reversed.clone()]]).unwrap();
        assert_eq!((r.read, r.shape, r.gather), (0..0, vec![0, 2], None));
        let r = slice_layout(Dtype::U8, &[8, 2], &[vec![(0..2).into(), reversed]]).unwrap();
        assert_eq!((r.read, r.shape), (0..4, vec![2, 2]));
    }

    #[test]
    fn invalid_slices_are_refused() {
        let u8 = Dtype::U8;
        assert_eq!(
            slice_layout(u8, &[4], &[vec![], vec![]]),
            Err(InvalidSlice::TooManySlices)
        );
        assert!(matches!(
            slice_layout(u8, &[4, 5], &[vec![], vec![(0..6).into()]]),
            Err(InvalidSlice::SliceOutOfRange { dim_index: 1, .. })
        ));
        assert!(matches!(
            slice_layout(u8, &[4], &[vec![TensorIndexer::Select(4)]]),
            Err(InvalidSlice::SliceOutOfRange {
                dim_index: 0,
                asked: 4,
                ..
            })
        ));
        // as in `slice_byte_ranges`, a start at the end is out of range even when empty
        assert!(matches!(
            slice_layout(u8, &[4, 5], &[vec![(4..4).into()]]),
            Err(InvalidSlice::SliceOutOfRange {
                dim_index: 0,
                asked: 4,
                ..
            })
        ));
        assert_eq!(
            slice_layout(u8, &[8], &[vec![(3..5).into(), (0..2).into()]]),
            Err(InvalidSlice::UnorderedRanges { dim_index: 0 })
        );
        assert_eq!(
            slice_layout(u8, &[8], &[vec![(0..4).into(), (2..6).into()]]),
            Err(InvalidSlice::UnorderedRanges { dim_index: 0 })
        );
        let by2 = TensorIndexer::Narrow(
            Bound::Unbounded,
            Bound::Unbounded,
            NonZeroUsize::new(2).unwrap(),
        );
        assert_eq!(
            slice_layout(u8, &[4, 8], &[vec![(0..4).into()], vec![by2]]),
            Err(InvalidSlice::SteppedSlice { dim_index: 1 })
        );
        assert!(slice_layout(
            Dtype::F4,
            &[4, 8],
            &[vec![(0..4).into()], vec![(2..6).into()]]
        )
        .is_ok());
        assert_eq!(
            slice_layout(
                Dtype::F4,
                &[4, 8],
                &[vec![(0..4).into()], vec![(1..5).into()]]
            ),
            Err(InvalidSlice::MisalignedSlice)
        );
        let every_other: Vec<TensorIndexer> = (0..130).map(|i| (2 * i..2 * i + 1).into()).collect();
        assert_eq!(
            slice_layout(u8, &[260, 260], &[every_other.clone(), every_other]),
            Err(InvalidSlice::TooFragmented { copies: 16_900 })
        );
    }

    fn arbitrary_slice() -> impl Strategy<Value = (Dtype, Vec<usize>, Vec<Vec<TensorIndexer>>)> {
        let dtype = prop_oneof![
            Just(Dtype::F4),
            Just(Dtype::F6_E2M3),
            Just(Dtype::U8),
            Just(Dtype::BF16),
            Just(Dtype::F32),
            Just(Dtype::F64),
        ];
        (dtype, prop::collection::vec(0..=6usize, 0..=4))
            .prop_flat_map(|(dtype, shape)| {
                let cuts: Vec<_> = shape
                    .iter()
                    .map(|&dim| prop::collection::vec((0..=dim, any::<bool>()), 0..=6))
                    .collect();
                (Just(dtype), Just(shape.clone()), 0..=shape.len(), cuts)
            })
            .prop_map(|(dtype, shape, sliced, cuts)| {
                // sorted cut points, paired into ranges; a one-element range may be a select
                let slices = cuts
                    .into_iter()
                    .zip(&shape)
                    .take(sliced)
                    .map(|(mut cuts, &dim)| {
                        cuts.sort();
                        cuts.chunks_exact(2)
                            .filter(|pair| pair[0].0 < dim)
                            .map(|pair| match (pair[0], pair[1].0) {
                                ((start, true), stop) if stop == start + 1 => {
                                    TensorIndexer::Select(start)
                                }
                                ((start, _), stop) => (start..stop).into(),
                            })
                            .collect()
                    })
                    .collect();
                (dtype, shape, slices)
            })
    }

    proptest! {
        #![proptest_config(ProptestConfig::with_cases(2048))]

        #[test]
        fn layout_matches_a_naive_gather((dtype, shape, slices) in arbitrary_slice()) {
            let bits = dtype.bitsize();
            let layout = match slice_layout(dtype, &shape, &slices) {
                Ok(layout) => layout,
                Err(InvalidSlice::MisalignedSlice) if bits % 8 != 0 => return Ok(()),
                Err(e) => panic!("{e:?}"),
            };
            let elements: usize = layout.shape.iter().product();
            let len = match &layout.gather {
                None => layout.read.len(),
                Some(g) => {
                    let src_end = |c: &StridedCopy| c.src + (c.height - 1) * c.src_pitch + c.width;
                    let mut rows = Vec::new();
                    for c in &g.copies {
                        prop_assert!(c.width > 0 && c.height > 0);
                        prop_assert!(src_end(c) <= layout.read.len());
                        rows.extend((0..c.height).map(|r| {
                            let d = c.dst + r * c.dst_pitch;
                            d..d + c.width
                        }));
                    }
                    rows.sort_by_key(|r| r.start);
                    prop_assert!(rows.windows(2).all(|w| w[0].end <= w[1].start));
                    prop_assert!(rows.last().is_none_or(|r| r.end <= g.len));
                    prop_assert_eq!(rows.iter().map(Range::len).sum::<usize>(), g.len);
                    prop_assert!(g.copies.len() <= MAX_SLICE_COPIES);
                    // `read` is tight
                    prop_assert_eq!(g.copies.iter().map(|c| c.src).min(), Some(0));
                    prop_assert_eq!(g.copies.iter().map(src_end).max(), Some(layout.read.len()));
                    g.len
                }
            };
            prop_assert_eq!(len * 8, elements * bits);
            let result = gathered(&layout);
            prop_assert_eq!(bits_of(&result), expected_bits(dtype, &shape, &slices));

            if slices.iter().all(|list| list.len() == 1) {
                let flat: Vec<TensorIndexer> = slices.iter().map(|l| l[0].clone()).collect();
                let data = vec![0u8; shape.iter().product::<usize>() * bits / 8];
                let view = TensorView::new(dtype, shape.clone(), &data);
                let it = view.as_ref().ok().and_then(|v| v.sliced_data(&flat).ok());
                // sub-byte dtypes may be refused as misaligned at different points
                prop_assert!(it.is_some() || bits % 8 != 0);
                if let Some(it) = it {
                    prop_assert_eq!(it.newshape(), layout.shape.clone());
                    let chunks: Vec<&[u8]> = it.collect();
                    prop_assert_eq!(layout.gather.is_none(), chunks.len() <= 1);
                    let offsets: Vec<usize> = chunks
                        .iter()
                        .flat_map(|c| {
                            let at = c.as_ptr() as usize - data.as_ptr() as usize;
                            at..at + c.len()
                        })
                        .collect();
                    prop_assert_eq!(offsets, result);
                }
            }
        }
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
            &[TensorIndexer::Narrow(Bound::Unbounded, Bound::Unbounded, NonZeroUsize::MIN)],
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
            &[TensorIndexer::Narrow(Bound::Unbounded, Bound::Unbounded, NonZeroUsize::MIN)],
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
            &[TensorIndexer::Narrow(Bound::Unbounded, Bound::Unbounded, NonZeroUsize::MIN)],
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
            &[TensorIndexer::Narrow(Bound::Unbounded, Bound::Unbounded, NonZeroUsize::MIN)],
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
    fn test_slice_empty() {
        for (start, stop, step) in [(3, 1, 1), (3, 1, 2), (3, 3, 1), (3, 3, 2)] {
            let slice = TensorIndexer::Narrow(
                Bound::Included(start),
                Bound::Excluded(stop),
                NonZeroUsize::new(step).unwrap(),
            );
            let (ranges, shape) = slice_byte_ranges(Dtype::F32, &[8, 2], &[slice]).unwrap();
            assert!(ranges.is_empty());
            assert_eq!(shape, vec![0, 2]);
        }
        for (dim, start) in [(8, 8), (0, 0)] {
            let slice =
                TensorIndexer::Narrow(Bound::Included(start), Bound::Unbounded, NonZeroUsize::MIN);
            let (ranges, shape) = slice_byte_ranges(Dtype::F32, &[dim, 2], &[slice]).unwrap();
            assert!(ranges.is_empty());
            assert_eq!(shape, vec![0, 2]);
        }
        assert!(slice_byte_ranges(Dtype::F32, &[0, 2], &[TensorIndexer::Select(0)]).is_err());
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
                    TensorIndexer::Narrow(Bound::Included(1), Bound::Excluded(4), NonZeroUsize::MIN),
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
                    TensorIndexer::Narrow(Bound::Included(4), Bound::Excluded(2), NonZeroUsize::MIN),
                ],
            ),
            Err(InvalidSlice::SliceOutOfRange {
                asked: 4,
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
