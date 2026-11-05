#![no_main]
//! Fuzz the lazy-slicing path over a parsed safetensors buffer.
//!
//! `TensorView::sliced_data` turns caller-supplied indexers into byte ranges
//! into the tensor's data section, and the iterator hands those ranges out as
//! `&[u8]`. The index math runs against a shape the file itself controls, so
//! this target drives both halves from one input.
//!
//! The last [`SPEC_LEN`] bytes are the slice spec and everything before them
//! is the file. The split is a fixed-size tail on purpose: the parser checks
//! the buffer length against the header's own offsets, so the file bytes have
//! to stay contiguous and correctly sized for the input to get past
//! `deserialize` at all. Decoding the spec from the front would shift them.

use libfuzzer_sys::fuzz_target;
use safetensors::slice::TensorIndexer;
use safetensors::SafeTensors;
use std::num::NonZeroUsize;
use std::ops::Bound;

/// Bytes reserved at the end of the input for the slice spec: one byte of
/// slice count, then two bytes per indexer.
const SPEC_LEN: usize = 7;
const MAX_SLICES: usize = 3;

fn decode_bound(kind: u8, value: u8) -> Bound<usize> {
    match kind {
        0 => Bound::Unbounded,
        1 => Bound::Included(value as usize),
        _ => Bound::Excluded(value as usize),
    }
}

/// Decode up to [`MAX_SLICES`] indexers from the spec tail. Values are left
/// unclamped so that out-of-range indices reach the bounds checks.
fn decode_slices(spec: &[u8; SPEC_LEN]) -> Vec<TensorIndexer> {
    let n = (spec[0] as usize) % (MAX_SLICES + 1);
    let mut slices = Vec::with_capacity(n);
    for i in 0..n {
        let flags = spec[1 + i * 2];
        let value = spec[2 + i * 2];
        if flags & 1 == 0 {
            slices.push(TensorIndexer::Select(value as usize));
        } else {
            let start = decode_bound((flags >> 1) & 0b11, value & 0x0f);
            let stop = decode_bound((flags >> 3) & 0b11, value >> 4);
            let step = NonZeroUsize::new(((flags >> 5) & 0b111) as usize + 1)
                .expect("step is at least 1");
            slices.push(TensorIndexer::Narrow(start, stop, step));
        }
    }
    slices
}

fuzz_target!(|data: &[u8]| {
    if data.len() <= SPEC_LEN {
        return;
    }
    let (file, spec) = data.split_at(data.len() - SPEC_LEN);
    let spec: &[u8; SPEC_LEN] = spec.try_into().expect("split at len - SPEC_LEN");
    let slices = decode_slices(spec);

    let Ok(st) = SafeTensors::deserialize(file) else {
        return;
    };

    for (name, view) in st.tensors() {
        let Ok(iterator) = view.sliced_data(&slices) else {
            continue;
        };
        let promised = iterator.remaining_byte_len();
        let newshape = iterator.newshape();

        let mut yielded = 0usize;
        for chunk in iterator {
            yielded += chunk.len();
        }
        assert_eq!(
            yielded, promised,
            "slice of {name:?} yielded {yielded} bytes, promised {promised}"
        );

        // The post-slice shape has to describe exactly the bytes handed out.
        let nelements = newshape.iter().copied().try_fold(1usize, usize::checked_mul);
        if let Some(nbits) = nelements.and_then(|n| n.checked_mul(view.dtype().bitsize())) {
            if nbits % 8 == 0 {
                assert_eq!(
                    yielded,
                    nbits / 8,
                    "slice of {name:?} disagrees with its own newshape"
                );
            }
        }
    }
});
