#![no_main]
//! Fuzz the untrusted-input entry point of the safetensors parser.
//!
//! `SafeTensors::deserialize` reads an 8-byte little-endian header length, a
//! JSON header of that length, and per-tensor `(dtype, shape, [begin, end))`
//! descriptors, then hands out `TensorView`s that index straight back into the
//! caller's buffer. Everything after a successful parse is therefore trusting
//! `Metadata::validate`, so this target does not stop at `deserialize`: it
//! walks every view the way a consumer would, and asserts the one invariant
//! `validate` promises. A panic here means a header got past validation that
//! should not have.

use libfuzzer_sys::fuzz_target;
use safetensors::SafeTensors;

fuzz_target!(|data: &[u8]| {
    // The header-only path, which downstream code uses to size an allocation
    // before reading any tensor data.
    let _ = SafeTensors::read_metadata(data);

    let Ok(st) = SafeTensors::deserialize(data) else {
        return;
    };

    // `tensors()` slices the data buffer with the header's own offsets
    // (`data[begin..end]`), so a validation gap surfaces here as a panic
    // rather than in `deserialize`.
    for (name, view) in st.tensors() {
        // Every tensor named in the header must be retrievable by that name.
        assert!(st.tensor(&name).is_ok(), "named tensor {name:?} not found");

        // `Metadata::validate` guarantees `end - begin` is exactly the byte
        // size implied by the shape and dtype, and that the size is a whole
        // number of bytes. Checked math throughout: an overflowing shape is
        // rejected by validate, so it cannot reach this point.
        let nelements = view
            .shape()
            .iter()
            .copied()
            .try_fold(1usize, usize::checked_mul);
        if let Some(nbits) = nelements.and_then(|n| n.checked_mul(view.dtype().bitsize())) {
            assert_eq!(nbits % 8, 0, "tensor {name:?} is not byte-aligned");
            assert_eq!(
                view.data().len(),
                nbits / 8,
                "tensor {name:?} view length disagrees with shape/dtype"
            );
        }
    }

    // The borrowing iterator is a separate code path from `tensors()`.
    for (_name, view) in st.iter() {
        let _ = view.data().len();
    }

    assert_eq!(st.len(), st.names().len());
});
