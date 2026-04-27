//! Python bindings for the prefetch pipeline.
//!
//! Thin wrappers over [`ionic_rs`]. The engine code lives in that sibling
//! crate (cuda FFI, io_uring, NUMA, errors); this module exposes only the
//! Python-facing surface — currently `PrefetchHandle` plus the
//! `safe_open.prefetch()` method which constructs it.
//!
//! **Phase state:** P1 lands the API shape with a trivial eager-load fallback
//! (iterates the name list and calls the existing `get_tensor`). P2/P3 have
//! the building blocks (CUDA wrappers, io_uring engine) ready in `ionic-rs`.
//! P4 will plug them in here without changing the Python surface.

use std::collections::VecDeque;

use pyo3::prelude::*;

use crate::{Open, SafetensorError};

// ── Error bridge ────────────────────────────────────────────────────────
//
// `ionic_rs::Error` is pyo3-free by design. The bindings crate provides this
// adapter so engine errors surface as the project's existing exception type.

pub(crate) struct PyIonicError(pub ionic_rs::Error);

impl From<ionic_rs::Error> for PyIonicError {
    fn from(e: ionic_rs::Error) -> Self {
        Self(e)
    }
}

impl From<PyIonicError> for PyErr {
    fn from(e: PyIonicError) -> Self {
        SafetensorError::new_err(e.0.to_string())
    }
}

// ── PrefetchHandle ──────────────────────────────────────────────────────

/// Handle returned by `safe_open.prefetch(...)`. Single-use iterator over
/// `(name, tensor)` pairs with drain-on-delivery semantics.
///
/// **As-completed futures semantic.** Once a tensor is yielded the handle
/// releases its reference; the caller becomes the sole owner. This keeps
/// peak memory at 1× the model size rather than 2× while the handle is
/// alive. Consequences: iteration is one-shot — re-iterating a drained
/// handle yields nothing.
///
/// **Iteration order is not specified.** The backend yields tensors in
/// whatever order they become ready — mmap+threadpool in completion order,
/// io_uring+DMA in H2D-event-firing order, future pread in offset order.
/// Callers that need a specific ordering should sort the yielded pairs.
///
/// **Cleanup on drop.** When the handle goes out of scope, any in-flight
/// work is cancelled and resources released. P1 has nothing to clean up;
/// P4 cancels io_uring submissions, joins the DMA worker, and syncs the
/// CUDA stream.
///
/// P1 implementation: `build_trivial` populates the queue by calling
/// `get_tensor()` eagerly per name. P4 swaps in a channel the DMA worker
/// pushes into as H2D events fire.
#[pyclass]
pub struct PrefetchHandle {
    /// Not-yet-delivered (name, tensor) pairs. Drained by `__next__` as the
    /// caller iterates. In P4 this becomes a blocking queue fed by the DMA
    /// worker thread.
    pending: VecDeque<(String, Py<PyAny>)>,
}

impl PrefetchHandle {
    pub fn build_trivial(open: &Open, names: Vec<String>) -> PyResult<Self> {
        let mut pending = VecDeque::with_capacity(names.len());
        for name in names {
            let t = open.get_tensor(&name)?;
            pending.push_back((name, t));
        }
        Ok(Self { pending })
    }
}

#[pymethods]
impl PrefetchHandle {
    /// Remaining (not-yet-delivered) tensor count. Useful for progress bars
    /// and sanity checks; decrements as iteration drains the handle.
    pub fn __len__(&self) -> usize {
        self.pending.len()
    }

    /// Handle is its own iterator — single-use.
    pub fn __iter__(slf: PyRef<'_, Self>) -> PyRef<'_, Self> {
        slf
    }

    pub fn __next__(mut slf: PyRefMut<'_, Self>) -> Option<(String, Py<PyAny>)> {
        slf.pending.pop_front()
    }
}

pub(crate) fn register(m: &Bound<'_, pyo3::types::PyModule>) -> PyResult<()> {
    m.add_class::<PrefetchHandle>()?;
    Ok(())
}
