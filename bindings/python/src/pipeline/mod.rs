//! Prefetch pipeline — Linux-only fast path for bulk tensor loads.
//!
//! Port of Morgan's `ionic` C library (`~/ionic`, branch `feat/planner_cuda`).
//! Architecture, in brief:
//!
//!   io_uring engine (registered pinned staging, QD=64, offset-coalesced)
//!     → DMA worker thread (2 async CUDA streams + events)
//!       → GPU buffer
//!         → atomic per-tensor "loaded" counter, signals readiness to main
//!
//! The public Python surface is one method on `safe_open`:
//! `.prefetch(names, max_inflight=8) -> PrefetchHandle`. The handle is a
//! single-use iterator — that's the whole API.
//!
//! **Phase state:** P1 lands the API shape with a trivial eager-load fallback
//! (iterates the name list and calls the existing `get_tensor`). Subsequent
//! phases swap the internals without changing the Python surface:
//!   - P2: CUDA FFI + NUMA pinning + dedicated stream.
//!   - P3: io_uring read backend with registered staging.
//!   - P4: DMA worker thread + atomic load tracking + true as-completed.
//!   - P5: DLPack output, `dtype=` fusion, error-surface polish.

pub mod cuda;
pub mod error;
pub mod iouring;
pub mod numa;

use std::collections::VecDeque;

use pyo3::prelude::*;

use crate::Open;

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

// ── Debug probes ───────────────────────────────────────────────────────
//
// Leading-underscore pyfunctions — internal, may change or disappear. These
// exist so tests under `bindings/python/tests/test_pipeline.py` can exercise
// the CUDA wrapper stack without depending on the real prefetch integration
// (which doesn't arrive until P4). A production caller should never import
// these.

use pyo3::types::{PyDict, PyList};

/// Enumerate visible CUDA devices and their sysfs-derived NUMA mapping.
///
/// Returns a list of dicts: `{"ordinal": int, "name": str, "pci_bus_id": str,
/// "numa_node": int, "numa_cpus": list[int]}`. If the CUDA driver is not
/// loadable, returns a one-element list with `{"error": "..."}` so tests
/// can skip cleanly rather than raising.
#[pyfunction]
fn _debug_cuda_probe(py: Python<'_>) -> PyResult<Py<PyAny>> {
    let list = PyList::empty(py);

    let count = match cuda::CuDevice::count() {
        Ok(n) => n,
        Err(e) => {
            let d = PyDict::new(py);
            d.set_item("error", e.to_string())?;
            list.append(d)?;
            return Ok(list.into());
        }
    };

    for ordinal in 0..count {
        let d = PyDict::new(py);
        d.set_item("ordinal", ordinal)?;

        match cuda::CuDevice::get(ordinal) {
            Ok(dev) => {
                d.set_item("name", dev.name().unwrap_or_else(|e| format!("<err: {e}>")))?;
                let bdf = dev.pci_bus_id().unwrap_or_else(|e| format!("<err: {e}>"));
                d.set_item("pci_bus_id", &bdf)?;

                let node = numa::numa_node_for_pci(&bdf).unwrap_or(-1);
                d.set_item("numa_node", node)?;
                let cpus = if node >= 0 {
                    numa::cpulist_for_node(node).unwrap_or_default()
                } else {
                    Vec::new()
                };
                d.set_item("numa_cpus", cpus)?;
            }
            Err(e) => {
                d.set_item("error", e.to_string())?;
            }
        }
        list.append(d)?;
    }

    Ok(list.into())
}

/// End-to-end smoke of the CUDA wrapper stack: retain primary context,
/// push/pop, create stream + event, allocate pinned + device, H2D copy,
/// synchronize, release. Returns bytes copied on success. Raises
/// `SafetensorError` if any CUDA call fails.
#[pyfunction]
fn _debug_cuda_ctx_smoke(ordinal: i32) -> PyResult<usize> {
    const BYTES: usize = 4096;
    let dev = cuda::CuDevice::get(ordinal)?;
    let ctx = cuda::CuContext::primary_retain(dev)?;
    let bytes = ctx.with_current(|| {
        let stream = cuda::CuStream::new()?;
        let event = cuda::CuEvent::new()?;
        let mut pinned = cuda::PinnedBuf::alloc(BYTES)?;
        let dev_buf = cuda::DeviceBuf::alloc(BYTES)?;
        for (i, b) in pinned.as_mut_slice().iter_mut().enumerate() {
            *b = (i & 0xFF) as u8;
        }
        cuda::memcpy_h2d_async(dev_buf.as_device_ptr(), pinned.as_ptr(), BYTES, &stream)?;
        event.record(&stream)?;
        event.synchronize()?;
        Ok(BYTES)
    })?;
    Ok(bytes)
}

pub(crate) fn register(m: &Bound<'_, pyo3::types::PyModule>) -> PyResult<()> {
    m.add_class::<PrefetchHandle>()?;
    m.add_function(wrap_pyfunction!(_debug_cuda_probe, m)?)?;
    m.add_function(wrap_pyfunction!(_debug_cuda_ctx_smoke, m)?)?;
    Ok(())
}
