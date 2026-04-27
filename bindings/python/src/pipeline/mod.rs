//! Python bindings for the prefetch pipeline.
//!
//! Thin wrappers over [`ionic_rs`]. The engine code lives in that sibling
//! crate (cuda FFI, io_uring, NUMA, errors); this module exposes only the
//! Python-facing surface — `PrefetchHandle`, the `safe_open.prefetch()`
//! method, and the DLPack v1 wrapping that hands tensors back to whichever
//! framework the user opened the file under.
//!
//! ## Backend dispatch
//!
//! [`PrefetchHandle::build`] picks the backend based on framework + device:
//! - **pytorch + CUDA:** `build_cuda` — allocates a `DeviceBuf` per tensor
//!   via the CUDA driver API, drives [`ionic_rs::pipeline::CudaPipeline`]
//!   outside the GIL to read + DMA in one pass, hands each buffer back as
//!   a torch tensor via DLPack v1.
//! - **everything else:** `build_trivial` — eager `get_tensor()` loop, the
//!   P1 fallback. Existing CPU paths already alias mmap and are fast; this
//!   keeps behavior unchanged for non-pytorch frameworks.

mod dlpack;

use std::collections::VecDeque;

use pyo3::intern;
use pyo3::prelude::*;
use pyo3::types::IntoPyDict;

use crate::{get_pydtype, Device, Framework, Open, OpenSources, SafetensorError, TORCH_MODULE};

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
#[pyclass]
pub struct PrefetchHandle {
    /// Not-yet-delivered (name, tensor) pairs. Drained by `__next__` as the
    /// caller iterates. Whether the construction populated this synchronously
    /// (P1/CUDA) or async-via-channel (future) is opaque to Python.
    pending: VecDeque<(String, Py<PyAny>)>,
}

impl PrefetchHandle {
    /// Dispatch entry point — picks the backend based on the open file's
    /// framework + device. Falls through to the eager `get_tensor` loop for
    /// any combination we don't have a fast path for yet.
    pub fn build(open: &Open, names: Vec<String>) -> PyResult<Self> {
        match (&open.framework, &open.device) {
            (Framework::Pytorch, Device::Cuda(ordinal)) => {
                Self::build_cuda(open, names, *ordinal as i32)
            }
            _ => Self::build_trivial(open, names),
        }
    }

    /// Multi-source-file CUDA fast path. Routes one
    /// [`ionic_rs::pipeline::CudaPipeline`] across N source files: one
    /// io_uring ring with all N files registered, one DMA worker, two
    /// streams. Reads fan out across the underlying storage (md0 RAID-0,
    /// multiple Weka OSTs, etc.); H2D goes through the same two-engine
    /// setup as the single-source path.
    ///
    /// This is what ionic does — `pipeline_execute` over a plan that
    /// spans every source at once. The previous threaded-per-source
    /// approach we'd fallen into spun up N pipelines with N rings and N
    /// DMA workers, which competed for the GPU's two physical copy
    /// engines and replicated N× of construction overhead.
    pub fn build_sources(open: &OpenSources, names: Vec<String>) -> PyResult<Self> {
        match (&open.framework, &open.device) {
            (Framework::Pytorch, Device::Cuda(ordinal)) => {
                Self::build_cuda_sources(open, names, *ordinal as i32)
            }
            (_, Device::Cpu) => Err(SafetensorError::new_err(
                "multi-source prefetch on CPU is not implemented yet \
                 (use safe_open + get_tensor for now)",
            )),
            _ => Err(SafetensorError::new_err(format!(
                "multi-source prefetch: unsupported framework/device combo \
                 ({:?} on {:?})",
                open.framework, open.device
            ))),
        }
    }

    fn build_cuda_sources(
        open: &OpenSources,
        names: Vec<String>,
        ordinal: i32,
    ) -> PyResult<Self> {
        Python::attach(|py| -> PyResult<Self> {
            let torch = TORCH_MODULE
                .get()
                .ok_or_else(|| SafetensorError::new_err("torch module not initialized"))?
                .bind(py);

            // One pipeline, all sources registered. Construction outside
            // the GIL — io_uring queue init + 64 PinnedBuf allocs +
            // stream/event creation are multi-millisecond.
            let source_paths: Vec<std::path::PathBuf> =
                open.sources.iter().map(|s| s.filename.clone()).collect();
            let mut pipeline = py
                .detach(|| {
                    let mut p = ionic_rs::pipeline::CudaPipeline::new(
                        ordinal,
                        ionic_rs::iouring::DEFAULT_QUEUE_DEPTH,
                        ionic_rs::iouring::DEFAULT_CHUNK_BYTES,
                    )?;
                    let mut fds = Vec::with_capacity(source_paths.len());
                    for path in &source_paths {
                        fds.push(p.register_file(path)?);
                    }
                    Ok::<_, ionic_rs::Error>((p, fds))
                })
                .map_err(PyIonicError)?;
            let fd_indices = pipeline.1;
            let pipeline_ref = &mut pipeline.0;

            // Phase 1 (GIL): per-tensor device alloc + segment build.
            let mut bufs: Vec<ionic_rs::cuda::DeviceBuf> = Vec::with_capacity(names.len());
            let mut shapes: Vec<Vec<i64>> = Vec::with_capacity(names.len());
            let mut dtypes: Vec<safetensors::Dtype> = Vec::with_capacity(names.len());
            let mut segments: Vec<ionic_rs::iouring::LogicalSegment> =
                Vec::with_capacity(names.len());

            for (i, name) in names.iter().enumerate() {
                let (sidx, info) = open.info(name).ok_or_else(|| {
                    SafetensorError::new_err(format!("tensor '{name}' not in any source"))
                })?;
                let source = &open.sources[sidx];
                let nbytes = info.data_offsets.1 - info.data_offsets.0;
                let buf = pipeline_ref.alloc_device_buf(nbytes).map_err(PyIonicError)?;
                let dst_ptr = buf.as_device_ptr() as usize;
                let from = (info.data_offsets.0 + source.offset) as u64;
                let to = (info.data_offsets.1 + source.offset) as u64;
                segments.push(ionic_rs::iouring::LogicalSegment {
                    fd_idx: fd_indices[sidx],
                    from,
                    to,
                    dst_offset: dst_ptr,
                    user_data: i as u64,
                });
                bufs.push(buf);
                shapes.push(info.shape.iter().map(|&n| n as i64).collect());
                dtypes.push(info.dtype);
            }

            // Phase 2 (no GIL): one pipeline.run drives the whole
            // checkpoint across all source files.
            py.detach(|| pipeline_ref.run(&segments)).map_err(PyIonicError)?;

            // Phase 3 (GIL): wrap each populated DeviceBuf in DLPack →
            // torch.from_dlpack. Same wrap as the single-source path;
            // just iterating the cross-source list now.
            let from_dlpack = torch.getattr(intern!(py, "from_dlpack"))?;
            let device = dlpack::cuda_device(ordinal);
            let mut pending: VecDeque<(String, Py<PyAny>)> = VecDeque::with_capacity(names.len());
            for ((name, buf), (shape, dtype)) in
                names.into_iter().zip(bufs).zip(shapes.into_iter().zip(dtypes))
            {
                let dl_dtype = dlpack::dtype_to_dlpack(dtype);
                let managed = ionic_rs::dlpack::make_managed(buf, shape, dl_dtype, device);
                let capsule = dlpack::make_capsule(py, managed)?;
                let tensor = from_dlpack.call1((capsule,))?;
                pending.push_back((name, tensor.into_pyobject(py)?.unbind().into_any()));
            }
            Ok(Self { pending })
        })
    }

    /// Eager fallback: iterate names, call `get_tensor` for each. Same shape
    /// as P1 — used for CPU targets, non-pytorch frameworks, and any future
    /// device we don't have a pipeline backend for.
    fn build_trivial(open: &Open, names: Vec<String>) -> PyResult<Self> {
        let mut pending = VecDeque::with_capacity(names.len());
        for name in names {
            let t = open.get_tensor(&name)?;
            pending.push_back((name, t));
        }
        Ok(Self { pending })
    }

    /// CUDA fast path: allocate a `DeviceBuf` per tensor via the CUDA
    /// driver API, drive the ionic-rs pipeline outside the GIL to read +
    /// DMA the file in one pass, then hand each buffer back to PyTorch as
    /// a tensor via DLPack v1 (`torch.from_dlpack`).
    ///
    /// The `DeviceBuf` ownership transfers into the DLPack capsule's
    /// `manager_ctx`; once `from_dlpack` consumes the capsule, torch owns
    /// the buffer and frees it (via our deleter) when its tensor handle
    /// drops. No torch caching allocator involvement — we control the
    /// lifecycle end-to-end, which is what makes future framework-agnostic
    /// outputs (JAX / Paddle / MLX) a one-line dispatch change.
    fn build_cuda(open: &Open, names: Vec<String>, ordinal: i32) -> PyResult<Self> {
        Python::attach(|py| -> PyResult<Self> {
            let torch = TORCH_MODULE
                .get()
                .ok_or_else(|| SafetensorError::new_err("torch module not initialized"))?
                .bind(py);

            // Pipeline construction allocates QD pinned staging buffers,
            // creates streams + events, retains the primary context. Cheap
            // re-construction on every call for now; future caching belongs
            // on the safe_open handle if profiling shows it matters.
            let filename = open.filename.clone();
            let mut pipeline = py
                .detach(|| {
                    ionic_rs::pipeline::CudaPipeline::new(
                        ordinal,
                        ionic_rs::iouring::DEFAULT_QUEUE_DEPTH,
                        ionic_rs::iouring::DEFAULT_CHUNK_BYTES,
                    )
                })
                .map_err(PyIonicError)?;
            let _fd_idx = pipeline.register_file(&filename).map_err(PyIonicError)?;

            // Phase 1 (GIL held): allocate per-tensor device buffers and
            // build the logical segment list pointing into them. The
            // pipeline's allocator threads ctx push/pop internally so we
            // don't need to babysit the CUDA context here.
            let mut bufs: Vec<ionic_rs::cuda::DeviceBuf> = Vec::with_capacity(names.len());
            let mut shapes: Vec<Vec<i64>> = Vec::with_capacity(names.len());
            let mut dtypes: Vec<safetensors::Dtype> = Vec::with_capacity(names.len());
            let mut segments: Vec<ionic_rs::iouring::LogicalSegment> =
                Vec::with_capacity(names.len());

            for (i, name) in names.iter().enumerate() {
                let info = open.metadata.info(name).ok_or_else(|| {
                    SafetensorError::new_err(format!("File does not contain tensor {name}"))
                })?;
                let nbytes = info.data_offsets.1 - info.data_offsets.0;
                let buf = pipeline.alloc_device_buf(nbytes).map_err(PyIonicError)?;
                let dst_ptr = buf.as_device_ptr() as usize;

                let from = (info.data_offsets.0 + open.offset) as u64;
                let to = (info.data_offsets.1 + open.offset) as u64;
                segments.push(ionic_rs::iouring::LogicalSegment {
                    fd_idx: 0,
                    from,
                    to,
                    // dst_offset is reinterpreted as a CUdeviceptr by the
                    // ionic pipeline — see CudaPipeline::run.
                    dst_offset: dst_ptr,
                    user_data: i as u64,
                });
                bufs.push(buf);
                shapes.push(info.shape.iter().map(|&n| n as i64).collect());
                dtypes.push(info.dtype);
            }

            // Phase 2 (GIL released): drive the pipeline. Multi-millisecond
            // I/O + H2D bound; holding the GIL across this would block any
            // other Python thread.
            py.detach(|| pipeline.run(&segments)).map_err(PyIonicError)?;

            // Phase 3 (GIL held): wrap each populated DeviceBuf as a DLPack
            // v1 capsule, hand to torch.from_dlpack. Ownership of each
            // DeviceBuf transfers into its capsule via make_managed; once
            // from_dlpack consumes, the capsule is renamed and torch owns
            // the lifecycle.
            let from_dlpack = torch.getattr(intern!(py, "from_dlpack"))?;
            let device = dlpack::cuda_device(ordinal);

            let mut pending: VecDeque<(String, Py<PyAny>)> = VecDeque::with_capacity(names.len());
            for ((name, buf), (shape, dtype)) in
                names.into_iter().zip(bufs).zip(shapes.into_iter().zip(dtypes))
            {
                let dl_dtype = dlpack::dtype_to_dlpack(dtype);
                let managed = ionic_rs::dlpack::make_managed(buf, shape, dl_dtype, device);
                let capsule = dlpack::make_capsule(py, managed)?;
                let tensor = from_dlpack.call1((capsule,))?;
                pending.push_back((name, tensor.into_pyobject(py)?.unbind().into_any()));
            }

            Ok(Self { pending })
        })
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
