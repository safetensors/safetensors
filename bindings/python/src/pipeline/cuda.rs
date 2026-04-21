//! Runtime-discovered CUDA driver bindings.
//!
//! TODO(P2): dlopen `libcuda.so.1` via `libloading::Library`, resolve the
//! ~12 symbols we need (`cuInit`, `cuDeviceGet`, `cuDevicePrimaryCtxRetain`,
//! `cuDevicePrimaryCtxRelease`, `cuCtxPushCurrent_v2`, `cuCtxPopCurrent_v2`,
//! `cuStreamCreate`, `cuStreamDestroy`, `cuStreamSynchronize`,
//! `cuEventCreate`, `cuEventDestroy`, `cuEventRecord`, `cuEventSynchronize`,
//! `cuMemAlloc_v2`, `cuMemFree_v2`, `cuMemHostAlloc`, `cuMemFreeHost`,
//! `cuMemcpyHtoDAsync_v2`, `cuDeviceGetPCIBusId`), and wrap each in a safe
//! Rust function that returns `PipelineResult<_>`.
//!
//! Use primary-context attach (`cuDevicePrimaryCtxRetain` + push/pop), not
//! `cuCtxCreate`. PyTorch initializes the primary context on first use; we
//! run on that context, not compete with it.

#![allow(dead_code)] // scaffolding; implemented in P2.
