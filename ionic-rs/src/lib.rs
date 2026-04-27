//! ionic-rs — Rust port of Morgan Funtowicz's `ionic` C library.
//!
//! Three-stage pipeline for saturating PCIe with safetensors weight loads:
//! io_uring engine with registered pinned staging → DMA worker on a dedicated
//! CUDA stream → GPU buffer, with atomic per-tensor "loaded" tracking.
//!
//! P3 ships the io_uring engine ([`iouring::ReadEngine`]) and the CUDA
//! driver-API wrappers ([`cuda::CuDevice`], [`cuda::CuStream`], …).
//! The pipeline orchestration that ties them together arrives in P4.
//!
//! ## Linux-only
//!
//! `io_uring` is a Linux kernel interface; this crate has no portability
//! story. CUDA is discovered at runtime via `libloading`, so a single build
//! works on Linux boxes with and without an Nvidia driver — calls to CUDA
//! primitives return [`Error::CudaUnavailable`] when the driver is absent.

#![cfg_attr(not(target_os = "linux"), allow(dead_code))]

pub mod cuda;
pub mod error;
pub mod iouring;
pub mod numa;

pub use error::{Error, Result};
