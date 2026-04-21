//! io_uring I/O engine with registered pinned staging buffers.
//!
//! TODO(P3): port the ionic engine from `~/ionic/lib/platform/linux/iouring.c`:
//!   - slot pool (QD=64 by default, aligned 128 KiB chunks)
//!   - registered buffers (`io_uring_register_buffers`) and files
//!     (`io_uring_register_files`)
//!   - offset coalescing (`get_seq_chunks` at ionic `:168` — dedupe path+offset
//!     pairs, build scatter-gather entries over tensor byte ranges)
//!   - `fetch()` submit+poll loop using `IORING_OP_READ_FIXED`
//!   - SQPOLL thread pinned to a CPU on the GPU's NUMA node via
//!     `Builder::setup_sqpoll_cpu(cpu)`; current thread pinned via
//!     `numa::pin_current_thread(..)`

#![allow(dead_code)] // scaffolding; implemented in P3.
