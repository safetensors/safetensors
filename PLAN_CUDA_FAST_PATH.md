# CUDA fast-path for safetensors — session handoff

Briefing doc for continuing this work on a CUDA-equipped machine in a new
Claude Code session. Read this first. It encodes the problem, the locked
design decisions, the PR sequence, and what to do right now.

---

## Mission

Fix the `safe_open(device="cuda:n")` path, which currently tops out around
85 MB/s loading safetensors weights onto GPU (see
[safetensors/safetensors#736](https://github.com/safetensors/safetensors/pull/736)).
The primary consumer is `transformers`' weight loader
(`src/transformers/core_model_loading.py`), particularly the TP loader
pattern on the `refactor-tp-loading` branch which works around the slow
path via its own thread pool + manual pinning.

Long-term goal: a proper pread-based prefetch pipeline that saturates NVMe
bandwidth. Short-term: close the obvious perf gap with pinned staging +
async H2D on a safetensors-owned stream, expose it behind a new opt-in
`prefetch()` API that doesn't touch the existing public surface.

---

## Constraints (from the maintainer)

1. **Do not change the existing public API.** `safe_open`, `get_tensor`,
   `get_slice`, `load_file`, `save_file` keep their current behavior.
   Users opt into the new fast path explicitly. Old API can be made
   faster later without breaking callers.
2. **Reduce the new public API surface as much as possible.** Ideally
   one new method on the `safe_open` handle: `.prefetch(...)`.
3. **Heavy lifting in Rust.** Thin Python wrappers only. The
   `bindings/python` crate owns threads, pinned memory, streams, FFI.
4. **Safety guarantees must not regress.** mmap lifetime tied to handle;
   DLPack deleter as single owner of device buffers; FFI contained behind
   safe Rust; bounds checks preserved.
5. **Core `safetensors` crate stays minimal.** Only promote abstractions
   there after they've been proven in `bindings/python`. Current core
   crate has 3 deps and ~1700 LOC — preserve that discipline.

---

## Branch state

Working on branch: **`feat/cuda_load_bench_harness`**

What just landed on this branch:

- `bindings/python/benches/test_cuda_load.py` — new file. Baseline
  benchmarks for CUDA load paths using pytest-benchmark. See below for
  the paths it covers and how to run it.

Untracked scratch files at repo root (keep or delete as you like):

- `AGENTS.md`
- `PLAN_MULTI_DEVICE_LOADING.md`
- `TODO.md`
- `.vscode/`

---

## Immediate next steps

1. **Run the new bench harness on the CUDA machine** to establish
   baseline numbers for every downstream PR to measure against.

   ```bash
   cd bindings/python
   # If not built yet:
   pip install -e . --no-build-isolation
   pip install pytest pytest-benchmark torch

   # Collect-only first (sanity check imports, should show ~8 tests):
   python -m pytest benches/test_cuda_load.py --collect-only -q

   # Full run (file ~2 GB, takes a few minutes):
   python -m pytest benches/test_cuda_load.py \
       --benchmark-only \
       --benchmark-columns=min,mean,median,stddev,rounds \
       --benchmark-sort=mean \
       -v
   ```

   Save the output. Each test stores `file_size_mb` in `extra_info`, so
   MB/s is `file_size_mb / mean_seconds`.

   **Cold-cache caveat:** the PR #736 ~85 MB/s report is cold-cache,
   disk-I/O-bound. If you don't have root to `drop_caches`, the bench
   runs warm-cache after round 1 and is instead CPU-memcpy-bound +
   pageable-H2D-bound. That's actually the regime PR 1 and PR 3
   attack, so warm-cache baselines are what we compare against. The
   cold number is a separate story PR 6 (pread backend) owns.

2. **Commit** the bench harness + this plan doc once numbers look sane.
   Then open a PR for the bench harness as **PR 0** (see roadmap below).

3. **Start PR 1** (kill the bytearray copy — see details below). Use the
   `test_safe_open_cuda_per_tensor` / `test_safe_open_cpu_then_to_cuda`
   numbers as the before-and-after comparison (both live on the slow
   `get_tensor` path).

### Captured baseline (warm cache, 2 GB file, H100 PCIe Gen5)

Run on `feat/cuda_load_bench_harness` at commit `fbe545e`
(2026-04-20). Artifacts at `bindings/python/benches/baseline_warm_cache.{json,log}`.

| test | mean (ms) | MB/s |
|---|---:|---:|
| `test_load_file_cpu` (already mmap-aliased, no copy) | 1.36 | ~1.5 TB/s |
| `test_transformers_pattern_threaded[1]` | 196.8 | 10,387 |
| `test_safe_open_cpu_then_to_cuda` | 198.4 | 10,302 |
| `test_safe_open_cuda_per_tensor` (**PR #736 path**) | 199.8 | 10,230 |
| `test_transformers_pattern_threaded[16]` | 200.1 | 10,214 |
| `test_transformers_pattern_threaded[8]` | 200.5 | 10,196 |
| `test_transformers_pattern_threaded[4]` | 208.1 | 9,822 |
| `test_load_file_cuda` | 209.5 | 9,756 |

All CUDA paths cluster at ~200 ms: every one eats one bytearray
memcpy (~135 ms at 15 GB/s DDR) + one pageable H2D on the default
stream (~130 ms at ~15 GB/s on Gen5, partially overlapping in the
driver). Thread count doesn't help because both resources serialize.
This is exactly what PR 1 + PR 3 are designed to unblock.

---

## What the bench harness covers

File: `bindings/python/benches/test_cuda_load.py`

Synthetic Llama-ish weights (~2 GB in bf16) written once per module
run via a `tmp_path_factory` fixture. Test matrix:

| Test | What it measures |
|---|---|
| `test_load_file_cpu` | CPU-only baseline (floor: no device copy) |
| `test_load_file_cuda` | Public `load_file(device="cuda:0")` bulk path |
| `test_safe_open_cuda_per_tensor` | The **PR #736 slow path**: `safe_open(device="cuda:0") + get_tensor` |
| `test_safe_open_cpu_then_to_cuda` | Two-step `get_tensor → .to("cuda")` — transformers' non-TP path |
| `test_transformers_pattern_threaded[n_workers]` | Worker does `get_slice[...] → .to(dtype)`; main thread does `.to("cuda")`. Parametrized 1/4/8/16. Mirrors the TP loader worker body at `core_model_loading.py:1559` + `_materialize` at `:1564-1570`. Default workers in transformers is 4 (`GLOBAL_WORKERS`, `core_model_loading.py:853`). |

Caveat documented in the file docstring: OS page cache is warm after the
first iteration. Absolute cold-cache numbers will be lower; relative
ordering is what matters for PR-over-PR comparison.

---

## Code map — where the slow path lives

**The bytearray copy that PR 1 kills:**
- `bindings/python/src/lib.rs:741` —
  `PyByteArray::new(py, data)` allocates a fresh bytearray and copies
  the whole mmap slice into it before handing to `torch.frombuffer`.
  This is one CPU memcpy per tensor that can be eliminated via the
  buffer protocol (or `PyMemoryView` around an object that pins the
  `Arc<Mmap>`).

**The forced sync `.to(device)` on the default stream:**
- `bindings/python/src/lib.rs:1520` —
  `tensor.call_method("to", (device,), Some(&kwargs))` with empty
  kwargs. Blocking H2D on the default CUDA stream, from pageable
  memory. This is what the PR #736 user worked around.

**The `Open` / `safe_open` structure:**
- `bindings/python/src/lib.rs:525` — `struct Open`
- `bindings/python/src/lib.rs:974` — `struct safe_open` (the pyclass)
- `bindings/python/src/lib.rs:727` — `Open::get_tensor`
- `bindings/python/src/lib.rs:943` — `Open::get_slice`
- `bindings/python/src/lib.rs:1397` — `fn create_tensor` (the dispatch
  point for per-framework tensor construction, including the
  pytorch-cuda `.to()` call)

**Dependencies** (`bindings/python/Cargo.toml`): pyo3, memmap2,
serde_json, plus `safetensors` from the workspace. Notably lean — adding
`cudarc` would ~3-4× the transitive dep count, which is why we settled
on raw FFI for CUDA (~12 symbols, ~200 LOC of `unsafe extern "C"`).

---

## What `transformers`' TP loader does

Reference branch: `refactor-tp-loading` in the transformers repo.
File: `src/transformers/core_model_loading.py` (~1678 LOC).

Relevant pieces (verified against `refactor-tp-loading` at commit
`21ae32d470`):

- **Thread pool for concurrent reads** (`core_model_loading.py:1235`)
  — `ThreadPoolExecutor(max_workers=GLOBAL_WORKERS)`, where
  `GLOBAL_WORKERS = min(4, os.cpu_count())` (`:853`). Created per
  `convert_and_load_state_dict_in_model` call, shut down in the
  `finally` block at `:1622`. Relies on safetensors being thread-safe
  under multiple `get_slice` materializations.
- **Worker body** (`core_model_loading.py:1559`) —
  `lambda ts, d: ts[...].to(dtype=d)`. Just read + dtype cast. No H2D
  in the worker, no `pin_memory()`, no `non_blocking`.
- **Main-thread H2D** (`core_model_loading.py:1564-1570`,
  `_materialize`) — `fut.result().to(device=tp_device)`. Pageable copy
  on the default CUDA stream. This is what our bench models.
- **Batched pipeline** (`core_model_loading.py:1547-1577`) — `BATCH=64`;
  `_schedule_reads(batch_idx)` queues next batch to the pool while
  `_materialize(cur_futs)` drains the previous batch on the main
  thread. Manual double-buffering still present after the recent
  refactors.
- **No pinned memory on this branch.** They tried it and removed it.
  Pre-`498a72694b` (2026-04-16) the worker ran `_read_to_pinned`,
  which did `cpu.pin_memory()` before returning. That commit renamed
  it to `_read_to_cpu` and dropped the pin. Subsequent refactors
  collapsed it to the current inline lambda. The comment block at
  `:1410-1413` still claims "pinned memory DMA overlap on a dedicated
  CUDA stream" — that is stale documentation, not behavior.
  **Design implication for us:** pinning without an async copy on a
  dedicated stream is a known dead end. PR 3 must ship pinned staging
  **and** a safetensors-owned stream together, or not bother.
- **Rank partitioning** (`core_model_loading.py:918`,
  `partition_mappings_across_ranks`) — greedy bin-packing so only the
  owning rank reads each tensor from disk. The other ranks receive
  over NCCL or symmetric memory.

**What belongs in safetensors (the `.prefetch()` API):**
- Thread pool / worker scheduling
- Pinned staging pool
- CUDA stream ownership + async H2D
- On-disk-offset ordering of reads
- dtype fusion (cast during stage or on GPU)

**What stays in transformers:**
- WeightConverter ops (Chunk, MergeModulelist, Transpose, …)
- TP plan resolution and rank partitioning
- Cross-rank redistribution (scatter / symmetric memory)
- Weight renaming

---

## Locked public API (the only new surface)

```python
with safe_open(path, framework="pt", device="cuda:0") as sf:
    # Returns a single-use iterator over (name, tensor) pairs.
    for name, tensor in sf.prefetch(
        names,                # list[str]  (or  dict[str, slice-spec]  later)
        dtype=None,           # optional, scalar or per-name dict
        max_inflight=8,       # bounded backpressure
    ):
        ...

    # Or materialize all at once:
    weights = dict(sf.prefetch(names))

# __exit__ on sf cancels in-flight work and syncs the stream.
```

Design decisions locked in:

- **Iterator-only API.** `prefetch()` returns a `PrefetchHandle` that's
  a single-use iterator — no `.get(name)`, no `.wait_all()`. One idiom,
  matches `concurrent.futures.as_completed`. If you need a specific
  tensor by name, use the existing `sf.get_tensor(name)`; it doesn't
  benefit from pipelining but the API is already there.
- **As-completed iteration, not request-order.** Backend is free to
  yield in whatever order tensors become ready. Same API works for
  mmap+threadpool (finish order), pread+prefetcher (offset order),
  future FFI backends (transfer-done order). Callers that need a
  specific order should sort the stream themselves.
- **Drain-on-delivery.** Once a tensor is yielded, the handle releases
  its reference. Peak memory stays at 1× the model size rather than 2×
  while the handle is alive.
- **Cleanup on drop.** Handle's `Drop` cancels in-flight io_uring ops,
  joins the DMA worker, and syncs the CUDA stream. No explicit
  `.close()` or `.wait_all()` needed; if the user breaks out of the
  iteration early, cleanup still runs deterministically.
- **Bounded `max_inflight`.** Reader workers block when the ready-queue
  is full. Natural backpressure; prevents runaway memory usage.
- **Sync-on-next contract for CUDA in PR 3.** `next()` synchronizes
  on the tensor's H2D completion before yielding. Loses a touch of
  overlap; keeps semantics foolproof. A `strict_sync=False` variant
  can land later for advanced callers that manage their own streams.
- **DLPack for output format starting PR 3 (CPU path) / PR 4 (CUDA).**
  CPU-path DLPack works for all frameworks with a `from_dlpack`
  consumer on day one. CUDA DLPack lands when the FFI allocator does,
  because before that we'd be wrapping a torch tensor in a DLPack
  capsule just to unwrap it again — pure ceremony.

---

## PR roadmap

| # | Title | Public API change | Where | Notes |
|---|---|---|---|---|
| **0** | **Bench harness** | — | `bindings/python/benches/` | ← done on this branch |
| 1 | Kill bytearray copy in existing mmap path | none | bindings Rust | Improves old API for un-migrated callers |
| 2 | Internal `Loader` / `Plan` refactor (metadata + offset-ordered iteration) | none | bindings Rust | Sets up 3 and 6 |
| 3 | `prefetch()` iterator API + DLPack output (CPU); pytorch-pinned + stream for CUDA path | **+1 method** | bindings Rust | Main user-facing change. CPU DLPack works for all frameworks; CUDA is pytorch-only for now. |
| 4 | Raw FFI CUDA allocations; swap pytorch-pinned machinery under `prefetch`; JAX/Paddle CUDA auto-light up via DLPack | none | bindings Rust (`cuda` feature, dlopen via libloading) | Pure internal swap; same API |
| 5 | `dtype=` fusion (cast during stage or on GPU) | extends kwargs | bindings Rust | |
| 6 | Pread-based backend under `Loader` (selectable or auto for bulk loads) | none | bindings Rust | NVMe-bound speeds |
| 7 | Promote `Loader` + plan/iterator/backend traits from bindings to core `safetensors` crate | none | core + bindings | Earned the promotion |

**Total public API delta across all seven PRs: one new method.**

---

## PR 1 spec (what to tackle after baselines are captured)

**Goal:** Remove the forced bytearray copy in the mmap → torch handoff.
Benefits the current slow path and the existing CPU `get_tensor` / `load_file`
users.

**Change site:** `bindings/python/src/lib.rs:735-749` (the
`Storage::Mmap` branch in `Open::get_tensor`).

**Today:**
```rust
Storage::Mmap(mmap) => {
    let data = &mmap[info.data_offsets.0 + self.offset..info.data_offsets.1 + self.offset];
    let array: PyObject =
        Python::with_gil(|py| PyByteArray::new(py, data).into_any().into());
    create_tensor(&self.framework, info.dtype, &info.shape, array, &self.device)
}
```
That `PyByteArray::new(py, data)` both allocates a fresh Python bytearray
**and** copies the whole mmap slice into it. Then `torch.frombuffer` inside
`create_tensor` aliases that bytearray — zero-copy from bytearray to torch
tensor, but the bytearray itself is already a copy of mmap.

**Target behavior:**
- Hand `torch.frombuffer` / `torch.asarray` a Python object that implements
  the buffer protocol and aliases the mmap region directly. No intermediate
  allocation, no memcpy.
- Attach the `Arc<Mmap>` (or a refcounted handle to it) to the resulting
  torch tensor so the mmap stays alive until the tensor is dropped. The
  Paddle branch already does this via
  `tensor.setattr("_safetensors_storage", storage)?` (see
  `bindings/python/src/lib.rs:836`) — same pattern.

**Implementation approach — two options, pick whichever benches cleaner:**

a. `PyMemoryView::from_object` over a small pyclass wrapper that holds the
   `Arc<Mmap>` and exposes `__buffer__` / buffer protocol via pyo3.

b. Build a `memoryview` in Python from a raw pointer + length, using
   `PyMemoryView::from_buffer` (pyo3 0.25 supports this). Still needs the
   `Arc<Mmap>` attached to the resulting tensor for lifetime.

Option (a) is more idiomatic. Verify pyo3 0.25's buffer-protocol support
(`PyBufferProtocol` or the newer trait approach) on the current pin.

**Correctness checks:**
- Run the full test suite: `pytest bindings/python/tests/ -x`
- Confirm `test_load_file_cpu` (and the equivalent existing torch tests
  in `tests/test_pt_comparison.py`) still pass byte-for-byte.
- Bench delta: `test_safe_open_cuda_per_tensor`,
  `test_safe_open_cpu_then_to_cuda`, and the threaded variants should
  improve by the size of the saved CPU memcpy (file_size ×
  1 / memcpy_bandwidth). On the current ~2 GB harness file with
  ~15 GB/s memory bandwidth, expect ~130 ms saved per load — the
  ~200 ms cluster should drop to ~70–80 ms (bounded by pageable H2D).
  `test_load_file_cpu` stays flat at ~1.4 ms: the `load_file` CPU path
  is already pointer-aliased to mmap, the bytearray copy only lives on
  the `safe_open.get_tensor` path. If the cluster does **not** split
  after PR 1, the bottleneck isn't the memcpy — re-profile before
  moving on.

**Risks:**
- Lifetime: if anything releases the `Arc<Mmap>` before the tensor is
  dropped, you get a use-after-free (would segfault on tensor access).
  The `setattr` pattern is the belt-and-suspenders fix.
- torch version skew: older torch might not accept all buffer-protocol
  objects for `frombuffer`. The existing tests exercise a wide range;
  lean on them.
- `test_threadable.py` stresses cross-thread tensor lifetimes —
  particularly important to keep passing.

---

## Safety notes to preserve across all PRs

1. **mmap slice escape.** The current code keeps mmap alive via the
   `storage` reference on the `Open` / `safe_open` handle. Any tensor
   that aliases mmap bytes must extend that lifetime (setattr pattern).
   Don't return raw `&[u8]` to Python land without a parent object.

2. **DLPack deleter ownership.** When we emit DLPack (PR 3+), the
   capsule's `deleter` callback is the **single** owner of the backing
   allocation. No secondary free paths. For CPU-DLPack backed by mmap,
   the deleter decrements the `Arc<Mmap>` refcount. For FFI-CUDA
   (PR 4+), the deleter calls `cuMemFree` or returns to our pool.

3. **FFI contained.** PR 4 adds `cuda.rs` with ~12 `extern "C"`
   declarations. Wrap all of them in safe Rust functions inside the
   module. No raw pointers in public types. `cuResult` return codes
   must be checked — never assume success.

4. **Primary-context attach, not create.** In PR 4, use
   `cuDevicePrimaryCtxRetain` + push/pop instead of `cuCtxCreate`.
   PyTorch initializes the runtime's primary context on first use;
   we need to run on that context, not compete with it.

5. **Bounds checks stay.** `Open::get_tensor` bounds-checks via the
   metadata lookup and data_offsets. Any new fast path must do the
   same — do not trust offsets from unvalidated sources.

---

## References

- Slow-path PR we're responding to:
  [safetensors/safetensors#736](https://github.com/safetensors/safetensors/pull/736)
- Transformers TP loader (reference implementation of what we're
  lifting into safetensors): `refactor-tp-loading` branch, file
  `src/transformers/core_model_loading.py`
- DLPack spec (v0 capsules, `"dltensor"` capsule name; v1 versioned
  capsules `"dltensor_versioned"` with stream field):
  https://dmlc.github.io/dlpack/latest/

---

## Glossary

- **H2D** — Host (CPU memory) to Device (GPU memory) copy.
- **Pinned memory** — page-locked host allocation. Required for
  `cudaMemcpyAsync` to actually run asynchronously and at full PCIe
  bandwidth.
- **DLPack** — C-ABI contract for zero-copy tensor handoff between
  frameworks. All modern tensor libraries (PyTorch, JAX, Paddle, CuPy,
  MLX) can import/export DLPack capsules.
- **mmap readahead** — kernel's speculative prefetch of pages ahead
  of the current read position. Demand-paged; kernel decides window
  size. `madvise(MADV_SEQUENTIAL)` makes it more aggressive. Still
  much slower than explicit large-block `pread` for bulk loads.
- **Symmetric memory / NVSHMEM** — PyTorch's one-sided NVLink pulls
  across ranks. Not our problem (lives in transformers); mentioned
  here only because the TP loader uses it for redistribute, which
  partly motivates the rank-partitioned-read architecture.
