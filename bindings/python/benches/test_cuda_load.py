"""Baseline benchmarks for CUDA loading paths.

Establishes throughput numbers for the ways callers currently move
safetensors weights onto a CUDA device. Downstream perf PRs (bytearray-copy
removal, `prefetch()` API, pinned staging, FFI CUDA, DLPack output, pread
backend) should be measured against the numbers collected here.

Paths tracked:

  1. `load_file(device="cuda:0")`
     Whole-file bulk load through the public helper.

  2. `safe_open(device="cuda:0") -> get_tensor`
     Per-tensor direct-to-CUDA path. Known slow (reported at ~85 MB/s in
     safetensors/safetensors#736): bytearray copy from mmap + pageable H2D
     on the default stream.

  3. `safe_open(device="cpu") -> get_tensor -> .to("cuda:0")`
     Two-step load. The non-TP path in transformers' core_model_loading.

  4. `safe_open(device="cpu") -> get_slice -> [...] -> .to(dtype) -> .to(cuda)`
     Exact pattern used by transformers' TP loader under a ThreadPoolExecutor
     (see `_schedule_reads` in `core_model_loading.py`). Parametrized over
     worker count so we can see the concurrency curve.

The non-threaded tests run in both warm- and cold-cache modes. Cold cache is
established without root via `os.sync()` + `posix_fadvise(POSIX_FADV_DONTNEED)`
per round. That reproduces the disk-I/O-bound regime the PR #736 report
observed; warm-cache numbers are memcpy + pageable-H2D bound, which is what
the pipeline optimization targets directly.
"""

import os
from concurrent.futures import ThreadPoolExecutor

import pytest
import torch

from safetensors import safe_open
from safetensors.torch import load_file, save_file


cuda_required = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="requires cuda"
)

_FADVISE_AVAILABLE = hasattr(os, "posix_fadvise") and hasattr(os, "POSIX_FADV_DONTNEED")


def _make_llm_weights(
    n_layers: int = 4,
    hidden: int = 4096,
    intermediate: int = 11008,
    vocab: int = 32000,
    dtype: torch.dtype = torch.bfloat16,
):
    """Llama-ish shape. Defaults produce ~2 GB in bf16."""
    tensors = {"model.embed_tokens.weight": torch.zeros((vocab, hidden), dtype=dtype)}
    for i in range(n_layers):
        p = f"model.layers.{i}"
        tensors[f"{p}.input_layernorm.weight"] = torch.zeros((hidden,), dtype=dtype)
        tensors[f"{p}.self_attn.q_proj.weight"] = torch.zeros(
            (hidden, hidden), dtype=dtype
        )
        tensors[f"{p}.self_attn.k_proj.weight"] = torch.zeros(
            (hidden, hidden), dtype=dtype
        )
        tensors[f"{p}.self_attn.v_proj.weight"] = torch.zeros(
            (hidden, hidden), dtype=dtype
        )
        tensors[f"{p}.self_attn.o_proj.weight"] = torch.zeros(
            (hidden, hidden), dtype=dtype
        )
        tensors[f"{p}.post_attention_layernorm.weight"] = torch.zeros(
            (hidden,), dtype=dtype
        )
        tensors[f"{p}.mlp.gate_proj.weight"] = torch.zeros(
            (intermediate, hidden), dtype=dtype
        )
        tensors[f"{p}.mlp.up_proj.weight"] = torch.zeros(
            (intermediate, hidden), dtype=dtype
        )
        tensors[f"{p}.mlp.down_proj.weight"] = torch.zeros(
            (hidden, intermediate), dtype=dtype
        )
    tensors["model.norm.weight"] = torch.zeros((hidden,), dtype=dtype)
    tensors["lm_head.weight"] = torch.zeros((vocab, hidden), dtype=dtype)
    return tensors


@pytest.fixture(scope="module")
def llm_file(tmp_path_factory):
    path = tmp_path_factory.mktemp("sf_bench_cuda") / "llm.safetensors"
    save_file(_make_llm_weights(), str(path))
    # Flush so the file's pages are clean and eligible for POSIX_FADV_DONTNEED.
    if hasattr(os, "sync"):
        os.sync()
    return str(path)


@pytest.fixture(scope="module")
def file_size_mb(llm_file):
    return os.path.getsize(llm_file) / (1024 * 1024)


@pytest.fixture(scope="module")
def real_model_dir():
    """Directory containing a real HF model checkpoint (multi-source
    `*.safetensors` files). Set via REAL_MODEL_DIR env var; tests that
    need it `skip` if it's unset or empty."""
    p = os.environ.get("REAL_MODEL_DIR")
    if not p or not os.path.isdir(p):
        return None
    return p


@pytest.fixture(scope="module")
def real_sources(real_model_dir):
    """List of all `*.safetensors` source-file paths in `real_model_dir`,
    sorted so runs are reproducible. Falls through to None if the model
    dir isn't set; tests skip in that case."""
    if real_model_dir is None:
        return None
    paths = sorted(
        os.path.join(real_model_dir, f)
        for f in os.listdir(real_model_dir)
        if f.endswith(".safetensors")
    )
    if not paths:
        return None
    if hasattr(os, "sync"):
        os.sync()
    return paths


@pytest.fixture(scope="module")
def real_total_size_mb(real_sources):
    if real_sources is None:
        return 0.0
    return sum(os.path.getsize(p) for p in real_sources) / (1024 * 1024)


@pytest.fixture(scope="module", autouse=True)
def _cuda_warmup():
    # Prevents the first benchmark round from absorbing CUDA context init cost.
    if torch.cuda.is_available():
        x = torch.zeros(1024, device="cuda:0")
        torch.cuda.synchronize()
        del x


@pytest.fixture(autouse=True)
def _gpu_cleanup_between_tests():
    """Force a full GPU/Python cleanup between bench tests. Real-model
    tests allocate tens of GB per round; without an explicit barrier
    pytest moves to the next test before the previous one's `cuMemFree_v2`
    calls have all settled, and cumulative pressure on an 80 GB card OOMs
    the cold-cache-runs-last variants. `gc.collect` drains any cycles
    holding torch tensors; `torch.cuda.synchronize` waits for in-flight
    work; `empty_cache` returns torch's allocator pool to the driver.
    """
    yield
    import gc
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.synchronize()
        torch.cuda.empty_cache()


def _evict_page_cache(path: str) -> None:
    # os.sync() flushes dirty pages so POSIX_FADV_DONTNEED can drop them.
    # Without the sync, dirty pages silently stay resident.
    os.sync()
    fd = os.open(path, os.O_RDONLY)
    try:
        os.posix_fadvise(fd, 0, 0, os.POSIX_FADV_DONTNEED)
    finally:
        os.close(fd)


def _warm_page_cache(path: str) -> None:
    with open(path, "rb") as f:
        while f.read(4 * 1024 * 1024):
            pass


@pytest.fixture(params=["warm", "cold"])
def cache_state(request):
    """Pure parametrize on the cache mode — tests do their own setup. We
    used to also warm/evict `llm_file` here, but that pulled in the 2 GB
    synthetic-file fixture for real-model tests that don't need it.
    """
    if request.param == "cold" and not _FADVISE_AVAILABLE:
        pytest.skip("cold-cache variant requires Linux posix_fadvise")
    return request.param


def _record_throughput(benchmark, file_size_mb, cache_state=None):
    benchmark.extra_info["file_size_mb"] = round(file_size_mb, 2)
    if cache_state is not None:
        benchmark.extra_info["cache_state"] = cache_state


def _run_parametrized(benchmark, llm_file, file_size_mb, cache_state, fn):
    # cache_state is now just the mode string — set up state here so
    # synthetic-file tests still get the per-round cold eviction.
    _setup_cache([llm_file], cache_state)
    setup = (lambda: _evict_page_cache(llm_file)) if cache_state == "cold" else None
    benchmark.pedantic(fn, setup=setup, rounds=5, iterations=1, warmup_rounds=0)
    _record_throughput(benchmark, file_size_mb, cache_state)


def test_load_file_cpu(benchmark, llm_file, file_size_mb, cache_state):
    _run_parametrized(
        benchmark, llm_file, file_size_mb, cache_state, lambda: load_file(llm_file)
    )


@cuda_required
def test_load_file_cuda(benchmark, llm_file, file_size_mb, cache_state):
    def _load():
        tensors = load_file(llm_file, device="cuda:0")
        torch.cuda.synchronize()
        return tensors

    _run_parametrized(benchmark, llm_file, file_size_mb, cache_state, _load)


@cuda_required
def test_safe_open_cuda_per_tensor(benchmark, llm_file, file_size_mb, cache_state):
    def _load():
        out = {}
        with safe_open(llm_file, framework="pt", device="cuda:0") as sf:
            for name in sf.keys():
                out[name] = sf.get_tensor(name)
        torch.cuda.synchronize()
        return out

    _run_parametrized(benchmark, llm_file, file_size_mb, cache_state, _load)


@cuda_required
def test_safe_open_cpu_then_to_cuda(benchmark, llm_file, file_size_mb, cache_state):
    def _load():
        out = {}
        with safe_open(llm_file, framework="pt", device="cpu") as sf:
            for name in sf.keys():
                out[name] = sf.get_tensor(name).to("cuda:0")
        torch.cuda.synchronize()
        return out

    _run_parametrized(benchmark, llm_file, file_size_mb, cache_state, _load)


@cuda_required
def test_safe_open_prefetch(benchmark, llm_file, file_size_mb, cache_state):
    # P4 fast path: ionic-rs CUDA pipeline (io_uring + dedicated stream).
    # No `torch.cuda.synchronize()` here — the pipeline ends `run()` with
    # `cuStreamSynchronize` on both producer streams, which is a host-side
    # barrier promoting the H2Ds to a happens-before edge any subsequent
    # CUDA work can rely on. The other cells (asarray/to-cuda paths) need
    # an outer sync because their async copies don't barrier internally;
    # this one doesn't.
    def _load():
        out = {}
        with safe_open(llm_file, framework="pt", device="cuda:0") as sf:
            for name, tensor in sf.prefetch(sf.keys()):
                out[name] = tensor
        return out

    _run_parametrized(benchmark, llm_file, file_size_mb, cache_state, _load)


def _setup_cache(paths, mode):
    """Set page-cache state for all source files: 'warm' reads them
    through, 'cold' drops their pages via fadvise."""
    if mode == "warm":
        for p in paths:
            _warm_page_cache(p)
    else:
        for p in paths:
            _evict_page_cache(p)


def _load_sources_threaded(paths, max_workers, use_prefetch):
    """Threadpool-per-source load: one safe_open + prefetch (or
    get_tensor) per source, fanning out across `max_workers` threads.
    This is what transformers' TP loader does. Each thread spins up its
    own pipeline — N threads = N io_uring rings + N DMA workers
    competing for the GPU's two physical copy engines.

    Returns the tensor count, not the dict, so pedantic's per-round
    `result` slot doesn't pin gigabytes across rounds (52 GB on an 80 GB
    card OOMs immediately if a previous round's tensors haven't dropped).
    Clearing the dict inside the timed window is ~one cuMemFree_v2 per
    tensor — microseconds each, sub-ms total.
    """
    def _one_source(path):
        out = {}
        with safe_open(path, framework="pt", device="cuda:0") as sf:
            if use_prefetch:
                for name, tensor in sf.prefetch(sf.keys()):
                    out[name] = tensor
            else:
                for name in sf.keys():
                    out[name] = sf.get_tensor(name)
        return out

    out = {}
    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        for chunk in pool.map(_one_source, paths):
            out.update(chunk)
    # The prefetch path syncs its own streams in `pipeline.run()`; only
    # the get_tensor variant needs an outer barrier here (its async H2Ds
    # ride torch's default stream and don't internally synchronize).
    if not use_prefetch:
        torch.cuda.synchronize()
    n = len(out)
    out.clear()
    return n


def _load_sources_unified(paths):
    """Single-pipeline multi-source load: one safe_open over the full
    source list, one prefetch call across the union of keys. Goes through
    the new ionic-rs CudaPipeline path with all sources registered on
    one io_uring ring — what ionic does.
    """
    with safe_open(paths, framework="pt", device="cuda:0") as sf:
        out = dict(sf.prefetch(sf.keys()))
    # No outer sync — pipeline.run() ended with cuStreamSynchronize on
    # both producer streams, which is a host-side barrier ensuring all
    # H2Ds are globally visible before the iterator yields.
    n = len(out)
    out.clear()
    return n


@cuda_required
@pytest.mark.parametrize(
    "variant", ["unified_prefetch", "threaded_prefetch", "threaded_get_tensor"]
)
def test_real_model_multisource(
    benchmark, real_sources, real_total_size_mb, variant, cache_state
):
    """Multi-source load of a real HF checkpoint, three variants:

    - `unified_prefetch`: single `safe_open(paths)` + single
      `prefetch()`. One ionic-rs pipeline registers every source file,
      reads + DMAs across them on one ring. The production target.
    - `threaded_prefetch`: per-source `safe_open` + `prefetch`, fanned
      out via a thread pool. N pipelines, each owning its own ring + DMA
      worker. The intermediate state we measured earlier.
    - `threaded_get_tensor`: per-source `safe_open` + per-tensor
      `get_tensor`, threaded. The pre-pipeline baseline (what
      transformers' TP loader does today).

    Skipped if REAL_MODEL_DIR isn't set.
    """
    if real_sources is None:
        pytest.skip("set REAL_MODEL_DIR to a directory with *.safetensors source files")

    paths = real_sources
    n_sources = len(paths)

    _setup_cache(paths, cache_state)
    setup = (lambda: _setup_cache(paths, "cold")) if cache_state == "cold" else None

    if variant == "unified_prefetch":
        fn = lambda: _load_sources_unified(paths)
    elif variant == "threaded_prefetch":
        fn = lambda: _load_sources_threaded(paths, n_sources, use_prefetch=True)
    else:  # threaded_get_tensor
        fn = lambda: _load_sources_threaded(paths, n_sources, use_prefetch=False)

    benchmark.pedantic(fn, setup=setup, rounds=3, iterations=1, warmup_rounds=0)
    _record_throughput(benchmark, real_total_size_mb, cache_state)
    benchmark.extra_info["n_sources"] = n_sources
    benchmark.extra_info["variant"] = variant


@cuda_required
@pytest.mark.parametrize("n_workers", [1, 4, 8, 16])
def test_transformers_pattern_threaded(benchmark, llm_file, file_size_mb, n_workers):
    # Worker-count scaling curve; warm-cache only to keep the parametrize matrix
    # small. Cold-cache adds a factor orthogonal to what this test measures.
    target_dtype = torch.bfloat16
    _warm_page_cache(llm_file)

    def _job(ts, dtype):
        return ts[...].to(dtype=dtype)

    def _load():
        out = {}
        with safe_open(llm_file, framework="pt", device="cpu") as sf:
            names = list(sf.keys())
            slices = [(n, sf.get_slice(n)) for n in names]
            with ThreadPoolExecutor(max_workers=n_workers) as pool:
                futures = [(n, pool.submit(_job, s, target_dtype)) for n, s in slices]
                for name, fut in futures:
                    out[name] = fut.result().to("cuda:0")
        torch.cuda.synchronize()
        return out

    benchmark(_load)
    _record_throughput(benchmark, file_size_mb, cache_state="warm")
