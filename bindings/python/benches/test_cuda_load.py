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


@pytest.fixture(scope="module", autouse=True)
def _cuda_warmup():
    # Prevents the first benchmark round from absorbing CUDA context init cost.
    if torch.cuda.is_available():
        x = torch.zeros(1024, device="cuda:0")
        torch.cuda.synchronize()
        del x


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
def cache_state(request, llm_file):
    if request.param == "cold" and not _FADVISE_AVAILABLE:
        pytest.skip("cold-cache variant requires Linux posix_fadvise")
    if request.param == "warm":
        _warm_page_cache(llm_file)
    else:
        _evict_page_cache(llm_file)
    return request.param


def _record_throughput(benchmark, file_size_mb, cache_state=None):
    benchmark.extra_info["file_size_mb"] = round(file_size_mb, 2)
    if cache_state is not None:
        benchmark.extra_info["cache_state"] = cache_state


def _run_parametrized(benchmark, llm_file, file_size_mb, cache_state, fn):
    # Per-round setup evicts before each cold round; warm rounds rely on the
    # module-scope warm-up in the fixture. Pedantic gives us explicit control
    # over rounds/iterations so the eviction cost doesn't bleed into timings.
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
