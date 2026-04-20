import os
from concurrent.futures import ThreadPoolExecutor

import pytest
import torch

from safetensors import safe_open
from safetensors.torch import load_file, save_file


cuda_required = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="requires cuda"
)


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


def _record_throughput(benchmark, file_size_mb):
    # pytest-benchmark doesn't derive throughput on its own; stash the inputs
    # so `--benchmark-columns` or post-processing can compute MB/s.
    benchmark.extra_info["file_size_mb"] = round(file_size_mb, 2)


def test_load_file_cpu(benchmark, llm_file, file_size_mb):
    benchmark(load_file, llm_file)
    _record_throughput(benchmark, file_size_mb)


@cuda_required
def test_load_file_cuda(benchmark, llm_file, file_size_mb):
    def _load():
        tensors = load_file(llm_file, device="cuda:0")
        torch.cuda.synchronize()
        return tensors

    benchmark(_load)
    _record_throughput(benchmark, file_size_mb)


@cuda_required
def test_safe_open_cuda_per_tensor(benchmark, llm_file, file_size_mb):
    def _load():
        out = {}
        with safe_open(llm_file, framework="pt", device="cuda:0") as sf:
            for name in sf.keys():
                out[name] = sf.get_tensor(name)
        torch.cuda.synchronize()
        return out

    benchmark(_load)
    _record_throughput(benchmark, file_size_mb)


@cuda_required
def test_safe_open_cpu_then_to_cuda(benchmark, llm_file, file_size_mb):
    def _load():
        out = {}
        with safe_open(llm_file, framework="pt", device="cpu") as sf:
            for name in sf.keys():
                out[name] = sf.get_tensor(name).to("cuda:0")
        torch.cuda.synchronize()
        return out

    benchmark(_load)
    _record_throughput(benchmark, file_size_mb)


@cuda_required
@pytest.mark.parametrize("n_workers", [1, 4, 8, 16])
def test_transformers_pattern_threaded(benchmark, llm_file, file_size_mb, n_workers):
    target_dtype = torch.bfloat16

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
    _record_throughput(benchmark, file_size_mb)
