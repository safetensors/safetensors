import os
import tempfile

import pytest
import torch

from safetensors.torch import load_file, save_file


def create_gpt2(n_layers: int):
    tensors = {}
    tensors["wte"] = torch.zeros((50257, 768))
    tensors["wpe"] = torch.zeros((1024, 768))
    for i in range(n_layers):
        tensors[f"h.{i}.ln_1.weight"] = torch.zeros((768,))
        tensors[f"h.{i}.ln_1.bias"] = torch.zeros((768,))
        tensors[f"h.{i}.attn.bias"] = torch.zeros((1, 1, 1024, 1024))
        tensors[f"h.{i}.attn.c_attn.weight"] = torch.zeros((768, 2304))
        tensors[f"h.{i}.attn.c_attn.bias"] = torch.zeros((2304))
        tensors[f"h.{i}.attn.c_proj.weight"] = torch.zeros((768, 768))
        tensors[f"h.{i}.attn.c_proj.bias"] = torch.zeros((768))
        tensors[f"h.{i}.ln_2.weight"] = torch.zeros((768))
        tensors[f"h.{i}.ln_2.bias"] = torch.zeros((768))
        tensors[f"h.{i}.mlp.c_fc.weight"] = torch.zeros((768, 3072))
        tensors[f"h.{i}.mlp.c_fc.bias"] = torch.zeros((3072))
        tensors[f"h.{i}.mlp.c_proj.weight"] = torch.zeros((3072, 768))
        tensors[f"h.{i}.mlp.c_proj.bias"] = torch.zeros((768))
    tensors["ln_f.weight"] = torch.zeros((768))
    tensors["ln_f.bias"] = torch.zeros((768))
    return tensors


def create_lora(n_layers: int):
    tensors = {}
    for i in range(n_layers):
        tensors[f"lora.{i}.up.weight"] = torch.zeros((32, 32))
        tensors[f"lora.{i}.down.weight"] = torch.zeros((32, 32))
    return tensors


def test_pt_pt_load_cpu(benchmark):
    # benchmark something
    weights = create_gpt2(12)
    with tempfile.NamedTemporaryFile(delete=False) as f:
        torch.save(weights, f)
        result = benchmark(torch.load, f.name)
    os.unlink(f.name)

    for k, v in weights.items():
        tv = result[k]
        assert torch.allclose(v, tv)


def test_pt_sf_load_cpu(benchmark):
    # benchmark something
    weights = create_gpt2(12)
    with tempfile.NamedTemporaryFile(delete=False) as f:
        save_file(weights, f.name)
        result = benchmark(load_file, f.name)
    os.unlink(f.name)

    for k, v in weights.items():
        tv = result[k]
        assert torch.allclose(v, tv)


def test_pt_sf_load_cpu_pread(benchmark):
    weights = create_gpt2(12)
    with tempfile.NamedTemporaryFile(delete=False) as f:
        save_file(weights, f.name)
        result = benchmark(load_file, f.name, backend="pread")
    os.unlink(f.name)

    for k, v in weights.items():
        tv = result[k]
        assert torch.allclose(v, tv)


def test_pt_pt_load_cpu_small(benchmark):
    weights = create_lora(500)
    with tempfile.NamedTemporaryFile(delete=False) as f:
        torch.save(weights, f)
        result = benchmark(torch.load, f.name)
    os.unlink(f.name)

    for k, v in weights.items():
        tv = result[k]
        assert torch.allclose(v, tv)


def test_pt_sf_load_cpu_small(benchmark):
    weights = create_lora(500)

    with tempfile.NamedTemporaryFile(delete=False) as f:
        save_file(weights, f.name)
        result = benchmark(load_file, f.name)
    os.unlink(f.name)

    for k, v in weights.items():
        tv = result[k]
        assert torch.allclose(v, tv)


def test_pt_sf_load_cpu_small_pread(benchmark):
    weights = create_lora(500)

    with tempfile.NamedTemporaryFile(delete=False) as f:
        save_file(weights, f.name)
        result = benchmark(load_file, f.name, backend="pread")
    os.unlink(f.name)

    for k, v in weights.items():
        tv = result[k]
        assert torch.allclose(v, tv)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires cuda")
def test_pt_pt_load_gpu(benchmark):
    # benchmark something
    weights = create_gpt2(12)
    with tempfile.NamedTemporaryFile(delete=False) as f:
        torch.save(weights, f)
        result = benchmark(torch.load, f.name, map_location="cuda:0")
    os.unlink(f.name)

    for k, v in weights.items():
        v = v.cuda()
        tv = result[k]
        assert torch.allclose(v, tv)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires cuda")
def test_pt_sf_load_gpu(benchmark):
    # benchmark something
    weights = create_gpt2(12)
    with tempfile.NamedTemporaryFile(delete=False) as f:
        save_file(weights, f.name)
        result = benchmark(load_file, f.name, device="cuda:0")
    os.unlink(f.name)

    for k, v in weights.items():
        v = v.cuda()
        tv = result[k]
        assert torch.allclose(v, tv)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires cuda")
def test_pt_sf_load_gpu_pread(benchmark):
    weights = create_gpt2(12)
    with tempfile.NamedTemporaryFile(delete=False) as f:
        save_file(weights, f.name)
        result = benchmark(load_file, f.name, device="cuda:0", backend="pread")
    os.unlink(f.name)

    for k, v in weights.items():
        v = v.cuda()
        tv = result[k]
        assert torch.allclose(v, tv)


@pytest.mark.skipif(
    not hasattr(torch.backends, "mps") or not torch.backends.mps.is_available(),
    reason="requires mps",
)
def test_pt_pt_load_mps(benchmark):
    # benchmark something
    weights = create_gpt2(12)
    with tempfile.NamedTemporaryFile(delete=False) as f:
        torch.save(weights, f)
        result = benchmark(torch.load, f.name, map_location="mps")
    os.unlink(f.name)

    for k, v in weights.items():
        v = v.to(device="mps")
        tv = result[k]
        assert torch.allclose(v, tv)


@pytest.mark.skipif(
    not hasattr(torch.backends, "mps") or not torch.backends.mps.is_available(),
    reason="requires mps",
)
def test_pt_sf_load_mps(benchmark):
    # benchmark something
    weights = create_gpt2(12)
    with tempfile.NamedTemporaryFile(delete=False) as f:
        save_file(weights, f.name)
        result = benchmark(load_file, f.name, device="mps")
    os.unlink(f.name)

    for k, v in weights.items():
        v = v.to(device="mps")
        tv = result[k]
        assert torch.allclose(v, tv)


@pytest.mark.skipif(
    not hasattr(torch.backends, "mps") or not torch.backends.mps.is_available(),
    reason="requires mps",
)
def test_pt_sf_load_mps_pread(benchmark):
    # On Apple-silicon MPS, get_tensors() allocates Shared MTLBuffers and
    # parallel-preads into them, then hands off via DLPack. The `backend`
    # kwarg selects how bytes are sourced (pread here).
    weights = create_gpt2(12)
    with tempfile.NamedTemporaryFile(delete=False) as f:
        save_file(weights, f.name)
        result = benchmark(load_file, f.name, device="mps", backend="pread")
    os.unlink(f.name)

    for k, v in weights.items():
        v = v.to(device="mps")
        tv = result[k]
        assert torch.allclose(v, tv)


def test_pt_sf_save_cpu(benchmark):
    weights = create_gpt2(12)

    filename = "tmp.safetensors"

    # XXX: On some platforms (tested on Linux x86_64 ext4), writing to an already existing file is slower than creating a new one.
    # On others, such as MacOS (APFS), it's the opposite. To have more consistent benchmarks,
    # we ensure the file does not exist before each write, which is also closer to real world usage.
    def setup():
        try:
            os.unlink(filename)
        except Exception:
            pass

    benchmark.pedantic(
        save_file, args=(weights, filename), setup=setup, iterations=1, rounds=5
    )

    # Clean up files
    os.unlink(filename)
