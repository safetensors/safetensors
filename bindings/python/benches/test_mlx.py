import os
import platform
import tempfile

import pytest


if platform.system() == "Darwin":
    import mlx.core as mx
    from safetensors.mlx import load_file, save_file

    def create_gpt2(n_layers: int):
        tensors = {}
        tensors["wte"] = mx.zeros((50257, 768))
        tensors["wpe"] = mx.zeros((1024, 768))
        for i in range(n_layers):
            tensors[f"h.{i}.ln_1.weight"] = mx.zeros((768,))
            tensors[f"h.{i}.ln_1.bias"] = mx.zeros((768,))
            tensors[f"h.{i}.attn.bias"] = mx.zeros((1, 1, 1024, 1024))
            tensors[f"h.{i}.attn.c_attn.weight"] = mx.zeros((768, 2304))
            tensors[f"h.{i}.attn.c_attn.bias"] = mx.zeros((2304))
            tensors[f"h.{i}.attn.c_proj.weight"] = mx.zeros((768, 768))
            tensors[f"h.{i}.attn.c_proj.bias"] = mx.zeros((768))
            tensors[f"h.{i}.ln_2.weight"] = mx.zeros((768))
            tensors[f"h.{i}.ln_2.bias"] = mx.zeros((768))
            tensors[f"h.{i}.mlp.c_fc.weight"] = mx.zeros((768, 3072))
            tensors[f"h.{i}.mlp.c_fc.bias"] = mx.zeros((3072))
            tensors[f"h.{i}.mlp.c_proj.weight"] = mx.zeros((3072, 768))
            tensors[f"h.{i}.mlp.c_proj.bias"] = mx.zeros((768))
        tensors["ln_f.weight"] = mx.zeros((768))
        tensors["ln_f.bias"] = mx.zeros((768))
        return tensors

    def load(filename):
        return mx.load(filename)

    def test_mlx_mlx_load(benchmark):
        # benchmark something
        weights = create_gpt2(12)
        with tempfile.NamedTemporaryFile(delete=False) as f:
            filename = f"{f.name}.npz"
            mx.savez(filename, **weights)
            result = benchmark(load, filename)
        os.unlink(f.name)

        for k, v in weights.items():
            tv = result[k]
            assert mx.allclose(v, tv)

    def test_mlx_sf_load(benchmark):
        # benchmark something
        weights = create_gpt2(12)
        with tempfile.NamedTemporaryFile(delete=False) as f:
            save_file(weights, f.name)
            result = benchmark(load_file, f.name)
        os.unlink(f.name)

        for k, v in weights.items():
            tv = result[k]
            assert mx.allclose(v, tv)

    def _metal_available() -> bool:
        return mx.metal.is_available()

    @pytest.mark.skipif(not _metal_available(), reason="requires metal")
    def test_mlx_sf_load_mps(benchmark):
        # benchmark something
        weights = create_gpt2(12)
        with tempfile.NamedTemporaryFile(delete=False) as f:
            save_file(weights, f.name)
            result = benchmark(load_file, f.name, device="mps")
        os.unlink(f.name)

        for k, v in weights.items():
            tv = result[k]
            assert mx.allclose(v, tv)

    @pytest.mark.skipif(not _metal_available(), reason="requires metal")
    def test_mlx_sf_load_mps_pread(benchmark):
        # On Apple-silicon MPS, get_tensors() allocates Shared MTLBuffers and
        # parallel-preads into them, then hands off to MLX via DLPack. The
        # `backend` kwarg selects how bytes are sourced (pread here).
        weights = create_gpt2(12)
        with tempfile.NamedTemporaryFile(delete=False) as f:
            save_file(weights, f.name)
            result = benchmark(load_file, f.name, device="mps", backend="pread")
        os.unlink(f.name)

        for k, v in weights.items():
            tv = result[k]
            assert mx.allclose(v, tv)
