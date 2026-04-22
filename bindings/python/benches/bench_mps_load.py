from __future__ import annotations

import argparse
import contextlib
import os
import subprocess
import sys
import time
from pathlib import Path

import torch

from safetensors.torch import load_file, save_file


def create_llm(total_gb: float) -> dict[str, torch.Tensor]:
    H, I, V, BPE = 4096, 12288, 151936, 2
    Q, KV = 32 * 128, 8 * 128
    fixed = (2 * V * H + H) * BPE
    per_layer = (H * Q + 2 * H * KV + Q * H + 3 * H * I + 2 * H) * BPE
    n = max(1, int((total_gb * 1024 ** 3 - fixed) / per_layer))
    d = torch.bfloat16
    t: dict[str, torch.Tensor] = {
        "model.embed_tokens.weight": torch.empty((V, H), dtype=d),
        "model.norm.weight": torch.empty((H,), dtype=d),
        "lm_head.weight": torch.empty((V, H), dtype=d),
    }
    for i in range(n):
        p = f"model.layers.{i}"
        t[f"{p}.self_attn.q_proj.weight"] = torch.empty((Q, H), dtype=d)
        t[f"{p}.self_attn.k_proj.weight"] = torch.empty((KV, H), dtype=d)
        t[f"{p}.self_attn.v_proj.weight"] = torch.empty((KV, H), dtype=d)
        t[f"{p}.self_attn.o_proj.weight"] = torch.empty((H, Q), dtype=d)
        t[f"{p}.mlp.gate_proj.weight"] = torch.empty((I, H), dtype=d)
        t[f"{p}.mlp.up_proj.weight"] = torch.empty((I, H), dtype=d)
        t[f"{p}.mlp.down_proj.weight"] = torch.empty((H, I), dtype=d)
        t[f"{p}.input_layernorm.weight"] = torch.empty((H,), dtype=d)
        t[f"{p}.post_attention_layernorm.weight"] = torch.empty((H,), dtype=d)
    print(f"  {n} layers, {len(t)} tensors")
    return t


@contextlib.contextmanager
def force_slow():
    saved = torch.mps.__dict__.pop("_host_alias_storage", None)
    try:
        yield
    finally:
        if saved is not None:
            torch.mps._host_alias_storage = saved


def purge() -> bool:
    return subprocess.run(
        ["sudo", "-n", "purge"],
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=30,
    ).returncode == 0


def time_one(path: str, fast: bool) -> float:
    torch.mps.synchronize()
    t0 = time.perf_counter()
    ctx = contextlib.nullcontext() if fast else force_slow()
    with ctx:
        out = load_file(path, device="mps")
    torch.mps.synchronize()
    dt = time.perf_counter() - t0
    for v in list(out.values())[:3]:
        if v.numel():
            v.flatten()[:1].item()
    return dt


def bench(path: str, iters: int, cold: bool) -> None:
    n = os.path.getsize(path)
    print(f"\nFile: {path}  ({n / 1024 ** 3:.2f} GB)\nwarmup...")
    time_one(path, False); time_one(path, True)

    rows = []
    for label, fast in (("slow (safe_open)", False), ("fast (MPSBulkLoad)", True)):
        ts = []
        for i in range(iters):
            if cold and not purge():
                print("  ! sudo -n purge failed, results are warm-cache")
                cold = False
            dt = time_one(path, fast)
            ts.append(dt)
            print(f"  {label:20s} {i+1}/{iters}: {dt:6.3f}s  ({n/dt/1024**3:5.2f} GB/s)")
        rows.append((label, min(ts), sum(ts) / len(ts)))

    print(f"\n  {'path':22s} {'best':>8s} {'mean':>8s} {'GB/s':>8s}")
    for lbl, b, m in rows:
        print(f"  {lbl:22s} {b:6.3f}s {m:6.3f}s {n/b/1024**3:8.2f}")
    print(f"\n  speedup: {rows[0][1] / rows[1][1]:.2f}x")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--file", type=Path, required=True)
    p.add_argument("--gb", type=float, default=16.0)
    p.add_argument("--iters", type=int, default=3)
    p.add_argument("--cold", action="store_true")
    a = p.parse_args()

    if not torch.backends.mps.is_available():
        sys.exit("MPS not available")
    if not hasattr(torch.mps, "_host_alias_storage"):
        sys.exit("needs torch.mps._host_alias_storage (pytorch #180961)")

    path = str(a.file)
    if not os.path.exists(path):
        print(f"Generating {path} ({a.gb} GB target) ...")
        t0 = time.perf_counter()
        save_file(create_llm(a.gb), path)
        print(f"  wrote {os.path.getsize(path) / 1024 ** 3:.2f} GB "
              f"in {time.perf_counter() - t0:.1f}s")
    bench(path, a.iters, a.cold)


if __name__ == "__main__":
    main()
