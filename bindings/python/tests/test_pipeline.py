"""Tests for the prefetch pipeline (`pipeline` module in the Rust binding).

Exercises the opt-in fast path and its building blocks in layered order:

  1. `safe_open.prefetch(...)` public API (P1 trivial fallback).
  2. CUDA FFI wrappers + NUMA resolution via the debug probes
     `_debug_cuda_probe` / `_debug_cuda_ctx_smoke` (P2).

Tests skip gracefully when the necessary hardware/drivers aren't present:
- non-Linux: the whole pipeline module is cfg-gated out.
- no CUDA driver: `_debug_cuda_probe` returns `[{"error": "..."}]` and the
  CUDA-specific tests skip on that signal.
"""

import os
import re
import sys

import pytest

from safetensors import safe_open
from safetensors.torch import save_file


PLATFORM_SUPPORTED = sys.platform == "linux"

_debug_cuda_probe = None
_debug_cuda_ctx_smoke = None
if PLATFORM_SUPPORTED:
    from safetensors._safetensors_rust import (  # type: ignore[attr-defined]
        _debug_cuda_probe as _probe,
        _debug_cuda_ctx_smoke as _smoke,
    )

    _debug_cuda_probe = _probe
    _debug_cuda_ctx_smoke = _smoke


requires_linux = pytest.mark.skipif(
    not PLATFORM_SUPPORTED, reason="pipeline is linux-only"
)


def _cuda_info():
    """Returns list[dict] from the probe, or None if CUDA is unavailable."""
    if _debug_cuda_probe is None:
        return None
    probe = _debug_cuda_probe()
    if len(probe) == 1 and "error" in probe[0]:
        return None
    return probe


# ── P1: prefetch API shape ──────────────────────────────────────────────


@pytest.fixture
def tiny_file(tmp_path):
    import torch

    path = tmp_path / "tiny.safetensors"
    save_file(
        {
            "a": torch.zeros((4, 8)),
            "b": torch.ones((3, 3)),
            "c": torch.arange(16).reshape(4, 4).to(torch.float32),
        },
        str(path),
    )
    return str(path)


@requires_linux
def test_prefetch_present_on_linux(tiny_file):
    with safe_open(tiny_file, framework="pt", device="cpu") as sf:
        assert hasattr(sf, "prefetch"), "prefetch should be available on Linux"


@requires_linux
def test_prefetch_iterates_as_expected(tiny_file):
    with safe_open(tiny_file, framework="pt", device="cpu") as sf:
        h = sf.prefetch(["a", "b", "c"])
        names = {name for name, _ in h}
        assert names == {"a", "b", "c"}
        assert len(h) == 0  # drained


@requires_linux
def test_prefetch_drain_on_delivery(tiny_file):
    with safe_open(tiny_file, framework="pt", device="cpu") as sf:
        h = sf.prefetch(["a", "b", "c"])
        assert len(h) == 3
        # Pull one via iteration.
        first = next(iter(h))
        assert len(h) == 2
        assert first[0] in {"a", "b", "c"}


@requires_linux
def test_prefetch_iteration_is_single_use(tiny_file):
    with safe_open(tiny_file, framework="pt", device="cpu") as sf:
        h = sf.prefetch(["a", "b"])
        list(h)
        # Second pass yields nothing.
        assert list(h) == []


@requires_linux
def test_prefetch_dict_materialization(tiny_file):
    with safe_open(tiny_file, framework="pt", device="cpu") as sf:
        d = dict(sf.prefetch(["a", "b", "c"]))
        assert set(d) == {"a", "b", "c"}


@requires_linux
def test_prefetch_accepts_kwargs(tiny_file):
    # `dtype` and `max_inflight` are accepted and currently ignored; making
    # sure the signature matches so callers can already pass them.
    with safe_open(tiny_file, framework="pt", device="cpu") as sf:
        h = sf.prefetch(["a"], dtype=None, max_inflight=4)
        assert len(h) == 1


@requires_linux
def test_prefetch_no_get_or_wait_all(tiny_file):
    with safe_open(tiny_file, framework="pt", device="cpu") as sf:
        h = sf.prefetch(["a"])
        assert not hasattr(h, "get"), "API is iterator-only"
        assert not hasattr(h, "wait_all"), "API is iterator-only"


# ── P2: CUDA FFI wrappers + NUMA ─────────────────────────────────────────


@requires_linux
def test_debug_probe_returns_list():
    probe = _debug_cuda_probe()
    assert isinstance(probe, list)
    # Either at least one device, or a single {'error': '...'} sentinel.
    assert len(probe) >= 1


@requires_linux
def test_cuda_devices_enumerated():
    info = _cuda_info()
    if info is None:
        pytest.skip("libcuda.so.1 not loadable on this box")
    for dev in info:
        assert "ordinal" in dev
        assert "name" in dev
        assert "pci_bus_id" in dev
        assert "numa_node" in dev
        assert "numa_cpus" in dev
        # Driver returns uppercase; our wrapper lowercases.
        assert re.fullmatch(r"[0-9a-f]{4}:[0-9a-f]{2}:[0-9a-f]{2}\.[0-9a-f]", dev["pci_bus_id"]), dev["pci_bus_id"]


@requires_linux
def test_numa_mapping_matches_sysfs():
    """Cross-check the probe's numa_node against a direct sysfs read."""
    info = _cuda_info()
    if info is None:
        pytest.skip("libcuda.so.1 not loadable on this box")
    for dev in info:
        bdf = dev["pci_bus_id"]
        expected_path = f"/sys/bus/pci/devices/{bdf}/numa_node"
        if not os.path.exists(expected_path):
            # Some virtualized / unusual topologies don't expose this.
            continue
        with open(expected_path) as f:
            expected = int(f.read().strip())
        assert dev["numa_node"] == expected, (
            f"probe reports {dev['numa_node']} for {bdf}, sysfs says {expected}"
        )


@requires_linux
def test_numa_cpus_populated_when_node_known():
    info = _cuda_info()
    if info is None:
        pytest.skip("libcuda.so.1 not loadable on this box")
    for dev in info:
        if dev["numa_node"] >= 0:
            assert len(dev["numa_cpus"]) > 0, dev
            # CPUs should be non-negative integers.
            assert all(isinstance(c, int) and c >= 0 for c in dev["numa_cpus"])


@requires_linux
def test_cuda_ctx_smoke_roundtrip():
    """Retain primary context, push, alloc pinned+device, H2D, sync, release."""
    info = _cuda_info()
    if info is None:
        pytest.skip("libcuda.so.1 not loadable on this box")
    if not info:
        pytest.skip("no CUDA devices visible")
    bytes_copied = _debug_cuda_ctx_smoke(info[0]["ordinal"])
    assert bytes_copied == 4096


@requires_linux
def test_cuda_ctx_smoke_rejects_bad_ordinal():
    info = _cuda_info()
    if info is None:
        pytest.skip("libcuda.so.1 not loadable on this box")
    bad_ordinal = len(info) + 10
    with pytest.raises(Exception):
        _debug_cuda_ctx_smoke(bad_ordinal)
