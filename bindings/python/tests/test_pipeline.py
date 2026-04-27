"""Python-side tests for the prefetch pipeline.

Covers only what is genuinely Python-API behavior — `PrefetchHandle`
iteration, drain semantics, `dict()` materialization, `len()`. Engine
behavior (io_uring, CUDA wrappers, NUMA) lives in pure Rust tests in
`ionic-rs/tests/` and runs via plain `cargo test`.
"""

import sys

import pytest

from safetensors import safe_open
from safetensors.torch import save_file


PLATFORM_SUPPORTED = sys.platform == "linux"

requires_linux = pytest.mark.skipif(
    not PLATFORM_SUPPORTED, reason="pipeline is linux-only"
)


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
        first = next(iter(h))
        assert len(h) == 2
        assert first[0] in {"a", "b", "c"}


@requires_linux
def test_prefetch_iteration_is_single_use(tiny_file):
    with safe_open(tiny_file, framework="pt", device="cpu") as sf:
        h = sf.prefetch(["a", "b"])
        list(h)
        assert list(h) == []  # drained


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
