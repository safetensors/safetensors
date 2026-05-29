"""Llama-shaped safetensors fixture generator (stdlib-only).

Existing test coverage in this repository exercises empty-header round-trips
(``{}``) and very small synthetic tensors. Real-model deployments hit headers
that hold hundreds of entries and tensor data running to gigabytes; bugs that
only surface at that scale (header parser limits, offset arithmetic, layered
dtype mixes) are not currently covered by the suite.

This module emits a Llama-3.2-shaped safetensors file using only the Python
standard library. It depends on neither ``torch`` nor ``safetensors`` itself,
which keeps it usable for any test that wants a real-shape input regardless of
which framework extra is installed. The structural shape (embedding +
N transformer layers with attention/MLP/norms + final norm + LM head) is
parameterised so the same generator covers both a small-default fixture
suitable for unit tests and a larger configuration if a downstream test needs
one.

The output follows the safetensors binary contract exactly:

* bytes ``[0, 8)``: little-endian ``u64`` header length ``N``
* bytes ``[8, 8 + N)``: UTF-8 JSON header
* bytes ``[8 + N, EOF)``: contiguous tensor data, dtype-typed and aligned per
  the writer's emission order

The default shape produces 201 tensors (embedding + 22 layers x 9 tensors per
layer + final norm + LM head) matching the structural inventory of
TinyLlama-1.1B / Llama-3.2-1B-Instruct, scaled to a small fixture footprint.
Determinism is controlled by an explicit ``seed`` argument routed through
``random.Random``; the same seed produces byte-identical output, which is what
regression tests want.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import struct
import sys
from dataclasses import dataclass, field
from typing import Dict, List, Tuple


# Element-size table covering the dtype families exercised by Llama-shaped
# weights. Bit widths come from the safetensors README dtype table; this map
# only needs the byte widths because the generator writes raw bytes.
DTYPE_BYTES: Dict[str, int] = {
    "BOOL": 1,
    "U8": 1,
    "I8": 1,
    "F16": 2,
    "BF16": 2,
    "I16": 2,
    "U16": 2,
    "F32": 4,
    "I32": 4,
    "U32": 4,
    "F64": 8,
    "I64": 8,
    "U64": 8,
}


@dataclass(frozen=True)
class LlamaShape:
    """Structural parameters for a Llama-style architecture.

    The defaults match a downscaled TinyLlama / Llama-3.2 layout: 22 transformer
    blocks with grouped-query attention (``num_kv_heads`` < ``num_heads``),
    SwiGLU-style MLP (gate + up + down), and pre-/post-attention RMSNorms.
    The tensor count produced by these defaults is 201, identical to the
    inventory of the TinyLlama-1.1B canonical safetensors file.
    """

    vocab_size: int = 32000
    hidden_size: int = 128
    num_hidden_layers: int = 22
    num_attention_heads: int = 4
    num_kv_heads: int = 2
    head_dim: int = 32
    intermediate_size: int = 256
    weight_dtype: str = "F32"
    norm_dtype: str = "F32"

    @property
    def attn_proj_dim(self) -> int:
        return self.num_attention_heads * self.head_dim

    @property
    def kv_proj_dim(self) -> int:
        return self.num_kv_heads * self.head_dim


@dataclass
class TensorEntry:
    name: str
    dtype: str
    shape: Tuple[int, ...]
    data_offsets: Tuple[int, int] = field(default=(0, 0))

    def nbytes(self) -> int:
        nel = 1
        for d in self.shape:
            nel *= d
        return nel * DTYPE_BYTES[self.dtype]


def _add(
    entries: List[TensorEntry],
    offset: int,
    name: str,
    shape: Tuple[int, ...],
    dtype: str,
) -> int:
    """Append a tensor entry rooted at ``offset`` and return the next offset."""
    e = TensorEntry(name=name, dtype=dtype, shape=tuple(shape))
    nbytes = e.nbytes()
    e.data_offsets = (offset, offset + nbytes)
    entries.append(e)
    return offset + nbytes


def build_entries(shape: LlamaShape) -> List[TensorEntry]:
    """Return the full ordered list of tensor entries for ``shape``.

    The emission order mirrors what HuggingFace ``transformers`` writes for a
    LlamaForCausalLM checkpoint: token embedding, then per-layer
    {q,k,v,o, gate,up,down, input_norm, post_attn_norm}, then final norm and
    LM head. Keeping the order stable matters for byte-identical regression
    tests.
    """
    entries: List[TensorEntry] = []
    offset = 0
    wdtype = shape.weight_dtype
    ndtype = shape.norm_dtype

    offset = _add(
        entries,
        offset,
        "model.embed_tokens.weight",
        (shape.vocab_size, shape.hidden_size),
        wdtype,
    )

    for layer in range(shape.num_hidden_layers):
        p = f"model.layers.{layer}"
        offset = _add(
            entries,
            offset,
            f"{p}.self_attn.q_proj.weight",
            (shape.attn_proj_dim, shape.hidden_size),
            wdtype,
        )
        offset = _add(
            entries,
            offset,
            f"{p}.self_attn.k_proj.weight",
            (shape.kv_proj_dim, shape.hidden_size),
            wdtype,
        )
        offset = _add(
            entries,
            offset,
            f"{p}.self_attn.v_proj.weight",
            (shape.kv_proj_dim, shape.hidden_size),
            wdtype,
        )
        offset = _add(
            entries,
            offset,
            f"{p}.self_attn.o_proj.weight",
            (shape.hidden_size, shape.attn_proj_dim),
            wdtype,
        )
        offset = _add(
            entries,
            offset,
            f"{p}.mlp.gate_proj.weight",
            (shape.intermediate_size, shape.hidden_size),
            wdtype,
        )
        offset = _add(
            entries,
            offset,
            f"{p}.mlp.up_proj.weight",
            (shape.intermediate_size, shape.hidden_size),
            wdtype,
        )
        offset = _add(
            entries,
            offset,
            f"{p}.mlp.down_proj.weight",
            (shape.hidden_size, shape.intermediate_size),
            wdtype,
        )
        offset = _add(
            entries,
            offset,
            f"{p}.input_layernorm.weight",
            (shape.hidden_size,),
            ndtype,
        )
        offset = _add(
            entries,
            offset,
            f"{p}.post_attention_layernorm.weight",
            (shape.hidden_size,),
            ndtype,
        )

    offset = _add(
        entries,
        offset,
        "model.norm.weight",
        (shape.hidden_size,),
        ndtype,
    )
    offset = _add(
        entries,
        offset,
        "lm_head.weight",
        (shape.vocab_size, shape.hidden_size),
        wdtype,
    )

    return entries


def build_header(entries: List[TensorEntry], shape: LlamaShape) -> Tuple[bytes, int]:
    """Serialise the JSON header. Returns ``(header_bytes, total_data_bytes)``.

    The header is a single ``json.dumps`` pass with ``sort_keys=False`` so the
    emission order matches what the generator built; this is the same property
    the reference Rust writer maintains.
    """
    header: Dict[str, object] = {}
    for e in entries:
        header[e.name] = {
            "dtype": e.dtype,
            "shape": list(e.shape),
            "data_offsets": list(e.data_offsets),
        }
    header["__metadata__"] = {
        "format": "pt",
        "model_type": "llama",
        "vocab_size": str(shape.vocab_size),
        "hidden_size": str(shape.hidden_size),
        "num_hidden_layers": str(shape.num_hidden_layers),
        "num_attention_heads": str(shape.num_attention_heads),
        "num_kv_heads": str(shape.num_kv_heads),
        "head_dim": str(shape.head_dim),
        "intermediate_size": str(shape.intermediate_size),
        "generator": "safetensors.tests.fixtures.llama_generator",
    }
    header_json = json.dumps(header).encode("utf-8")
    total_data = entries[-1].data_offsets[1] if entries else 0
    return header_json, total_data


def _seeded_bytes(rng: random.Random, n: int) -> bytes:
    """Return ``n`` deterministic bytes drawn from ``rng``.

    ``random.Random.randbytes`` is the canonical stdlib path on Python 3.9+ and
    produces byte-identical output across runs for a fixed seed, which is the
    only property the fixture needs.
    """
    return rng.randbytes(n)


def write_fixture(
    path: str,
    shape: LlamaShape = LlamaShape(),
    seed: int = 0,
    zero_fill: bool = False,
) -> Dict[str, int]:
    """Write a Llama-shaped safetensors file to ``path``.

    Parameters
    ----------
    path:
        Output filesystem path. Parent directory is created if missing.
    shape:
        Architectural parameters. Defaults produce 201 tensors.
    seed:
        Deterministic RNG seed. Identical ``seed`` + ``shape`` + ``zero_fill``
        produces byte-identical output.
    zero_fill:
        When true, the data segment is all-zero. Useful for very large fixtures
        where the data bytes themselves are not under test.

    Returns
    -------
    A small summary dict with ``n_tensors``, ``header_bytes``, ``data_bytes``,
    and ``total_bytes`` for logging or assertion.
    """
    entries = build_entries(shape)
    header_json, total_data = build_header(entries, shape)

    parent = os.path.dirname(path)
    if parent:
        os.makedirs(parent, exist_ok=True)

    rng = random.Random(seed)
    with open(path, "wb") as f:
        f.write(struct.pack("<Q", len(header_json)))
        f.write(header_json)
        if zero_fill:
            f.write(b"\x00" * total_data)
        else:
            # Write in chunks so multi-gigabyte configurations do not load the
            # entire data segment into memory before flushing.
            chunk = 1 << 20
            remaining = total_data
            while remaining > 0:
                n = min(chunk, remaining)
                f.write(_seeded_bytes(rng, n))
                remaining -= n

    return {
        "n_tensors": sum(1 for e in entries),
        "header_bytes": len(header_json),
        "data_bytes": total_data,
        "total_bytes": 8 + len(header_json) + total_data,
    }


def _parse_args(argv: List[str]) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Emit a Llama-shaped safetensors fixture using stdlib only. "
            "Default shape produces 201 tensors matching TinyLlama-1.1B / "
            "Llama-3.2-1B layout, scaled to a small fixture footprint."
        )
    )
    p.add_argument(
        "--out",
        default="llama_fixture.safetensors",
        help="output path (default: %(default)s)",
    )
    p.add_argument(
        "--seed",
        type=int,
        default=0,
        help="deterministic RNG seed (default: %(default)s)",
    )
    p.add_argument(
        "--zero-fill",
        action="store_true",
        help="fill data segment with zeros instead of seeded bytes",
    )
    p.add_argument("--vocab-size", type=int, default=LlamaShape.vocab_size)
    p.add_argument("--hidden-size", type=int, default=LlamaShape.hidden_size)
    p.add_argument("--num-layers", type=int, default=LlamaShape.num_hidden_layers)
    p.add_argument("--num-heads", type=int, default=LlamaShape.num_attention_heads)
    p.add_argument("--num-kv-heads", type=int, default=LlamaShape.num_kv_heads)
    p.add_argument("--head-dim", type=int, default=LlamaShape.head_dim)
    p.add_argument(
        "--intermediate-size", type=int, default=LlamaShape.intermediate_size
    )
    p.add_argument(
        "--weight-dtype",
        default=LlamaShape.weight_dtype,
        choices=sorted(DTYPE_BYTES.keys()),
    )
    p.add_argument(
        "--norm-dtype",
        default=LlamaShape.norm_dtype,
        choices=sorted(DTYPE_BYTES.keys()),
    )
    return p.parse_args(argv)


def main(argv: List[str]) -> int:
    args = _parse_args(argv)
    shape = LlamaShape(
        vocab_size=args.vocab_size,
        hidden_size=args.hidden_size,
        num_hidden_layers=args.num_layers,
        num_attention_heads=args.num_heads,
        num_kv_heads=args.num_kv_heads,
        head_dim=args.head_dim,
        intermediate_size=args.intermediate_size,
        weight_dtype=args.weight_dtype,
        norm_dtype=args.norm_dtype,
    )
    summary = write_fixture(
        args.out, shape=shape, seed=args.seed, zero_fill=args.zero_fill
    )
    for k, v in summary.items():
        print(f"{k:14s} {v}", file=sys.stderr)
    print(args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
