# Buffered ceiling — the apples-to-apples target for the current pread design
#
# fio --name=buf --filename=/mnt/nvme/fio.test --size=64G --rw=read \
#     --bs=16m --direct=0 --ioengine=psync --numjobs=8 --group_reporting \
#     --time_based --runtime=30

# O_DIRECT ceiling — the headroom an io_uring engine could claim
#
# fio --name=dio --filename=/mnt/nvme/fio.test --size=64G --rw=read \
#     --bs=16m --direct=1 --ioengine=libaio --iodepth=32 --numjobs=8 \
#     --group_reporting --time_based --runtime=30

# Qwen3-30B-A3B-NVFP4 TP 1-4
# Qwen2.5-32B-Instruct TP 1-4
# Qwen2.5-14B-Instruct TP 1-4
# optional: Qwen3-30B-A3B

import ctypes
import mmap
import os
from pathlib import Path
import re
import signal
import subprocess
import sys
import time

import safetensors
from safetensors import safe_open
import torch

_PAGE = os.sysconf("SC_PAGE_SIZE")
_libc = ctypes.CDLL("libc.so.6", use_errno=True)

LOAD_RE = re.compile(r"Loading weights took ([\d.]+) seconds")


def resident_bytes(path: os.PathLike) -> int:
    size = os.path.getsize(path)
    if size == 0:
        return 0
    fd = os.open(path, os.O_RDONLY)
    try:
        file = mmap.mmap(fd, size, access=mmap.ACCESS_COPY)
    finally:
        os.close(fd)
    try:
        addr = ctypes.addressof(ctypes.c_char.from_buffer(file))
        npages = (size + _PAGE - 1) // _PAGE
        vec = (ctypes.c_ubyte * npages)()
        if _libc.mincore(ctypes.c_void_p(addr), ctypes.c_size_t(size), vec) != 0:
            raise OSError(ctypes.get_errno(), "mincore failed")
        return sum(v & 1 for v in vec) * _PAGE
    finally:
        file.close()


def evict(paths: list[Path], tolerance: float = 0.01) -> None:
    os.sync()
    for p in paths:
        fd = os.open(p, os.O_RDONLY)
        try:
            os.posix_fadvise(fd, 0, 0, os.POSIX_FADV_DONTNEED)
        finally:
            os.close(fd)
    total = sum(os.path.getsize(p) for p in paths)
    left = sum(resident_bytes(p) for p in paths)
    if left > total * tolerance:
        raise RuntimeError(
            f"eviction failed: {left / 2**30:.2f} GiB of {total / 2**30:.2f} GiB still cached"
        )


def prewarm(paths: list[str], tolerance: float = 0.99) -> None:
    for p in paths:
        with open(p, "rb", buffering=0) as f:
            while f.read(64 << 20):
                pass
    total = sum(os.path.getsize(p) for p in paths)
    got = sum(resident_bytes(p) for p in paths)
    if got < total * tolerance:
        raise RuntimeError(f"prewarm incomplete: {got / total:.1%} resident")


def env():
    env = os.environ.copy()
    venv = Path(sys.executable).parent.parent
    env["VIRTUAL_ENV"] = str(venv)
    env["PATH"] = f"{venv / 'bin'}:{env['PATH']}"
    env.pop("PYTHONHOME", None)
    env.pop("PYTHONPATH", None)
    env["VLLM_ENGINE_READY_TIMEOUT_S"] = "9000"
    return env


DONE_RE = re.compile(r"Model loading took .*and ([\d.]+) seconds")


def wait_for_load(
    logpath: str, proc: subprocess.Popen, timeout_s: float = 2400, interval: float = 0.5
) -> str:
    """Tail the server log until weight loading has completed."""
    t0 = time.monotonic()
    pos, buf = 0, ""
    while True:
        with open(logpath, "r", errors="replace") as f:
            f.seek(pos)
            buf += f.read()
            pos = f.tell()
        if DONE_RE.search(buf):
            return buf
        if proc.poll() is not None:
            raise RuntimeError(
                f"server exited with {proc.returncode} before loading finished\n"
                + "\n".join(buf.splitlines()[-25:])
            )
        if time.monotonic() - t0 > timeout_s:
            raise TimeoutError(f"weights not loaded after {timeout_s}s")
        time.sleep(interval)


def kill(proc: subprocess.Popen, grace: float = 15):
    if proc.poll() is not None:
        return
    os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
    try:
        proc.wait(timeout=grace)
    except subprocess.TimeoutExpired:
        os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
        proc.wait(timeout=60)


def run_vllm(cmd: list[str], port: int, paths: list) -> tuple[float, float]:
    logpath = str(Path.home() / "vllm.log")
    Path(logpath).write_text("")
    evict([str(p) for p in paths])
    with open(logpath, "w") as log:
        proc = subprocess.Popen(
            cmd,
            stdout=log,
            stderr=subprocess.STDOUT,
            env=env(),
            start_new_session=True,
        )
        t0 = time.monotonic()
        try:
            buf = wait_for_load(logpath, proc)
            wall = time.monotonic() - t0
        finally:
            kill(proc)
    hits = [float(x) for x in LOAD_RE.findall(buf)]
    if not hits:
        raise RuntimeError("no 'Loading weights took' line in log")
    return max(hits), wall


def print_setup(model: str, tp_size: int, loader: str):
    print("python  ", sys.version.split()[0])
    print(
        "torch   ",
        torch.__version__,
        torch.cuda.is_available(),
        torch.cuda.device_count(),
    )
    print("st      ", safetensors.__version__, safetensors.__file__)
    try:
        import vllm

        print("vllm    ", vllm.version.__version__, vllm.__file__)
    except Exception as e:
        print("vllm    ", e)

    print(f"running: {model}")
    print(f"tp_size: {tp_size}")
    print(f"loader: {loader}")


def vllm_cmd(
    vllm: os.PathLike, model: str, tp_size: int, loader: str, port: int
) -> list[str]:
    return [
        str(vllm),
        "serve",
        model,
        "--tensor-parallel-size",
        str(tp_size),
        "--load-format",
        loader,
        "--max-model-len",
        "4096",
        "--gpu-memory-utilization",
        "0.90",
        "--enforce-eager",
        "--port",
        str(port),
    ]


if __name__ == "__main__":
    VLLM = Path(sys.executable).with_name("vllm")
    assert VLLM.exists(), f"vllm not found in {sys.executable}"
    model = sys.argv[1] if len(sys.argv) > 1 else "/raid/luc/models/qwen3-30b-a3b-nvfp4"
    tp_size = int(sys.argv[2]) if len(sys.argv) > 2 else 1
    loader = sys.argv[3] if len(sys.argv) > 3 else "safetensors"
    port = int(sys.argv[4]) if len(sys.argv) > 4 else 8000
    print_setup(model, tp_size, loader)
    cmd = vllm_cmd(VLLM, model, tp_size, loader, port)
    loaded, ready = run_vllm(cmd, port, sorted(Path(model).glob("*.safetensors")))
    print(f"RESULT\t{Path(model).name}\t{tp_size}\t{loader}\t{loaded}\t{ready:.2f}")
