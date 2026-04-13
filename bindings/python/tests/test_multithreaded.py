import os
import sys
import tempfile
import threading

import numpy as np
import torch

from safetensors.numpy import load_file as load_file_np
from safetensors.numpy import save_file as save_file_np
from safetensors.torch import load_file as load_file_pt
from safetensors.torch import save_file as save_file_pt


NUM_THREADS = 4
NUM_ITERATIONS = 10


def run_thread_pool(save_worker, barrier, tensors):
    try:
        # the default thread switch interval is 5 milliseconds
        orig_switch = sys.getswitchinterval()
        sys.setswitchinterval(0.000001)  # in seconds

        tasks = [
            threading.Thread(target=save_worker, args=(tensors, barrier))
            for _ in range(NUM_THREADS)
        ]
        [t.start() for t in tasks]
        [t.join() for t in tasks]
    finally:
        sys.setswitchinterval(orig_switch)
        # just in case one of the threads never started, to avoid a deadlock
        barrier.abort()


def test_multithreaded_roundtripping_numpy():
    def save_worker(tensors, barrier):
        barrier.wait()
        for _ in range(NUM_ITERATIONS):
            # NamedTemporaryFile can't be reopened on Windows (ERROR_SHARING_VIOLATION
            # due to delete-on-close semantics), so use a TemporaryDirectory + path join.
            with tempfile.TemporaryDirectory() as tmpdir:
                path = os.path.join(tmpdir, "test.safetensors")
                save_file_np(tensors, path)
                loaded_tensors = load_file_np(path)
                for name, tensor in tensors.items():
                    assert np.all(loaded_tensors[name] == tensor)

    tensors = {
        "1": np.random.randn(5, 25),
        "2": np.random.randn(876, 768, 2),
        "3": np.ones(5000),
        "4": np.array(5000.0),
        "5": np.array(768, dtype=np.int32),
    }

    run_thread_pool(save_worker, threading.Barrier(NUM_THREADS), tensors)


def test_multithreaded_roundtripping_torch():
    def save_worker(tensors, barrier):
        barrier.wait()
        for _ in range(NUM_ITERATIONS):
            with tempfile.TemporaryDirectory() as tmpdir:
                path = os.path.join(tmpdir, "test.safetensors")
                save_file_pt(tensors, path)
                loaded_tensors = load_file_pt(path)
                for name, tensor in tensors.items():
                    assert torch.all(loaded_tensors[name] == tensor)

    tensors = {
        "1": torch.randn(5, 25),
        "2": torch.randn(876, 768, 2),
        "3": torch.ones(5000),
        "4": torch.tensor(5000.0),
        "5": torch.tensor(768, dtype=torch.int32),
    }

    run_thread_pool(save_worker, threading.Barrier(NUM_THREADS), tensors)
