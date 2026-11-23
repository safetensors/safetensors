# Roadmap

What `byte-mover` is for: one backend for moving bytes from files into memory
a caller owns, shared by safetensors and by inference engines, in place of the
readers each of them writes.

## Now

- Reads queued and reaped one by one, with an id and a result each.
- Two backends: threads that `pread`, and threads that drive an io_uring each,
  with registered memory where the locked-memory limit allows.

## Next

- Cancelling a read that has not started.
- Two priorities: reads needed now go before reads issued ahead.
- Direct I/O helpers: opening a file for it, falling back to buffered reads
  when the filesystem refuses (no filesystem detection), `F_NOCACHE` on macOS.
- Picking the backend and the reads in flight that suit the machine.
- io_uring: one registration shared by the rings, in place of one each.
- Staging: an aligned pool for destinations that are not aligned, and
  completions that wait on a copy still in flight (device memory).
- A CUDA destination: pinned staging and asynchronous copies, from the
  safetensors loader.

## Adoption

- safetensors' Python bindings load through it, in place of their own engine.
- A reader in safetensors on top of it: ranges named by tensor and slice, read
  into tensors the caller gives.

## Bindings

- A C interface.
- Python.

## Platforms

- Linux and macOS today.
- Windows.

## Release

- Its own version and tags (`byte-mover-v*`), published to crates.io by its
  own workflow.
