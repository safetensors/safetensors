# byte-mover

Reads byte ranges of files into memory the caller owns.

Reads are queued on a `Mover`, run by a pool of threads that `pread` or drive
an io_uring each, or by an io_uring the calling threads drive, and reaped one
by one, each with the id it was queued under.
Any thread may queue and reap. A mover knows nothing of what the bytes are: file formats, caches and
devices are the caller's.

It is the backend of safetensors' reader, for programs that page model weights
in and out while they run (the experts of a mixture-of-experts model, rows of a
large table), where the time a read takes to come back matters as much as the
bytes per second.

```rust
use byte_mover::{Mover, Read};
use std::os::fd::AsRawFd;

let file = std::fs::File::open("model.safetensors")?;
let mut record = vec![0u8; 4096];
let mover = Mover::threads(8)?;
let read = Read {
    id: 7,
    fd: file.as_raw_fd(),
    offset: 1 << 20,
    dst: record.as_mut_ptr(),
    len: record.len(),
};
// SAFETY: `record` and `file` outlive the read, which is reaped below
unsafe { mover.submit(read) };
let done = mover.wait_done().expect("one read is pending");
assert_eq!(done.id, 7);
done.result?;
```

## Backends

- `Mover::threads(n)`: `pread`s on `n` threads. Works on every Unix.
- `Mover::rings(n, depth)`: `n` io_urings, a thread each, with up to `depth`
  reads in the kernel each, on Linux, behind the `io-uring` feature (on by
  default). `register` gives the rings memory to read into without pinning
  pages at each read. It fails where io_uring is absent or forbidden, as under
  the default seccomp profile of most container runtimes: fall back to threads.
- `Mover::ring(depth)`: one io_uring the calling threads drive, with no thread
  of its own: a read starts when it is queued and its completion wakes the
  thread that waits for it. For few reads in flight, where a hand-off between
  threads costs more than the read.

## Status

Internal to safetensors: it is not published and its API changes without
notice. It is a crate of its own so that it builds, tests and benchmarks
without Python.
