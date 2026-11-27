# C++ bindings

These bindings use [CXX](https://cxx.rs/) to call the existing Rust safetensors
parser and serializer from C++. They do not implement a separate file format.

## Build and run (Linux)

Install a stable Rust toolchain and a C++14 compiler, then run from this directory:

```sh
cargo build
c++ -std=c++14 -I target/cxxbridge examples/round_trip.cpp \
    target/debug/libsafetensors_cpp.a -ldl -lpthread -lm -o target/round_trip
./target/round_trip target/example.safetensors
```

Expected output:

```text
weights: dtype=F32 shape=[2] bytes=8
```

The example saves a tensor and metadata, reloads the file, and verifies its dtype,
shape, and bytes. Include `safetensors-cpp/src/lib.rs.h` from the generated
`target/cxxbridge` include directory and link `libsafetensors_cpp.a` in your own
application. Use `cargo build --release` and `target/release/` for optimized builds.
The CXX headers and library must come from the same build. Platform-specific system
link libraries can be obtained with `cargo rustc -- --print native-static-libs`.

## API and ownership

- `Tensor` borrows a name, format dtype string (`F32`, `BF16`, `I64`, etc.),
  dimensions, and bytes from the caller. All must remain valid and unchanged until
  `serialize` or `save_file` returns. Tensor data must already be contiguous in
  C order and little endian. No dtype conversion or device transfer is performed.
- `serialize` returns an owned `rust::Vec<std::uint8_t>` containing a complete
  safetensors file. `save_file` writes directly through the Rust file writer
  without allocating a second whole-file buffer.
- `deserialize` takes ownership of a `rust::Vec<std::uint8_t>` without copying its
  data; pass it with `std::move`. `load_file` reads the complete file into memory.
  Neither function memory-maps files or loads tensors onto a GPU.
- Both loaders return a `rust::Box<Archive>`. The archive validates the file once
  and owns the file bytes. `data(name)` and `shape(name)` return read-only slices
  borrowing from it: **do not use those slices after the archive is destroyed**.
  There is no per-tensor data copy when retrieving a slice.
- Tensor bytes have no guaranteed alignment for a C++ numeric type. Copy them
  with `std::memcpy` into suitably aligned storage rather than dereferencing a
  cast pointer. Handle byte order on big-endian hosts.
- Shapes count logical elements, including packed dtypes such as `F4`. Supply the
  corresponding packed bytes, not one byte per logical element.
- `metadata()` returns string pairs. Passing an empty metadata list when saving
  omits the optional metadata map. Duplicate tensor names, the reserved tensor
  name `__metadata__`, and duplicate metadata keys are rejected.
- Rust validation and I/O errors cross the bridge as `rust::Error`, derived from
  `std::exception`. Include `rust/cxx.h` to catch `rust::Error`, or catch
  `std::exception` as the example does.
- Filenames are UTF-8 strings.

## Tests

```sh
cargo test
cargo fmt --check
cargo clippy --all-targets -- -D warnings
cargo build
c++ -std=c++14 -Wall -Wextra -Werror -I target/cxxbridge tests/bridge.cpp \
    target/debug/libsafetensors_cpp.a -ldl -lpthread -lm -o target/bridge_test
./target/bridge_test
```

The Rust tests exercise the format boundary and errors. The standalone C++ test
also verifies the generated API, ownership transfer, and Rust-to-C++ exceptions.
