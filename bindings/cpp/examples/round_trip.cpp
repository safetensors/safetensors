#include "safetensors-cpp/src/lib.rs.h"

#include <array>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <stdexcept>

int main(int argc, char **argv) {
  if (argc != 2) {
    std::cerr << "usage: round_trip output.safetensors\n";
    return 1;
  }
  try {
    // Safetensors stores little-endian, contiguous data. Encode 1.0f and 2.0f
    // explicitly here so the example also writes correctly on big-endian hosts.
    const std::array<std::uint8_t, 8> bytes = {0, 0, 128, 63, 0, 0, 0, 64};
    const std::array<std::size_t, 1> dimensions = {2};
    const std::array<safetensors::Tensor, 1> tensors = {{
        {"weights", "F32", {dimensions.data(), dimensions.size()},
         {bytes.data(), bytes.size()}}
    }};
    const std::array<safetensors::MetadataEntry, 1> metadata = {{
        {"source", "C++ example"}
    }};
    safetensors::save_file({tensors.data(), tensors.size()},
                          {metadata.data(), metadata.size()}, argv[1]);

    auto archive = safetensors::load_file(argv[1]);
    const auto data = archive->data("weights");
    const auto shape = archive->shape("weights");
    if (archive->dtype("weights") != "F32" || shape.size() != 1 ||
        shape[0] != 2 || data.size() != bytes.size() ||
        std::memcmp(data.data(), bytes.data(), bytes.size()) != 0) {
      throw std::runtime_error("round trip changed the tensor");
    }
    std::cout << "weights: dtype=F32 shape=[2] bytes=" << data.size() << '\n';
    // `data` and `shape` borrow from archive. Keep archive alive while using them.
  } catch (const std::exception &error) {
    std::cerr << error.what() << '\n';
    return 1;
  }
}
