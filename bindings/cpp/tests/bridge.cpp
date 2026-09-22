#include "rust/cxx.h"
#include "safetensors-cpp/src/lib.rs.h"

#include <array>
#include <cstdint>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <utility>

void require(bool condition) {
  if (!condition) {
    throw std::runtime_error("C++ bridge check failed");
  }
}

template <typename Function> void require_error(Function function) {
  try {
    function();
  } catch (const rust::Error &) {
    return;
  }
  throw std::runtime_error("expected a rust::Error");
}

int main() {
  try {
    const std::array<std::uint8_t, 2> bytes = {42, 43};
    const std::array<std::size_t, 1> dimensions = {2};
    const std::array<safetensors::Tensor, 1> tensors = {{
        {"values", "U8", {dimensions.data(), dimensions.size()},
         {bytes.data(), bytes.size()}}
    }};
    const rust::Slice<const safetensors::Tensor> input(tensors.data(), tensors.size());
    const std::array<safetensors::MetadataEntry, 1> metadata = {{
        {"origin", "bridge test"}
    }};
    auto buffer = safetensors::serialize(input, {metadata.data(), metadata.size()});
    require(buffer.size() >= bytes.size());
    const auto original_data = buffer.data() + buffer.size() - bytes.size();
    auto archive = safetensors::deserialize(std::move(buffer));
    require(archive->names().size() == 1);
    require(archive->names()[0] == "values");
    require(archive->dtype("values") == "U8");
    require(archive->shape("values").size() == 1);
    require(archive->shape("values")[0] == 2);
    const auto data = archive->data("values");
    require(data.size() == bytes.size());
    require(data.data() == original_data);
    require(data[0] == 42);
    require(data[1] == 43);
    const auto loaded_metadata = archive->metadata();
    require(loaded_metadata.size() == 1);
    require(loaded_metadata[0].key == "origin");
    require(loaded_metadata[0].value == "bridge test");

    require_error([&] { archive->data("missing"); });
    require_error([&] { archive->shape("missing"); });
    require_error([&] { archive->dtype("missing"); });
    require_error([] { safetensors::deserialize({}); });

    const std::array<std::size_t, 2> overflow_shape = {
        std::numeric_limits<std::size_t>::max(), 2};
    const std::array<safetensors::Tensor, 1> invalid = {{
        {"overflow", "U8", {overflow_shape.data(), overflow_shape.size()},
         {bytes.data(), bytes.size()}}
    }};
    require_error([&] { safetensors::serialize({invalid.data(), invalid.size()}, {}); });
    std::cout << "C++ bridge: round trip, ownership, metadata, and exceptions passed\n";
  } catch (const std::exception &error) {
    std::cerr << error.what() << '\n';
    return 1;
  }
}
