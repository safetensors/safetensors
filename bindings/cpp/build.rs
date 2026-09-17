fn main() {
    cxx_build::bridge("src/lib.rs")
        .std("c++14")
        .compile("safetensors-cpp");
    println!("cargo:rerun-if-changed=src/lib.rs");
}
