//! Cross-checks NUMA probe results against direct sysfs reads.
//!
//! Skips gracefully when CUDA is absent — without a device ordinal we
//! can't resolve the GPU's NUMA node, but the sysfs-only path
//! (`numa_node_for_pci`, `cpulist_for_node`) is still exercised below.

use ionic_rs::{cuda, numa};

#[test]
fn numa_node_matches_sysfs() {
    let count = match cuda::CuDevice::count() {
        Ok(n) if n > 0 => n,
        _ => {
            eprintln!("skipping: libcuda.so.1 not loadable or no devices");
            return;
        }
    };
    for ordinal in 0..count {
        let dev = cuda::CuDevice::get(ordinal).expect("CuDevice::get");
        let bdf = dev.pci_bus_id().expect("pci_bus_id");
        let path = format!("/sys/bus/pci/devices/{bdf}/numa_node");
        let Ok(s) = std::fs::read_to_string(&path) else {
            // Some virtualized topologies don't expose this; not a probe bug.
            continue;
        };
        let expected: i32 = s.trim().parse().expect("parse numa_node");
        let probed = numa::numa_node_for_pci(&bdf).expect("numa_node_for_pci");
        assert_eq!(probed, expected, "probe vs sysfs for {bdf}");
    }
}

#[test]
fn cpulist_for_node_zero_populated_if_present() {
    // node0 always exists on a real Linux box.
    let path = "/sys/devices/system/node/node0/cpulist";
    if !std::path::Path::new(path).exists() {
        eprintln!("skipping: {path} absent");
        return;
    }
    let cpus = numa::cpulist_for_node(0).expect("cpulist_for_node");
    assert!(!cpus.is_empty(), "node0 cpulist should not be empty");
    assert!(cpus.iter().all(|&c| c < 4096), "absurd CPU id: {cpus:?}");
}

#[test]
fn cpulist_for_negative_node_returns_empty() {
    let cpus = numa::cpulist_for_node(-1).expect("cpulist_for_node(-1)");
    assert!(cpus.is_empty());
}
