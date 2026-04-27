//! NUMA topology probing and thread pinning.
//!
//! Ports the 98-line helper from hmll commit `c67e7c0` to Rust. Reads the
//! GPU's NUMA node via sysfs (PCI BDF → `/sys/bus/pci/devices/<bdf>/numa_node`)
//! and pins the calling thread to the full cpulist of that node. The io_uring
//! SQPOLL kernel thread gets pinned separately via
//! `io_uring::Builder::setup_sqpoll_cpu` in the iouring module.

use crate::error::{Error, Result};

/// Read `/sys/bus/pci/devices/<bdf>/numa_node` for a PCI BDF. The BDF must be
/// in lowercase `DDDD:BB:DD.F` form (CUDA returns uppercase hex — callers
/// lowercase it before calling this).
pub fn numa_node_for_pci(bdf: &str) -> Result<i32> {
    let path = format!("/sys/bus/pci/devices/{bdf}/numa_node");
    let s = std::fs::read_to_string(&path)
        .map_err(|e| Error::NumaProbe(format!("read {path}: {e}")))?;
    s.trim()
        .parse::<i32>()
        .map_err(|e| Error::NumaProbe(format!("parse {path}: {e}")))
}

/// Parse `/sys/devices/system/node/node<N>/cpulist`, which uses the
/// `"0-23,48-71"` range-list syntax. Returns the expanded CPU id set.
pub fn cpulist_for_node(node: i32) -> Result<Vec<usize>> {
    if node < 0 {
        return Ok(Vec::new());
    }
    let path = format!("/sys/devices/system/node/node{node}/cpulist");
    let s = std::fs::read_to_string(&path)
        .map_err(|e| Error::NumaProbe(format!("read {path}: {e}")))?;
    parse_cpulist(s.trim())
}

fn parse_cpulist(s: &str) -> Result<Vec<usize>> {
    let mut cpus = Vec::new();
    for tok in s.split(',') {
        let tok = tok.trim();
        if tok.is_empty() {
            continue;
        }
        if let Some((lo, hi)) = tok.split_once('-') {
            let lo: usize = lo
                .parse()
                .map_err(|e| Error::NumaProbe(format!("cpulist range lo: {e}")))?;
            let hi: usize = hi
                .parse()
                .map_err(|e| Error::NumaProbe(format!("cpulist range hi: {e}")))?;
            cpus.extend(lo..=hi);
        } else {
            let c: usize = tok
                .parse()
                .map_err(|e| Error::NumaProbe(format!("cpulist int: {e}")))?;
            cpus.push(c);
        }
    }
    Ok(cpus)
}

/// Resolve the NUMA node for a CUDA device ordinal. Returns -1 if the node
/// is unknown (virtualized GPUs, unusual topologies) or if CUDA isn't
/// available — both are non-fatal; callers degrade to not pinning.
pub fn numa_node_for_device(device_ordinal: i32) -> Result<i32> {
    let dev = crate::cuda::CuDevice::get(device_ordinal)?;
    let bdf = dev.pci_bus_id()?;
    numa_node_for_pci(&bdf)
}

/// Pin the calling thread to all CPUs on the GPU's NUMA node. Returns the
/// resolved node id on success, or `Ok(-1)` when the node is unknown (in
/// which case nothing was pinned — caller sees a best-effort result, not
/// an error).
pub fn bind_to_gpu_node(device_ordinal: i32) -> Result<i32> {
    let node = numa_node_for_device(device_ordinal)?;
    if node < 0 {
        return Ok(-1);
    }
    let cpus = cpulist_for_node(node)?;
    pin_current_thread(&cpus)?;
    Ok(node)
}

/// Pin the current thread to the given CPU set via `sched_setaffinity`.
/// Returns Ok(()) and does nothing if `cpus` is empty.
pub fn pin_current_thread(cpus: &[usize]) -> Result<()> {
    if cpus.is_empty() {
        return Ok(());
    }
    // SAFETY: zero-initialization of cpu_set_t is valid; CPU_SET / CPU_ZERO
    // are defined for any index in [0, CPU_SETSIZE).
    unsafe {
        let mut set: libc::cpu_set_t = std::mem::zeroed();
        libc::CPU_ZERO(&mut set);
        for &c in cpus {
            if c < libc::CPU_SETSIZE as usize {
                libc::CPU_SET(c, &mut set);
            }
        }
        let rc = libc::sched_setaffinity(0, std::mem::size_of::<libc::cpu_set_t>(), &set);
        if rc != 0 {
            let errno = *libc::__errno_location();
            return Err(Error::NumaProbe(format!(
                "sched_setaffinity errno={errno}"
            )));
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_range_list() {
        let cpus = parse_cpulist("0-3,8-9,12").unwrap();
        assert_eq!(cpus, vec![0, 1, 2, 3, 8, 9, 12]);
    }

    #[test]
    fn parses_empty() {
        let cpus = parse_cpulist("").unwrap();
        assert!(cpus.is_empty());
    }

    #[test]
    fn parses_single() {
        let cpus = parse_cpulist("5").unwrap();
        assert_eq!(cpus, vec![5]);
    }
}
