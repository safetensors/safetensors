//! NUMA topology probing and thread pinning.
//!
//! Ports the 98-line helper from hmll commit `c67e7c0` to Rust. Reads the
//! GPU's NUMA node via sysfs (PCI BDF → `/sys/bus/pci/devices/<bdf>/numa_node`)
//! and pins the calling thread to the full cpulist of that node. The io_uring
//! SQPOLL kernel thread gets pinned separately via
//! `io_uring::Builder::setup_sqpoll_cpu` in the iouring module.
//!
//! TODO(P2): fill in once the CUDA FFI wrappers land — we need
//! `cuDeviceGetPCIBusId` from `cuda::CuDevice::pci_bus_id()` before we can
//! resolve the GPU's NUMA node. The sysfs parsing can be written and unit
//! tested independently; that's what lives here for now.

#![allow(dead_code)] // scaffolding; wired up in P2.

use crate::pipeline::error::{PipelineError, PipelineResult};

/// Read `/sys/bus/pci/devices/<bdf>/numa_node` for a PCI BDF. The BDF must be
/// in lowercase `DDDD:BB:DD.F` form (CUDA returns uppercase hex — callers
/// lowercase it before calling this).
pub fn numa_node_for_pci(bdf: &str) -> PipelineResult<i32> {
    let path = format!("/sys/bus/pci/devices/{bdf}/numa_node");
    let s = std::fs::read_to_string(&path)
        .map_err(|e| PipelineError::NumaProbe(format!("read {path}: {e}")))?;
    s.trim()
        .parse::<i32>()
        .map_err(|e| PipelineError::NumaProbe(format!("parse {path}: {e}")))
}

/// Parse `/sys/devices/system/node/node<N>/cpulist`, which uses the
/// `"0-23,48-71"` range-list syntax. Returns the expanded CPU id set.
pub fn cpulist_for_node(node: i32) -> PipelineResult<Vec<usize>> {
    if node < 0 {
        return Ok(Vec::new());
    }
    let path = format!("/sys/devices/system/node/node{node}/cpulist");
    let s = std::fs::read_to_string(&path)
        .map_err(|e| PipelineError::NumaProbe(format!("read {path}: {e}")))?;
    parse_cpulist(s.trim())
}

fn parse_cpulist(s: &str) -> PipelineResult<Vec<usize>> {
    let mut cpus = Vec::new();
    for tok in s.split(',') {
        let tok = tok.trim();
        if tok.is_empty() {
            continue;
        }
        if let Some((lo, hi)) = tok.split_once('-') {
            let lo: usize = lo
                .parse()
                .map_err(|e| PipelineError::NumaProbe(format!("cpulist range lo: {e}")))?;
            let hi: usize = hi
                .parse()
                .map_err(|e| PipelineError::NumaProbe(format!("cpulist range hi: {e}")))?;
            cpus.extend(lo..=hi);
        } else {
            let c: usize = tok
                .parse()
                .map_err(|e| PipelineError::NumaProbe(format!("cpulist int: {e}")))?;
            cpus.push(c);
        }
    }
    Ok(cpus)
}

/// Pin the current thread to the given CPU set via `sched_setaffinity`.
/// TODO(P2): expose + call from the DMA worker. Returns Ok(()) and does
/// nothing if `cpus` is empty.
pub fn pin_current_thread(cpus: &[usize]) -> PipelineResult<()> {
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
            return Err(PipelineError::NumaProbe(format!(
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
