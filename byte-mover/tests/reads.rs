#![cfg(unix)]

use std::{
    fs::File,
    io::ErrorKind,
    os::fd::AsRawFd,
    path::PathBuf,
    sync::atomic::{AtomicUsize, Ordering},
};

use byte_mover::{Mover, Read};

/// `len` pseudo-random bytes in a new file, removed when dropped
struct Fixture {
    path: PathBuf,
    file: File,
    data: Vec<u8>,
}

impl Fixture {
    fn new(len: usize) -> Self {
        static COUNT: AtomicUsize = AtomicUsize::new(0);
        let mut state = 0x9E37_79B9_7F4A_7C15u64;
        let data: Vec<u8> = (0..len)
            .map(|_| {
                state = state
                    .wrapping_mul(6364136223846793005)
                    .wrapping_add(1442695040888963407);
                (state >> 56) as u8
            })
            .collect();
        let name = format!(
            "byte_mover_{}_{}",
            std::process::id(),
            COUNT.fetch_add(1, Ordering::Relaxed)
        );
        let path = std::env::temp_dir().join(name);
        std::fs::write(&path, &data).unwrap();
        let file = File::open(&path).unwrap();
        Self { path, file, data }
    }
}

impl Drop for Fixture {
    fn drop(&mut self) {
        let _ = std::fs::remove_file(&self.path);
    }
}

/// The movers this build has; an io_uring the kernel refuses is skipped.
fn movers() -> Vec<(&'static str, Mover)> {
    #[cfg_attr(not(all(target_os = "linux", feature = "io-uring")), allow(unused_mut))]
    let mut movers = vec![("threads", Mover::threads(4).unwrap())];
    #[cfg(all(target_os = "linux", feature = "io-uring"))]
    match Mover::uring(8) {
        Ok(mover) => movers.push(("uring", mover)),
        Err(err) => eprintln!("skipped io_uring: {err}"),
    }
    movers
}

#[test]
fn ranges_land_in_their_destinations() {
    let fixture = Fixture::new(3 * 1024 * 1024 + 123);
    // more reads than the ring is deep, of odd offsets and lengths
    let ranges: Vec<(usize, usize)> = (0..40)
        .map(|i| (i * 70_001 % fixture.data.len(), 1 + i * 9_973 % 200_000))
        .map(|(offset, len)| (offset, len.min(fixture.data.len() - offset)))
        .chain([(0, 0), (fixture.data.len() - 1, 1)])
        .collect();
    for (name, mover) in movers() {
        let mut buffers: Vec<Vec<u8>> = ranges.iter().map(|&(_, len)| vec![0; len]).collect();
        for (id, (&(offset, len), buffer)) in ranges.iter().zip(&mut buffers).enumerate() {
            let read = Read {
                id: id as u64,
                fd: fixture.file.as_raw_fd(),
                offset: offset as u64,
                dst: buffer.as_mut_ptr(),
                len,
            };
            // SAFETY: the buffers and the file outlive the reads, reaped below
            unsafe { mover.submit(read).unwrap() };
        }
        mover.flush().unwrap();
        assert_eq!(mover.pending(), ranges.len(), "{name}");
        let mut seen = vec![false; ranges.len()];
        while let Some(done) = mover.wait_done() {
            done.result.unwrap();
            assert!(
                !std::mem::replace(&mut seen[done.id as usize], true),
                "{name}"
            );
        }
        assert!(seen.iter().all(|&s| s), "{name}");
        assert!(mover.try_done().is_none(), "{name}");
        for (&(offset, len), buffer) in ranges.iter().zip(&buffers) {
            assert_eq!(buffer, &fixture.data[offset..offset + len], "{name}");
        }
    }
}

#[test]
fn a_read_past_the_end_fails_alone() {
    let fixture = Fixture::new(10_000);
    for (name, mover) in movers() {
        let (mut whole, mut past) = (vec![0u8; 4_000], vec![0u8; 4_000]);
        for (id, offset, buffer) in [(1, 0, &mut whole), (2, 8_000, &mut past)] {
            let read = Read {
                id,
                fd: fixture.file.as_raw_fd(),
                offset,
                dst: buffer.as_mut_ptr(),
                len: buffer.len(),
            };
            // SAFETY: the buffers and the file outlive the reads, reaped below
            unsafe { mover.submit(read).unwrap() };
        }
        let mut results = [mover.wait_done().unwrap(), mover.wait_done().unwrap()];
        results.sort_by_key(|done| done.id);
        assert!(results[0].result.is_ok(), "{name}");
        let err = results[1].result.as_ref().unwrap_err();
        assert_eq!(err.kind(), ErrorKind::UnexpectedEof, "{name}");
        assert_eq!(whole, fixture.data[..4_000], "{name}");
        assert_eq!(past[..2_000], fixture.data[8_000..], "{name}");
        assert!(mover.wait_done().is_none(), "{name}");
    }
}

#[test]
fn dropping_waits_for_the_reads() {
    let fixture = Fixture::new(1 << 20);
    for (name, mover) in movers() {
        let mut buffer = vec![0u8; fixture.data.len()];
        let read = Read {
            id: 0,
            fd: fixture.file.as_raw_fd(),
            offset: 0,
            dst: buffer.as_mut_ptr(),
            len: buffer.len(),
        };
        // SAFETY: the buffer and the file outlive the mover
        unsafe { mover.submit(read).unwrap() };
        drop(mover);
        assert_eq!(buffer, fixture.data, "{name}");
    }
}

#[test]
fn threads_queue_while_another_reaps() {
    const THREADS: usize = 4;
    const READS: usize = 50;
    const LEN: usize = 10_000;
    let fixture = Fixture::new(THREADS * READS * LEN);
    for (name, mover) in movers() {
        let mut buffer = vec![0u8; fixture.data.len()];
        let base = buffer.as_mut_ptr() as usize;
        let fd = fixture.file.as_raw_fd();
        let reaped = std::thread::scope(|scope| {
            for thread in 0..THREADS {
                let mover = &mover;
                scope.spawn(move || {
                    for i in 0..READS {
                        let at = (thread * READS + i) * LEN;
                        let read = Read {
                            id: (at / LEN) as u64,
                            fd,
                            offset: at as u64,
                            dst: (base + at) as *mut u8,
                            len: LEN,
                        };
                        // SAFETY: each read has its own piece of `buffer`, which
                        // outlives the mover, as the file does
                        unsafe { mover.submit(read).unwrap() };
                        mover.flush().unwrap();
                    }
                });
            }
            let mut seen = [false; THREADS * READS];
            let mut reaped = 0;
            while reaped < seen.len() {
                // `None` when the queuing threads are behind
                let Some(done) = mover.wait_done() else {
                    std::thread::yield_now();
                    continue;
                };
                done.result.unwrap();
                assert!(!std::mem::replace(&mut seen[done.id as usize], true));
                reaped += 1;
            }
            reaped
        });
        assert_eq!(reaped, THREADS * READS, "{name}");
        assert_eq!(mover.pending(), 0, "{name}");
        drop(mover);
        assert_eq!(buffer, fixture.data, "{name}");
    }
}

#[cfg(all(target_os = "linux", feature = "io-uring"))]
#[test]
fn reads_into_registered_memory() {
    let Ok(mut mover) = Mover::uring(8) else {
        eprintln!("skipped: no io_uring");
        return;
    };
    let fixture = Fixture::new(256 * 1024);
    let mut region = vec![0u8; 1 << 20];
    // SAFETY: `region` outlives the mover
    if let Err(err) = unsafe { mover.register(region.as_mut_ptr(), region.len()) } {
        eprintln!("skipped: cannot register memory: {err}");
        return;
    }
    // two reads into the registered region, one into memory that is not
    let mut outside = vec![0u8; 1_000];
    let inside = region.as_mut_ptr().wrapping_add(500_000);
    let reads = [
        (0, 0, region.as_mut_ptr(), 100_000),
        (1, 100_000, inside, 150_000),
        (2, 5, outside.as_mut_ptr(), outside.len()),
    ];
    for (id, offset, dst, len) in reads {
        let read = Read {
            id,
            fd: fixture.file.as_raw_fd(),
            offset,
            dst,
            len,
        };
        // SAFETY: the destinations and the file outlive the reads, reaped below
        unsafe { mover.submit(read).unwrap() };
    }
    while let Some(done) = mover.wait_done() {
        done.result.unwrap();
    }
    drop(mover);
    assert_eq!(region[..100_000], fixture.data[..100_000]);
    assert_eq!(region[500_000..650_000], fixture.data[100_000..250_000]);
    assert_eq!(outside, fixture.data[5..1_005]);
}
