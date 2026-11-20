#[cfg(target_os = "linux")]
use std::sync::atomic::{AtomicBool, Ordering};
use std::{fs::File, ops::Range, os::unix::fs::FileExt, sync::Arc};

use super::{
    pool::Slab,
    transfer::{Error, Source, Staged},
};

/// The largest alignment of offset, length and buffer a direct read may need
pub(super) const DIRECT_ALIGN: usize = 4096;

/// A file read with `O_DIRECT` where the filesystem allows it
pub(super) struct DirectFile {
    file: Arc<File>,
    #[cfg(target_os = "linux")]
    direct: Option<Direct>,
}

impl DirectFile {
    /// `direct` as in [`Loader::load`]. Direct reads need `aligned_slabs` and a
    /// filesystem that supports them; reads go through the page cache otherwise.
    pub(super) fn new(file: Arc<File>, direct: Option<bool>, aligned_slabs: bool) -> Self {
        #[cfg(target_os = "linux")]
        {
            use std::os::fd::AsRawFd;
            let direct = match direct {
                Some(true) | None if aligned_slabs => Direct::open(&file, direct.is_none()),
                _ => None,
            };
            if direct.is_none() {
                // Chunks are read out of order by several threads and a plan may
                // skip regions: kernel readahead would pull in bytes nobody asked
                // for. Best effort, the hint failing only costs throughput.
                // SAFETY: `file` holds an open descriptor for the call's duration.
                let _ =
                    unsafe { libc::posix_fadvise(file.as_raw_fd(), 0, 0, libc::POSIX_FADV_RANDOM) };
            }
            Self { file, direct }
        }
        #[cfg(not(target_os = "linux"))]
        {
            let _ = (direct, aligned_slabs);
            Self { file }
        }
    }

    /// Reads `len` bytes at `offset` into `slab`, aligned to [`DIRECT_ALIGN`] and
    /// at least `len + 2 * DIRECT_ALIGN` long, and returns where they start in it.
    pub(super) fn read(&self, slab: &mut [u8], offset: u64, len: usize) -> std::io::Result<usize> {
        #[cfg(target_os = "linux")]
        if let Some(direct) = self
            .direct
            .as_ref()
            .filter(|d| d.ok.load(Ordering::Relaxed))
        {
            if direct.cached_first && read_cached(&self.file, &mut slab[..len], offset) == len {
                return Ok(0);
            }
            match direct.read(slab, offset, len) {
                Err(e) if e.raw_os_error() == Some(libc::EINVAL) => {
                    direct.ok.store(false, Ordering::Relaxed);
                }
                res => return res,
            }
        }
        self.file.read_exact_at(&mut slab[..len], offset)?;
        Ok(0)
    }
}

impl Source for DirectFile {
    fn fill(&self, file: Range<u64>, mut slab: Slab) -> Result<Staged, Error> {
        let head = self.read(&mut slab, file.start, (file.end - file.start) as usize)?;
        Ok(Staged::new(slab, head, file))
    }
}

/// The file reopened `O_DIRECT`
#[cfg(target_os = "linux")]
struct Direct {
    file: File,
    /// of offsets, lengths and buffers; divides [`DIRECT_ALIGN`]
    align: usize,
    /// whether a chunk the page cache holds is read from it
    cached_first: bool,
    /// cleared when a direct read fails with `EINVAL`
    ok: AtomicBool,
}

#[cfg(target_os = "linux")]
impl Direct {
    fn open(file: &File, cached_first: bool) -> Option<Self> {
        use std::os::{fd::AsRawFd, unix::fs::OpenOptionsExt};
        let align = direct_align(file)?;
        let file = std::fs::OpenOptions::new()
            .read(true)
            .custom_flags(libc::O_DIRECT)
            .open(format!("/proc/self/fd/{}", file.as_raw_fd()))
            .ok()?;
        Some(Self {
            file,
            align,
            cached_first,
            ok: AtomicBool::new(true),
        })
    }

    /// Reads the [`Self::align`]ed superset of `len` bytes at `offset` into
    /// `slab` and returns where they start in it; the end of the file may cut the
    /// superset short.
    fn read(&self, slab: &mut [u8], offset: u64, len: usize) -> std::io::Result<usize> {
        let head = (offset % self.align as u64) as usize;
        let from = offset - head as u64;
        let buf = &mut slab[..(head + len).next_multiple_of(self.align)];
        let mut got = 0;
        while got < head + len {
            match self.file.read_at(&mut buf[got..], from + got as u64) {
                Ok(0) => return Err(std::io::ErrorKind::UnexpectedEof.into()),
                Ok(n) => got += n,
                Err(e) if e.kind() == std::io::ErrorKind::Interrupted => {}
                Err(e) => return Err(e),
            }
        }
        Ok(head)
    }
}

/// The alignment direct reads of `file` need, as `statx` reports it, or
/// [`DIRECT_ALIGN`] where it does not; `None` if it does not divide
/// [`DIRECT_ALIGN`] or the file takes no direct reads.
#[cfg(target_os = "linux")]
#[cfg_attr(not(target_env = "gnu"), allow(unused_variables))]
fn direct_align(file: &File) -> Option<usize> {
    #[cfg(target_env = "gnu")]
    {
        use std::os::fd::AsRawFd;
        // SAFETY: `statx` is plain integers, valid zeroed; `file` holds an open
        // descriptor, and `AT_EMPTY_PATH` with an empty path describes it
        let (r, stx) = unsafe {
            let mut stx: libc::statx = std::mem::zeroed();
            let r = libc::statx(
                file.as_raw_fd(),
                c"".as_ptr(),
                libc::AT_EMPTY_PATH,
                libc::STATX_DIOALIGN,
                &mut stx,
            );
            (r, stx)
        };
        if r == 0 && stx.stx_mask & libc::STATX_DIOALIGN != 0 {
            let align = stx.stx_dio_mem_align.max(stx.stx_dio_offset_align) as usize;
            return (align != 0 && DIRECT_ALIGN.is_multiple_of(align)).then_some(align);
        }
    }
    Some(DIRECT_ALIGN)
}

/// Reads what the page cache holds of `buf.len()` bytes at `offset`, up to the
/// first byte it does not, without waiting for the disk.
#[cfg(target_os = "linux")]
fn read_cached(file: &File, buf: &mut [u8], offset: u64) -> usize {
    use std::os::fd::AsRawFd;
    let Ok(offset) = libc::off_t::try_from(offset) else {
        return 0;
    };
    let iov = libc::iovec {
        iov_base: buf.as_mut_ptr().cast(),
        iov_len: buf.len(),
    };
    // SAFETY: `iov` describes `buf`, writable for the call's duration
    let n = unsafe { libc::preadv2(file.as_raw_fd(), &iov, 1, offset, libc::RWF_NOWAIT) };
    n.max(0) as usize
}
#[cfg(test)]
pub(super) mod tests {
    use std::ops::Range;

    use super::DIRECT_ALIGN;
    #[cfg(target_os = "linux")]
    use {
        super::DirectFile,
        crate::engine::pool::SLAB_SIZE,
        std::{fs::File, sync::Arc},
    };

    pub(crate) struct Rng(pub(crate) u64);
    impl Rng {
        pub(crate) fn below(&mut self, n: usize) -> usize {
            self.0 = self
                .0
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            ((self.0 >> 33) % n as u64) as usize
        }
    }

    /// `len` pseudo-random bytes in a new file, and the bytes
    #[cfg(target_os = "linux")]
    pub(crate) fn random_file(name: &str, len: usize) -> (std::path::PathBuf, Arc<File>, Vec<u8>) {
        let mut rng = Rng(len as u64);
        let data: Vec<u8> = (0..len).map(|_| rng.below(256) as u8).collect();
        let path = std::env::temp_dir().join(format!("st_{name}_{}", std::process::id()));
        std::fs::write(&path, &data).unwrap();
        (path.clone(), Arc::new(File::open(path).unwrap()), data)
    }

    /// Drops `file` from the page cache
    #[cfg(target_os = "linux")]
    pub(crate) fn evict(file: &File) {
        use std::os::fd::AsRawFd;
        file.sync_all().unwrap();
        // SAFETY: `file` holds an open descriptor for the call's duration
        unsafe { libc::posix_fadvise(file.as_raw_fd(), 0, 0, libc::POSIX_FADV_DONTNEED) };
    }

    /// A zeroed buffer of `len` bytes, `misalign` bytes past a [`DIRECT_ALIGN`] boundary
    pub(crate) fn slab_buf(len: usize, misalign: usize) -> (Vec<u8>, Range<usize>) {
        let buf = vec![0u8; len + DIRECT_ALIGN + misalign];
        let at = buf.as_ptr().align_offset(DIRECT_ALIGN) + misalign;
        (buf, at..at + len)
    }

    #[cfg(target_os = "linux")]
    #[test]
    fn test_source_reads_cold_and_cached() {
        // odd offsets, and a tail whose aligned superset reaches past the end of a
        // file of 125 blocks and 123 bytes
        let (path, file, data) = random_file("source", 125 * DIRECT_ALIGN + 123);
        let cases = [
            (0, 1),
            (1, 4095),
            (4095, 2),
            (4096, 8192),
            (4099, data.len() - 4099),
            (data.len() - 7, 7),
        ];
        let (mut buf, at) = slab_buf(SLAB_SIZE, 0);
        for direct in [None, Some(true), Some(false)] {
            let source = DirectFile::new(file.clone(), direct, true);
            for cold in [true, false] {
                for (offset, len) in cases {
                    if cold {
                        evict(&file);
                    }
                    let head = source
                        .read(&mut buf[at.clone()], offset as u64, len)
                        .unwrap();
                    assert_eq!(
                        &buf[at.start + head..][..len],
                        &data[offset..offset + len],
                        "{direct:?}, cold {cold}, {offset}+{len}"
                    );
                }
            }
        }
        std::fs::remove_file(path).unwrap();
    }

    #[cfg(target_os = "linux")]
    #[test]
    fn test_source_falls_back_when_direct_is_refused() {
        // a direct read into a misaligned buffer fails with EINVAL
        let (path, file, data) = random_file("einval", 3 * DIRECT_ALIGN);
        let (mut buf, at) = slab_buf(SLAB_SIZE, 1);
        let source = DirectFile::new(file.clone(), Some(true), true);
        if let Some(direct) = &source.direct {
            evict(&file);
            let head = source.read(&mut buf[at.clone()], 5, 5000).unwrap();
            assert_eq!(&buf[at.start + head..][..5000], &data[5..5005]);
            assert!(!direct.ok.load(std::sync::atomic::Ordering::Relaxed));
        } else {
            eprintln!("skipped: no direct reads in {:?}", std::env::temp_dir());
        }
        std::fs::remove_file(path).unwrap();
    }
}
