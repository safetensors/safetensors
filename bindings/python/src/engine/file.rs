use std::{fs::File, ops::Range, os::unix::fs::FileExt, sync::Arc};
#[cfg(target_os = "linux")]
use std::{
    path::{Path, PathBuf},
    sync::atomic::{AtomicBool, Ordering},
};

use super::{
    pool::Slab,
    transfer::{Error, ReadAt, Source, Staged},
};
#[cfg(target_os = "linux")]
use super::{
    pool::SlabPool,
    transfer::{transfer, Plan, Sink, CHUNK_SIZE},
};

/// The largest alignment of offset, length and buffer a direct read may need
pub(super) const DIRECT_ALIGN: usize = 4096;

/// A file read or written with `O_DIRECT` where the filesystem allows it
pub(super) struct DirectFile {
    file: Arc<File>,
    #[cfg(target_os = "linux")]
    direct: Option<Direct>,
    /// of a file written: a direct write pads the chunk that ends there
    #[cfg(target_os = "linux")]
    size: u64,
}

impl DirectFile {
    /// `direct` as in [`Loader::load`]. Direct reads need `aligned_slabs` and a
    /// filesystem that supports them; reads go through the page cache otherwise.
    pub(super) fn new(file: Arc<File>, direct: Option<bool>, aligned_slabs: bool) -> Self {
        #[cfg(target_os = "linux")]
        {
            use std::os::fd::AsRawFd;
            let direct = match direct {
                Some(true) | None if aligned_slabs => Direct::open(&file, false, direct.is_none()),
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
            Self {
                file,
                direct,
                size: 0,
            }
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

    /// `file`, empty, to write `size` bytes to; the bytes are allocated now.
    #[cfg(target_os = "linux")]
    pub(super) fn create(file: Arc<File>, size: u64) -> std::io::Result<Self> {
        use std::os::fd::AsRawFd;
        // SAFETY: `file` holds an open descriptor for the call's duration
        let allocated = libc::off_t::try_from(size)
            .is_ok_and(|len| unsafe { libc::fallocate(file.as_raw_fd(), 0, 0, len) } == 0);
        if !allocated {
            file.set_len(size)?;
        }
        Ok(Self {
            direct: Direct::open(&file, true, false),
            file,
            size,
        })
    }
}

#[cfg(target_os = "linux")]
impl Sink for DirectFile {
    fn drain(&self, mut staged: Staged) -> Result<(), Error> {
        let offset = staged.file().start;
        if let Some(direct) = self
            .direct
            .as_ref()
            .filter(|d| d.ok.load(Ordering::Relaxed))
        {
            match direct.write(&mut staged, self.size) {
                Some(Err(e)) if e.raw_os_error() == Some(libc::EINVAL) => {
                    direct.ok.store(false, Ordering::Relaxed);
                }
                Some(written) => return Ok(written?),
                None => {}
            }
        }
        self.file.write_all_at(staged.bytes(), offset)?;
        Ok(())
    }
}

/// A new file beside a path, renamed over it by [`Self::persist`] and
/// removed if dropped before
#[cfg(target_os = "linux")]
struct TempFile {
    path: Option<PathBuf>,
}

#[cfg(target_os = "linux")]
impl TempFile {
    /// `.tmp` and 6 random letters or digits, in the directory of `path`, with
    /// the permissions of a new file. Some filesystems can't remove an open
    /// file: drop the file first.
    fn beside(path: &Path) -> std::io::Result<(Self, Arc<File>)> {
        use std::{hash::BuildHasher, os::unix::fs::OpenOptionsExt};
        const CHARS: &[u8] = b"abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789";
        let dir = match path.parent() {
            Some(dir) if !dir.as_os_str().is_empty() => dir,
            _ => Path::new("."),
        };
        let mut attempts = 0u32;
        loop {
            let mut n = std::collections::hash_map::RandomState::new().hash_one(attempts);
            let name: String = (0..6)
                .map(|_| {
                    let c = CHARS[(n % CHARS.len() as u64) as usize];
                    n /= CHARS.len() as u64;
                    c as char
                })
                .collect();
            let temp = dir.join(format!(".tmp{name}"));
            let opened = std::fs::OpenOptions::new()
                .read(true)
                .write(true)
                .create_new(true)
                .mode(0o666)
                .open(&temp);
            match opened {
                Ok(file) => return Ok((Self { path: Some(temp) }, Arc::new(file))),
                Err(e) if e.kind() == std::io::ErrorKind::AlreadyExists && attempts < 1 << 16 => {
                    attempts += 1;
                }
                Err(e) => return Err(e),
            }
        }
    }

    fn persist(mut self, path: &Path) -> std::io::Result<()> {
        std::fs::rename(self.path.as_ref().unwrap(), path)?;
        self.path = None;
        Ok(())
    }
}

#[cfg(target_os = "linux")]
impl Drop for TempFile {
    fn drop(&mut self) {
        if let Some(path) = &self.path {
            let _ = std::fs::remove_file(path);
        }
    }
}

/// Writes the `size` bytes `from` holds to a new file, which replaces `path`
/// once complete: a mapping of a previous `path` keeps its bytes.
#[cfg(target_os = "linux")]
pub(super) fn write_file(
    path: &Path,
    size: u64,
    from: &impl Source,
    pool: &Arc<SlabPool>,
    threads: usize,
) -> Result<(), Error> {
    let chunk = CHUNK_SIZE.get() as u64;
    let chunks = || -> Box<[Range<u64>]> {
        (0..size.div_ceil(chunk))
            .map(|c| c * chunk..((c + 1) * chunk).min(size))
            .collect()
    };
    let temp = || -> std::io::Result<(TempFile, Arc<File>)> {
        let (temp, file) = TempFile::beside(path)?;
        // The file keeps the permissions of the one it replaces, where the
        // filesystem has any.
        if let Ok(replaced) = std::fs::metadata(path) {
            let _ = file.set_permissions(replaced.permissions());
        }
        Ok((temp, file))
    };
    let replaced = (|| {
        let (temp, file) = temp()?;
        let to = DirectFile::create(file.clone(), size)?;
        // The last chunk goes first: a filesystem that only takes a file front
        // to back refuses it before any byte is written.
        let mut chunks = chunks().into_vec();
        let last = chunks.pop().into_iter().collect();
        transfer(Plan::new(last), from, &to, pool, 1)?;
        transfer(Plan::new(chunks.into()), from, &to, pool, threads)?;
        // drops the padding of a direct write
        file.set_len(size)?;
        temp.persist(path)?;
        Ok(())
    })();
    match replaced {
        // Object store mounts only take a file written front to back, with no
        // truncation or write at an offset; some have no rename either, and
        // then only take a new file.
        Err(Error::Io(e)) if refused(&e) => {
            let front_to_back = |file: Arc<File>| {
                let to = DirectFile {
                    file,
                    direct: None,
                    size,
                };
                transfer(Plan::new(chunks()), from, &to, pool, 1)
            };
            if path.exists() {
                let (temp, file) = temp()?;
                front_to_back(file)?;
                Ok(temp.persist(path)?)
            } else {
                let file = File::options().write(true).create_new(true).open(path)?;
                front_to_back(Arc::new(file)).inspect_err(|_| {
                    let _ = std::fs::remove_file(path);
                })
            }
        }
        replaced => replaced,
    }
}

#[cfg(target_os = "linux")]
fn refused(e: &std::io::Error) -> bool {
    matches!(
        e.raw_os_error(),
        Some(libc::ENOSYS | libc::EOPNOTSUPP | libc::EPERM | libc::EINVAL)
    )
}

impl Source for DirectFile {
    fn fill(&self, file: Range<u64>, mut slab: Slab) -> Result<Staged, Error> {
        let head = self.read(&mut slab, file.start, (file.end - file.start) as usize)?;
        Ok(Staged::new(slab, head, file))
    }

    fn unstaged(&self) -> Option<&dyn ReadAt> {
        // a direct read needs the alignment of a slab
        #[cfg(target_os = "linux")]
        if self.direct.is_some() {
            return None;
        }
        Some(self)
    }
}

impl ReadAt for DirectFile {
    fn read_at(&self, offset: u64, dst: &mut [u8]) -> std::io::Result<()> {
        self.file.read_exact_at(dst, offset)
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
    /// cleared when a direct read or write fails with `EINVAL`
    ok: AtomicBool,
}

#[cfg(target_os = "linux")]
impl Direct {
    fn open(file: &File, write: bool, cached_first: bool) -> Option<Self> {
        use std::os::{fd::AsRawFd, unix::fs::OpenOptionsExt};
        let align = direct_align(file)?;
        let file = std::fs::OpenOptions::new()
            .read(!write)
            .write(write)
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

    /// Writes `staged` at its offset in a file of `size` bytes, the chunk that
    /// ends the file padded with zeros up to the alignment; `None` when the
    /// offset, length or bytes are not aligned.
    fn write(&self, staged: &mut Staged, size: u64) -> Option<std::io::Result<()>> {
        let (offset, len) = (staged.file().start, staged.bytes().len());
        let len = match staged.file().end == size {
            true => len.next_multiple_of(self.align),
            false => len,
        };
        let buf = staged.padded(len)?;
        let aligned = offset % self.align as u64 == 0
            && len.is_multiple_of(self.align)
            && buf.as_ptr().addr().is_multiple_of(self.align);
        aligned.then(|| self.file.write_all_at(buf, offset))
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
