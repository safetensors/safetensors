//! C++ access to the Rust safetensors parser and serializer.

use safetensors::tensor::{Metadata, TensorInfo, TensorView};
use safetensors::{Dtype, SafeTensorError, SafeTensors};
use serde::Deserialize;
use std::collections::{HashMap, HashSet};
use std::path::Path;

type Error = Box<dyn std::error::Error>;

#[cxx::bridge(namespace = "safetensors")]
pub mod ffi {
    /// Borrowed, contiguous, little-endian tensor data. The caller owns all slices.
    struct Tensor<'a> {
        name: &'a str,
        dtype: &'a str,
        shape: &'a [usize],
        data: &'a [u8],
    }

    /// A string-valued entry in the optional file metadata.
    struct MetadataEntry {
        key: String,
        value: String,
    }

    extern "Rust" {
        type Archive;

        /// Serialize borrowed tensor buffers into an owned file byte buffer.
        fn serialize(tensors: &[Tensor], metadata: &[MetadataEntry]) -> Result<Vec<u8>>;
        /// Write borrowed tensor buffers without allocating a whole-file buffer.
        fn save_file(tensors: &[Tensor], metadata: &[MetadataEntry], filename: &str) -> Result<()>;
        /// Parse an owned file byte buffer, transferring ownership without copying it.
        fn deserialize(buffer: Vec<u8>) -> Result<Box<Archive>>;
        /// Read a file into memory and validate it with the Rust parser.
        fn load_file(filename: &str) -> Result<Box<Archive>>;
        /// Tensor names in file-offset order.
        fn names(self: &Archive) -> Vec<String>;
        /// Optional user metadata; an absent metadata map returns an empty vector.
        fn metadata(self: &Archive) -> Vec<MetadataEntry>;
        /// The format dtype name, such as F32, I64, or BF16.
        fn dtype(self: &Archive, name: &str) -> Result<String>;
        /// Dimensions borrowed from this archive.
        ///
        /// # Safety
        /// C++ callers must keep the archive alive while using the returned slice.
        #[allow(clippy::needless_lifetimes)]
        // CXX requires an explicit owner for the returned slice.
        unsafe fn shape<'a>(self: &'a Archive, name: &str) -> Result<&'a [usize]>;
        /// Bytes borrowed from this archive.
        ///
        /// # Safety
        /// C++ callers must keep the archive alive while using the returned slice.
        #[allow(clippy::needless_lifetimes)]
        // CXX requires an explicit owner for the returned slice.
        unsafe fn data<'a>(self: &'a Archive, name: &str) -> Result<&'a [u8]>;
    }
}

/// An immutable, validated archive owning its file bytes and parsed metadata.
pub struct Archive {
    buffer: Vec<u8>,
    metadata: Metadata,
    data_start: usize,
}

fn tensor_views<'a>(
    tensors: &'a [ffi::Tensor<'a>],
) -> Result<Vec<(&'a str, TensorView<'a>)>, Error> {
    let mut names = HashSet::new();
    tensors
        .iter()
        .map(|tensor| {
            if tensor.name == "__metadata__" || !names.insert(tensor.name) {
                return Err(format!("reserved or duplicate tensor name: {}", tensor.name).into());
            }
            let dtype = Dtype::deserialize(serde::de::value::StrDeserializer::<
                serde::de::value::Error,
            >::new(tensor.dtype))?;
            // TensorView::new currently multiplies unchecked. Validate before crossing
            // into it so malformed C++ shapes produce an exception, not a Rust panic.
            tensor
                .shape
                .iter()
                .try_fold(1usize, |size, dim| size.checked_mul(*dim))
                .and_then(|size| size.checked_mul(dtype.bitsize()))
                .ok_or(SafeTensorError::ValidationOverflow)?;
            let view = TensorView::new(dtype, tensor.shape.to_vec(), tensor.data)?;
            Ok((tensor.name, view))
        })
        .collect()
}

fn metadata_map(entries: &[ffi::MetadataEntry]) -> Result<Option<HashMap<String, String>>, Error> {
    if entries.is_empty() {
        return Ok(None);
    }
    let mut metadata = HashMap::new();
    for entry in entries {
        if metadata
            .insert(entry.key.clone(), entry.value.clone())
            .is_some()
        {
            return Err(format!("duplicate metadata key: {}", entry.key).into());
        }
    }
    Ok(Some(metadata))
}

/// Serialize C++ tensor slices using the core serializer.
pub fn serialize(
    tensors: &[ffi::Tensor<'_>],
    metadata: &[ffi::MetadataEntry],
) -> Result<Vec<u8>, Error> {
    Ok(safetensors::serialize(
        tensor_views(tensors)?,
        metadata_map(metadata)?,
    )?)
}

/// Save C++ tensor slices using the core file writer.
pub fn save_file(
    tensors: &[ffi::Tensor<'_>],
    metadata: &[ffi::MetadataEntry],
    filename: &str,
) -> Result<(), Error> {
    Ok(safetensors::serialize_to_file(
        tensor_views(tensors)?,
        metadata_map(metadata)?,
        Path::new(filename),
    )?)
}

/// Validate and take ownership of a serialized archive without copying tensor data.
pub fn deserialize(buffer: Vec<u8>) -> Result<Box<Archive>, Error> {
    let (header_len, metadata) = SafeTensors::read_metadata(&buffer)?;
    Ok(Box::new(Archive {
        buffer,
        metadata,
        data_start: 8 + header_len,
    }))
}

/// Read and validate a safetensors file.
pub fn load_file(filename: &str) -> Result<Box<Archive>, Error> {
    deserialize(std::fs::read(filename)?)
}

impl Archive {
    fn info(&self, name: &str) -> Result<&TensorInfo, SafeTensorError> {
        self.metadata
            .info(name)
            .ok_or_else(|| SafeTensorError::TensorNotFound(name.to_owned()))
    }

    /// Return tensor names ordered by their position in the file.
    pub fn names(&self) -> Vec<String> {
        self.metadata.offset_keys()
    }

    /// Return metadata entries sorted by key.
    pub fn metadata(&self) -> Vec<ffi::MetadataEntry> {
        let mut entries: Vec<_> = self
            .metadata
            .metadata()
            .iter()
            .flatten()
            .map(|(key, value)| ffi::MetadataEntry {
                key: key.clone(),
                value: value.clone(),
            })
            .collect();
        entries.sort_by(|left, right| left.key.cmp(&right.key));
        entries
    }

    /// Return a tensor's format dtype name.
    pub fn dtype(&self, name: &str) -> Result<String, Error> {
        Ok(self.info(name)?.dtype.to_string())
    }

    /// Borrow a tensor's dimensions for the lifetime of this archive.
    pub fn shape(&self, name: &str) -> Result<&[usize], Error> {
        Ok(&self.info(name)?.shape)
    }

    /// Borrow a tensor's little-endian bytes for the lifetime of this archive.
    pub fn data(&self, name: &str) -> Result<&[u8], Error> {
        let (start, end) = self.info(name)?.data_offsets;
        Ok(&self.buffer[self.data_start + start..self.data_start + end])
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn round_trip_matches_core_parser() {
        let bytes = [0, 0, 128, 63, 0, 0, 0, 64];
        let tensors = [ffi::Tensor {
            name: "weights",
            dtype: "F32",
            shape: &[2],
            data: &bytes,
        }];
        let metadata = [ffi::MetadataEntry {
            key: "source".into(),
            value: "cpp".into(),
        }];
        let serialized = serialize(&tensors, &metadata).unwrap();
        let parsed = SafeTensors::deserialize(&serialized).unwrap();
        assert_eq!(parsed.tensor("weights").unwrap().data(), bytes);
        let archive = deserialize(serialized).unwrap();
        assert_eq!(archive.names(), ["weights"]);
        assert_eq!(archive.dtype("weights").unwrap(), "F32");
        assert_eq!(archive.shape("weights").unwrap(), [2]);
        assert_eq!(archive.data("weights").unwrap(), bytes);
        assert_eq!(archive.metadata()[0].value, "cpp");
    }

    #[test]
    fn file_round_trip() {
        let directory = tempfile::tempdir().unwrap();
        let filename = directory.path().join("weights.safetensors");
        let tensors = [ffi::Tensor {
            name: "scalar",
            dtype: "I8",
            shape: &[],
            data: &[42],
        }];
        save_file(&tensors, &[], filename.to_str().unwrap()).unwrap();
        let archive = load_file(filename.to_str().unwrap()).unwrap();
        assert!(archive.shape("scalar").unwrap().is_empty());
        assert_eq!(archive.data("scalar").unwrap(), [42]);
        assert!(archive.metadata().is_empty());
    }

    #[test]
    fn empty_and_packed_tensors() {
        let tensors = [
            ffi::Tensor {
                name: "empty",
                dtype: "F32",
                shape: &[0, 2],
                data: &[],
            },
            ffi::Tensor {
                name: "packed",
                dtype: "F4",
                shape: &[2],
                data: &[0x12],
            },
        ];
        let archive = deserialize(serialize(&tensors, &[]).unwrap()).unwrap();
        assert_eq!(archive.shape("empty").unwrap(), [0, 2]);
        assert!(archive.data("empty").unwrap().is_empty());
        assert_eq!(archive.data("packed").unwrap(), [0x12]);
        assert!(deserialize(serialize(&[], &[]).unwrap())
            .unwrap()
            .names()
            .is_empty());
    }

    #[test]
    fn rejects_invalid_tensors() {
        for tensor in [
            ffi::Tensor {
                name: "x",
                dtype: "not-a-dtype",
                shape: &[1],
                data: &[1],
            },
            ffi::Tensor {
                name: "x",
                dtype: "F32",
                shape: &[1],
                data: &[1],
            },
            ffi::Tensor {
                name: "x",
                dtype: "F4",
                shape: &[1],
                data: &[1],
            },
            ffi::Tensor {
                name: "x",
                dtype: "U8",
                shape: &[usize::MAX, 2],
                data: &[],
            },
            ffi::Tensor {
                name: "__metadata__",
                dtype: "U8",
                shape: &[1],
                data: &[1],
            },
        ] {
            assert!(serialize(&[tensor], &[]).is_err());
        }
        let tensors = [
            ffi::Tensor {
                name: "x",
                dtype: "U8",
                shape: &[1],
                data: &[1],
            },
            ffi::Tensor {
                name: "x",
                dtype: "U8",
                shape: &[1],
                data: &[2],
            },
        ];
        assert!(serialize(&tensors, &[]).is_err());
    }

    #[test]
    fn rejects_duplicate_metadata() {
        let metadata = [
            ffi::MetadataEntry {
                key: "source".into(),
                value: "a".into(),
            },
            ffi::MetadataEntry {
                key: "source".into(),
                value: "b".into(),
            },
        ];
        assert!(serialize(&[], &metadata).is_err());
    }

    #[test]
    fn rejects_invalid_archives_and_missing_names() {
        assert!(deserialize(vec![]).is_err());
        assert!(deserialize(vec![255; 16]).is_err());
        let archive = deserialize(serialize(&[], &[]).unwrap()).unwrap();
        assert!(archive.data("missing").is_err());
        assert!(archive.shape("missing").is_err());
        assert!(archive.dtype("missing").is_err());
    }

    #[test]
    fn missing_file_is_an_error() {
        let directory = tempfile::tempdir().unwrap();
        assert!(load_file(directory.path().join("missing").to_str().unwrap()).is_err());
    }

    #[test]
    fn data_is_borrowed_from_owned_buffer() {
        let tensors = [ffi::Tensor {
            name: "x",
            dtype: "U8",
            shape: &[1],
            data: &[42],
        }];
        let buffer = serialize(&tensors, &[]).unwrap();
        let data_address = buffer.last().unwrap() as *const u8;
        let archive = deserialize(buffer).unwrap();
        assert_eq!(archive.data("x").unwrap().as_ptr(), data_address);
    }
}
