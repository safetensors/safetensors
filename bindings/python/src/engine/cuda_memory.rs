//! CUDA for the transfer engine: the stream each device's copies are queued
//! on, events as [`Fence`]s.

use std::sync::atomic::{AtomicU64, Ordering};

use super::{
    cuda::{api, CudaApi, CudaError, DeviceGuard, Event, Stream},
    transfer::{Error, Fence},
};

/// Whether a CUDA runtime is loaded in this process, as [`Loader::load`] needs
pub fn available() -> bool {
    api().is_some()
}

/// This is a generous limit on the maximum number of devices that could be connected to a single host
pub const MAX_DEVICES: usize = 64;
static STREAMS: [AtomicU64; MAX_DEVICES] = [const { AtomicU64::new(0) }; MAX_DEVICES];

fn get_stream(stream_slots: &[AtomicU64], api: &CudaApi, device: i32) -> Result<Stream, CudaError> {
    let slot = stream_slots
        .get(device as usize)
        .unwrap_or_else(|| panic!("device index {device} exceeds MAX_DEVICES ({MAX_DEVICES})"));
    let stream = slot.load(Ordering::Acquire);
    if stream != 0 {
        return Ok(stream as Stream);
    }

    let new = api.with_device(device, |d| d.stream_create())?;
    match slot.compare_exchange(0, new as u64, Ordering::AcqRel, Ordering::Acquire) {
        Ok(_) => Ok(new),
        Err(existing) => {
            let _ = api.stream_destroy(new);
            Ok(existing as Stream)
        }
    }
}

pub(super) fn device_stream(api: &CudaApi, device: i32) -> Result<Stream, CudaError> {
    get_stream(&STREAMS, api, device)
}

/// An event recorded on a stream: the work queued on it before
pub struct CudaEvent {
    api: &'static CudaApi,
    event: Event,
}

// SAFETY: an event handle is used from any thread through the runtime API
unsafe impl Send for CudaEvent {}
// SAFETY: as for `Send`: the runtime API is thread safe
unsafe impl Sync for CudaEvent {}

impl CudaEvent {
    pub(super) fn record_on(
        api: &'static CudaApi,
        d: &DeviceGuard<'_>,
        stream: Stream,
    ) -> Result<Self, CudaError> {
        let event = d.event_create()?;
        if let Err(err) = api.event_record(event, stream) {
            let _ = api.event_destroy(event);
            return Err(err);
        }
        Ok(Self { api, event })
    }
}

impl Fence for CudaEvent {
    fn wait(&self) -> Result<(), Error> {
        Ok(self.api.event_sync(self.event)?)
    }
}

impl Drop for CudaEvent {
    fn drop(&mut self) {
        let _ = self.api.event_destroy(self.event);
    }
}
