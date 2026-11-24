//! Locks taken whether or not a thread panicked while holding them. For state
//! that is whole at every point a panic can happen: a poisoned lock is then no
//! reason to panic again, least of all in a drop, where that aborts.

use std::sync::{Condvar, Mutex, MutexGuard, PoisonError};

pub(crate) fn lock<T>(mutex: &Mutex<T>) -> MutexGuard<'_, T> {
    mutex.lock().unwrap_or_else(PoisonError::into_inner)
}

pub(crate) fn wait<'a, T>(condvar: &Condvar, guard: MutexGuard<'a, T>) -> MutexGuard<'a, T> {
    condvar.wait(guard).unwrap_or_else(PoisonError::into_inner)
}
