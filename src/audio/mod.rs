//! Container-independent audio input and codec initialization.
//!
//! Packet extraction is separate from PCM synthesis and device playback.

pub mod aac;
pub mod mp4;
pub mod opus;
pub mod transform;
pub mod vorbis;
pub mod webm;
