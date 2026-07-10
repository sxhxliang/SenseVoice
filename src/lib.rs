//! Rust bindings for the local SenseVoice/FSMN-VAD ggml runtime.
//!
//! The native runtime expects 16 kHz mono `f32` PCM samples in `[-1.0, 1.0]` for
//! streaming APIs. File APIs use the bundled miniaudio decoder on the C++ side.

use std::ffi::{CStr, CString};
use std::fmt;
use std::fs;
use std::os::raw::{c_char, c_float, c_int, c_void};
use std::path::{Path, PathBuf};
use std::ptr::{self, NonNull};
use std::slice;

#[allow(unsafe_code)]
mod ffi {
    use super::{c_char, c_float, c_int, c_void};

    #[repr(C)]
    pub struct SvConfig {
        pub sensevoice_model: *const c_char,
        pub vad_model: *const c_char,
        pub n_threads: c_int,
        pub vad_max_seg_ms: c_int,
        pub realtime_min_silence_ms: c_int,
        pub realtime_step_ms: c_int,
        pub keep_tags: c_int,
    }

    #[repr(C)]
    #[derive(Clone, Copy)]
    pub struct SvSegment {
        pub start_ms: c_int,
        pub end_ms: c_int,
    }

    #[link(name = "funasr_rs")]
    unsafe extern "C" {
        pub fn sv_recognizer_new(config: *const SvConfig, out: *mut *mut c_void) -> c_int;
        pub fn sv_recognizer_free(recognizer: *mut c_void);
        pub fn sv_last_error(recognizer: *const c_void) -> *const c_char;
        pub fn sv_recognize_file(
            recognizer: *mut c_void,
            audio_path: *const c_char,
            out_text: *mut *mut c_char,
        ) -> c_int;
        pub fn sv_recognize_pcm(
            recognizer: *mut c_void,
            samples: *const c_float,
            sample_count: usize,
            out_text: *mut *mut c_char,
        ) -> c_int;
        pub fn sv_vad_segments(
            recognizer: *mut c_void,
            samples: *const c_float,
            sample_count: usize,
            out_segments: *mut *mut SvSegment,
            out_count: *mut usize,
        ) -> c_int;
        pub fn sv_realtime_accept_pcm(
            recognizer: *mut c_void,
            samples: *const c_float,
            sample_count: usize,
            out_text: *mut *mut c_char,
        ) -> c_int;
        pub fn sv_realtime_flush(recognizer: *mut c_void, out_text: *mut *mut c_char) -> c_int;
        pub fn sv_realtime_reset(recognizer: *mut c_void);
        pub fn sv_string_free(text: *mut c_char);
        pub fn sv_segments_free(segments: *mut SvSegment);
    }
}

pub type Result<T> = std::result::Result<T, SenseVoiceError>;

#[derive(Debug, Clone, Eq, PartialEq)]
pub struct SenseVoiceError {
    message: String,
}

impl SenseVoiceError {
    fn new(message: impl Into<String>) -> Self {
        Self {
            message: message.into(),
        }
    }
}

impl fmt::Display for SenseVoiceError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.message)
    }
}

impl std::error::Error for SenseVoiceError {}

#[derive(Debug, Clone)]
pub struct RecognizerConfig {
    pub sensevoice_model: PathBuf,
    pub vad_model: Option<PathBuf>,
    pub n_threads: i32,
    pub vad_max_seg_ms: i32,
    pub realtime_min_silence_ms: i32,
    pub realtime_step_ms: i32,
    pub keep_tags: bool,
}

impl RecognizerConfig {
    pub fn new(sensevoice_model: impl Into<PathBuf>) -> Self {
        Self {
            sensevoice_model: sensevoice_model.into(),
            vad_model: None,
            n_threads: 8,
            vad_max_seg_ms: 30_000,
            realtime_min_silence_ms: 600,
            realtime_step_ms: 500,
            keep_tags: false,
        }
    }

    pub fn from_models_dir(models_dir: impl AsRef<Path>) -> Result<Self> {
        let models_dir = models_dir.as_ref();
        let sensevoice_model = find_gguf(models_dir, "sensevoice")?.ok_or_else(|| {
            SenseVoiceError::new(format!(
                "no SenseVoice GGUF found in {}",
                models_dir.display()
            ))
        })?;
        let vad_model = find_gguf(models_dir, "vad")?;
        Ok(Self {
            sensevoice_model,
            vad_model,
            ..Self::new("")
        })
    }

    pub fn with_vad_model(mut self, vad_model: impl Into<PathBuf>) -> Self {
        self.vad_model = Some(vad_model.into());
        self
    }

    pub fn with_threads(mut self, n_threads: i32) -> Self {
        self.n_threads = n_threads;
        self
    }

    pub fn with_keep_tags(mut self, keep_tags: bool) -> Self {
        self.keep_tags = keep_tags;
        self
    }
}

#[derive(Debug, Clone, Copy, Eq, PartialEq)]
pub struct SpeechSegment {
    pub start_ms: i32,
    pub end_ms: i32,
}

pub struct Recognizer {
    raw: NonNull<c_void>,
}

// SAFETY: the native handle owns its state and may be moved to another thread.
// Methods require `&mut self`, and `Recognizer` is intentionally not `Sync`, so
// safe Rust cannot call into the same native handle concurrently.
#[allow(unsafe_code)]
unsafe impl Send for Recognizer {}

#[allow(unsafe_code)]
impl Recognizer {
    pub fn new(config: RecognizerConfig) -> Result<Self> {
        if !config.sensevoice_model.is_file() {
            return Err(SenseVoiceError::new(format!(
                "SenseVoice model file not found: {}",
                config.sensevoice_model.display()
            )));
        }
        if let Some(vad_model) = config.vad_model.as_ref()
            && !vad_model.is_file()
        {
            return Err(SenseVoiceError::new(format!(
                "FSMN-VAD model file not found: {}",
                vad_model.display()
            )));
        }

        let sensevoice_model = path_to_cstring(&config.sensevoice_model)?;
        let vad_model = match config.vad_model.as_ref() {
            Some(path) => Some(path_to_cstring(path)?),
            None => None,
        };

        let raw_config = ffi::SvConfig {
            sensevoice_model: sensevoice_model.as_ptr(),
            vad_model: vad_model.as_ref().map_or(ptr::null(), |path| path.as_ptr()),
            n_threads: config.n_threads,
            vad_max_seg_ms: config.vad_max_seg_ms,
            realtime_min_silence_ms: config.realtime_min_silence_ms,
            realtime_step_ms: config.realtime_step_ms,
            keep_tags: if config.keep_tags { 1 } else { 0 },
        };

        let mut raw = ptr::null_mut();
        // SAFETY: `raw_config` points to C strings that live for this call, and
        // `raw` is a valid out-pointer initialized by the native constructor.
        let code = unsafe { ffi::sv_recognizer_new(&raw_config, &mut raw) };
        if code != 0 {
            return Err(SenseVoiceError::new(
                "failed to initialize SenseVoice recognizer; check model paths and GGUF format",
            ));
        }
        let raw = NonNull::new(raw)
            .ok_or_else(|| SenseVoiceError::new("native recognizer returned null"))?;
        Ok(Self { raw })
    }

    pub fn from_models_dir(models_dir: impl AsRef<Path>) -> Result<Self> {
        Self::new(RecognizerConfig::from_models_dir(models_dir)?)
    }

    pub fn transcribe_file(&mut self, audio_path: impl AsRef<Path>) -> Result<String> {
        let audio_path = path_to_cstring(audio_path.as_ref())?;
        self.call_string(|raw, out| {
            // SAFETY: `audio_path` is a valid C string for this call, and `out`
            // points to storage for the native allocated string.
            unsafe { ffi::sv_recognize_file(raw, audio_path.as_ptr(), out) }
        })
    }

    pub fn transcribe_pcm_16k(&mut self, samples: &[f32]) -> Result<String> {
        self.call_string(|raw, out| {
            // SAFETY: `samples.as_ptr()` is valid for `samples.len()` elements,
            // and the native function does not retain it after returning.
            unsafe { ffi::sv_recognize_pcm(raw, samples.as_ptr(), samples.len(), out) }
        })
    }

    pub fn vad_segments(&mut self, samples: &[f32]) -> Result<Vec<SpeechSegment>> {
        let mut raw_segments: *mut ffi::SvSegment = ptr::null_mut();
        let mut count = 0usize;
        // SAFETY: all pointers are valid for the duration of the call; returned
        // segment memory is freed with `sv_segments_free` below.
        let code = unsafe {
            ffi::sv_vad_segments(
                self.raw.as_ptr(),
                samples.as_ptr(),
                samples.len(),
                &mut raw_segments,
                &mut count,
            )
        };
        if code != 0 {
            return Err(self.last_error());
        }
        if raw_segments.is_null() || count == 0 {
            return Ok(Vec::new());
        }
        // SAFETY: native code returned `count` initialized `SvSegment` values.
        let segments = unsafe { slice::from_raw_parts(raw_segments, count) }
            .iter()
            .map(|segment| SpeechSegment {
                start_ms: segment.start_ms,
                end_ms: segment.end_ms,
            })
            .collect();
        // SAFETY: `raw_segments` was allocated by the native library.
        unsafe { ffi::sv_segments_free(raw_segments) };
        Ok(segments)
    }

    pub fn accept_realtime_pcm_16k(&mut self, samples: &[f32]) -> Result<Option<String>> {
        let text = self.call_string(|raw, out| {
            // SAFETY: `samples` is only borrowed for the call; native code copies
            // it into the recognizer stream buffer.
            unsafe { ffi::sv_realtime_accept_pcm(raw, samples.as_ptr(), samples.len(), out) }
        })?;
        Ok(non_empty(text))
    }

    pub fn flush_realtime(&mut self) -> Result<Option<String>> {
        let text = self.call_string(|raw, out| {
            // SAFETY: `out` points to storage for the native allocated string.
            unsafe { ffi::sv_realtime_flush(raw, out) }
        })?;
        Ok(non_empty(text))
    }

    pub fn reset_realtime(&mut self) {
        // SAFETY: `self.raw` is a live native recognizer handle.
        unsafe { ffi::sv_realtime_reset(self.raw.as_ptr()) };
    }

    fn call_string(
        &mut self,
        call: impl FnOnce(*mut c_void, *mut *mut c_char) -> c_int,
    ) -> Result<String> {
        let mut raw_text: *mut c_char = ptr::null_mut();
        let code = call(self.raw.as_ptr(), &mut raw_text);
        if code != 0 {
            return Err(self.last_error());
        }
        if raw_text.is_null() {
            return Ok(String::new());
        }
        // SAFETY: native code returns a NUL-terminated string allocated by the
        // same library; we copy it before freeing.
        let text = unsafe { CStr::from_ptr(raw_text) }
            .to_string_lossy()
            .into_owned();
        // SAFETY: `raw_text` was allocated by the native library.
        unsafe { ffi::sv_string_free(raw_text) };
        Ok(text)
    }

    fn last_error(&self) -> SenseVoiceError {
        // SAFETY: `self.raw` is a live handle; the returned pointer is owned by
        // the recognizer and valid until the next native call on this handle.
        let ptr = unsafe { ffi::sv_last_error(self.raw.as_ptr()) };
        if ptr.is_null() {
            return SenseVoiceError::new("native SenseVoice call failed");
        }
        // SAFETY: native code returns a NUL-terminated error string.
        let message = unsafe { CStr::from_ptr(ptr) }
            .to_string_lossy()
            .into_owned();
        SenseVoiceError::new(if message.is_empty() {
            "native SenseVoice call failed".to_string()
        } else {
            message
        })
    }
}

#[allow(unsafe_code)]
impl Drop for Recognizer {
    fn drop(&mut self) {
        // SAFETY: `self.raw` is owned by this wrapper and freed exactly once.
        unsafe { ffi::sv_recognizer_free(self.raw.as_ptr()) };
    }
}

pub struct RealtimeRecognizer {
    inner: Recognizer,
}

impl RealtimeRecognizer {
    pub fn new(config: RecognizerConfig) -> Result<Self> {
        if config.vad_model.is_none() {
            return Err(SenseVoiceError::new(
                "RealtimeRecognizer requires an FSMN-VAD model in RecognizerConfig::vad_model",
            ));
        }
        Ok(Self {
            inner: Recognizer::new(config)?,
        })
    }

    pub fn from_models_dir(models_dir: impl AsRef<Path>) -> Result<Self> {
        Self::new(RecognizerConfig::from_models_dir(models_dir)?)
    }

    pub fn accept_pcm_16k(&mut self, samples: &[f32]) -> Result<Option<String>> {
        self.inner.accept_realtime_pcm_16k(samples)
    }

    pub fn flush(&mut self) -> Result<Option<String>> {
        self.inner.flush_realtime()
    }

    pub fn reset(&mut self) {
        self.inner.reset_realtime();
    }

    pub fn into_inner(self) -> Recognizer {
        self.inner
    }
}

fn path_to_cstring(path: &Path) -> Result<CString> {
    let path = path.to_str().ok_or_else(|| {
        SenseVoiceError::new(format!("path is not valid UTF-8: {}", path.display()))
    })?;
    CString::new(path).map_err(|_| SenseVoiceError::new(format!("path contains NUL byte: {path}")))
}

fn find_gguf(models_dir: &Path, needle: &str) -> Result<Option<PathBuf>> {
    let entries = fs::read_dir(models_dir).map_err(|error| {
        SenseVoiceError::new(format!("failed to read {}: {error}", models_dir.display()))
    })?;
    let needle = needle.to_ascii_lowercase();
    let mut candidates = Vec::new();
    for entry in entries {
        let entry = entry.map_err(|error| {
            SenseVoiceError::new(format!("failed to read model entry: {error}"))
        })?;
        let path = entry.path();
        if !path.is_file() {
            continue;
        }
        let Some(file_name) = path.file_name().and_then(|name| name.to_str()) else {
            continue;
        };
        let file_name = file_name.to_ascii_lowercase();
        if file_name.ends_with(".gguf") && file_name.contains(&needle) {
            candidates.push(path);
        }
    }
    candidates.sort_by(|left, right| {
        let left_rank = model_rank(left);
        let right_rank = model_rank(right);
        left_rank.cmp(&right_rank).then_with(|| left.cmp(right))
    });
    Ok(candidates.into_iter().next())
}

fn model_rank(path: &Path) -> i32 {
    let name = path
        .file_name()
        .and_then(|name| name.to_str())
        .unwrap_or_default()
        .to_ascii_lowercase();
    if name.contains("q8") {
        0
    } else if name.contains("f16") {
        1
    } else {
        2
    }
}

fn non_empty(text: String) -> Option<String> {
    if text.is_empty() { None } else { Some(text) }
}

#[cfg(test)]
#[allow(unsafe_code)]
mod tests {
    use super::{ffi, ptr};

    #[test]
    fn native_runtime_can_be_loaded() {
        // SAFETY: C `free(NULL)` is a no-op. This call verifies that the native
        // library can be linked and loaded without requiring model files.
        unsafe { ffi::sv_string_free(ptr::null_mut()) };
    }
}
