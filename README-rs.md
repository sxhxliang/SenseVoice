# sensevoice

Rust bindings for the local SenseVoice GGUF runtime in `runtime/llama.cpp`.

This crate packages the Rust API and native ggml/FSMN-VAD runtime sources only.
It does not package model files.

## Models

Create a local model directory and place the GGUF files there:

```text
models/
  sensevoice-small-q8.gguf
  fsmn-vad.gguf
```

Pre-converted models are available from:

- SenseVoiceSmall GGUF: https://huggingface.co/FunAudioLLM/SenseVoiceSmall-GGUF
- FSMN-VAD GGUF: https://huggingface.co/FunAudioLLM/fsmn-vad-GGUF

## File transcription

```rust
use sensevoice::Recognizer;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut recognizer = Recognizer::from_models_dir("models")?;
    let text = recognizer.transcribe_file("audio.wav")?;
    println!("{text}");
    Ok(())
}
```

## Realtime PCM

The realtime API expects 16 kHz mono `f32` PCM samples in `[-1.0, 1.0]`.

```rust
use sensevoice::RealtimeRecognizer;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut recognizer = RealtimeRecognizer::from_models_dir("models")?;

    // Feed 16 kHz mono f32 PCM chunks.
    if let Some(text) = recognizer.accept_pcm_16k(&[0.0; 1600])? {
        print!("{text}");
    }
    if let Some(text) = recognizer.flush()? {
        print!("{text}");
    }

    Ok(())
}
```

## Microphone example

The microphone example uses `cpal`, which is a dev-dependency and is not required
by library users.

```bash
cargo run --example microphone_realtime -- --list-devices
cargo run --example microphone_realtime -- --device 2
```

Use `--verbose` to print input levels and endpoint diagnostics.

For lower-latency interactive use, the example continuously re-runs ASR on the
current speech buffer and refreshes the same stdout line before an endpoint is
detected.

## Native build

The native runtime is built by CMake. Install:

- Rust 1.85+
- CMake
- A C/C++ toolchain
- Linux microphone example only: ALSA development headers, e.g. `libasound2-dev`

Supported build targets are macOS, Linux, and Windows on the architectures
supported by `llama.cpp`/ggml. The crate is a native binding crate, so
cross-compilation also needs a matching C/C++ toolchain and CMake generator for
the target platform.

By default the CMake project fetches a pinned `llama.cpp` revision. For offline
or reproducible builds, provide a local checkout:

```bash
FUNASR_LLAMA_SOURCE=/path/to/llama.cpp cargo build
```

Model files are runtime inputs and are intentionally not part of the crate package.

## Publishing

The crate package uses an explicit `include` whitelist in `Cargo.toml`; `models/`
is also git-ignored. Release packages contain the Rust API, examples, and native
runtime bridge sources, but not SenseVoice/FSMN-VAD model files.

Before publishing:

```bash
cargo fmt --check
cargo check --examples
DOCS_RS=1 cargo doc --no-deps
cargo package --list
cargo publish --dry-run
```

If the local build should avoid network access to fetch `llama.cpp`, set
`FUNASR_LLAMA_SOURCE` for the `cargo check`, `cargo package`, and
`cargo publish --dry-run` commands.
