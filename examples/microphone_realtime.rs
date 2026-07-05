use std::env;
use std::error::Error;
use std::ffi::OsString;
use std::io::{self, Write};
use std::path::PathBuf;
use std::sync::mpsc;
use std::time::{Duration, Instant};

use cpal::traits::{DeviceTrait, HostTrait, StreamTrait};
use sensevoice::{Recognizer, RecognizerConfig};

const OUTPUT_SAMPLE_RATE: f64 = 16_000.0;
const OUTPUT_SAMPLE_RATE_USIZE: usize = 16_000;
const PRE_ROLL_SAMPLES: usize = OUTPUT_SAMPLE_RATE_USIZE / 4;
const MIN_PREVIEW_SAMPLES: usize = OUTPUT_SAMPLE_RATE_USIZE / 2;
const PREVIEW_INTERVAL: Duration = Duration::from_millis(900);
const SPEECH_RMS_THRESHOLD: f32 = 0.006;
const SPEECH_PEAK_THRESHOLD: f32 = 0.02;
const AUTO_FLUSH_SILENCE: usize = OUTPUT_SAMPLE_RATE_USIZE;
const MAX_UNFLUSHED_AUDIO: usize = OUTPUT_SAMPLE_RATE_USIZE * 10;
const LEVEL_LOG_INTERVAL: Duration = Duration::from_secs(2);
const SILENCE_HINT_INTERVAL: Duration = Duration::from_secs(8);

fn main() -> Result<(), Box<dyn Error>> {
    let args = Args::parse(env::args_os().skip(1))?;

    let host = cpal::default_host();
    if args.list_devices {
        list_input_devices(&host)?;
        return Ok(());
    }

    let mut config = RecognizerConfig::from_models_dir(&args.models_dir)?;
    config.vad_model = None;
    let mut recognizer = Recognizer::new(config)?;

    let device = select_input_device(&host, args.device.as_deref())?;
    let supported_config = device.default_input_config()?;
    let sample_format = supported_config.sample_format();
    let config: cpal::StreamConfig = supported_config.into();
    let input_sample_rate = config.sample_rate.0;
    let channels = usize::from(config.channels);

    eprintln!(
        "listening on '{}' ({} Hz, {} channels, {:?}); press Enter to stop",
        device.name()?,
        input_sample_rate,
        channels,
        sample_format
    );
    eprintln!("speak continuously; live preview updates on stdout");
    if args.verbose {
        eprintln!("verbose diagnostics enabled");
    }

    let (audio_tx, audio_rx) = mpsc::channel();
    let stream = build_input_stream(
        &device,
        &config,
        sample_format,
        input_sample_rate,
        channels,
        audio_tx,
    )?;
    stream.play()?;

    let (stop_tx, stop_rx) = mpsc::channel();
    std::thread::spawn(move || {
        let mut line = String::new();
        let _ = io::stdin().read_line(&mut line);
        let _ = stop_tx.send(());
    });

    let mut stdout = io::stdout().lock();
    let mut transcript = TranscriptPrinter::new();
    let mut endpoint = EndpointState::new();
    let mut current_audio = Vec::new();
    let mut pre_roll = Vec::new();
    let mut last_preview = Instant::now();
    let mut last_level_log = Instant::now();
    let mut last_silence_hint = Instant::now();
    let mut has_seen_nonzero_level = false;
    loop {
        if stop_rx.try_recv().is_ok() {
            break;
        }

        match audio_rx.recv_timeout(Duration::from_millis(100)) {
            Ok(samples) => {
                let level = Level::from_samples(&samples);
                if args.verbose && last_level_log.elapsed() >= LEVEL_LOG_INTERVAL {
                    eprintln!("input level: rms={:.4}, peak={:.4}", level.rms, level.peak);
                    last_level_log = Instant::now();
                }
                if level.peak > 0.0001 {
                    has_seen_nonzero_level = true;
                } else if args.verbose
                    && !has_seen_nonzero_level
                    && last_silence_hint.elapsed() >= SILENCE_HINT_INTERVAL
                {
                    eprintln!(
                        "input is still silent; run with --list-devices and choose another device with --device"
                    );
                    last_silence_hint = Instant::now();
                }

                let sounds_like_speech = level.sounds_like_speech();
                if sounds_like_speech && current_audio.is_empty() {
                    current_audio.extend_from_slice(&pre_roll);
                }
                if sounds_like_speech || endpoint.heard_speech() {
                    current_audio.extend_from_slice(&samples);
                } else {
                    append_with_limit(&mut pre_roll, &samples, PRE_ROLL_SAMPLES);
                }

                if endpoint.observe(samples.len(), level) {
                    if args.verbose {
                        eprintln!("endpoint detected; committing current utterance");
                    }
                    commit_current_utterance(
                        &mut recognizer,
                        &mut transcript,
                        &mut stdout,
                        &current_audio,
                    )?;
                    endpoint.reset();
                    current_audio.clear();
                    pre_roll.clear();
                    last_preview = Instant::now();
                } else if endpoint.heard_speech()
                    && current_audio.len() >= MIN_PREVIEW_SAMPLES
                    && last_preview.elapsed() >= PREVIEW_INTERVAL
                {
                    let text = recognizer.transcribe_pcm_16k(&current_audio)?;
                    transcript.set_preview(text.trim(), &mut stdout)?;
                    last_preview = Instant::now();
                }
            }
            Err(mpsc::RecvTimeoutError::Timeout) => {}
            Err(mpsc::RecvTimeoutError::Disconnected) => break,
        }
    }

    if !current_audio.is_empty() {
        commit_current_utterance(
            &mut recognizer,
            &mut transcript,
            &mut stdout,
            &current_audio,
        )?;
    }
    transcript.finish(&mut stdout)?;
    Ok(())
}

fn commit_current_utterance(
    recognizer: &mut Recognizer,
    transcript: &mut TranscriptPrinter,
    stdout: &mut impl Write,
    audio: &[f32],
) -> Result<(), Box<dyn Error>> {
    if audio.len() < MIN_PREVIEW_SAMPLES {
        transcript.clear_preview(stdout)?;
        return Ok(());
    }

    let text = recognizer.transcribe_pcm_16k(audio)?;
    transcript.commit(text.trim(), stdout)?;
    Ok(())
}

fn append_with_limit(buffer: &mut Vec<f32>, samples: &[f32], max_len: usize) {
    buffer.extend_from_slice(samples);
    if buffer.len() > max_len {
        let extra = buffer.len() - max_len;
        buffer.drain(..extra);
    }
}

#[derive(Debug, Default)]
struct TranscriptPrinter {
    confirmed: String,
    preview: String,
}

impl TranscriptPrinter {
    fn new() -> Self {
        Self::default()
    }

    fn set_preview(&mut self, preview: &str, stdout: &mut impl Write) -> io::Result<()> {
        if preview.is_empty() || preview == self.preview {
            return Ok(());
        }
        self.preview.clear();
        self.preview.push_str(preview);
        self.render(stdout)
    }

    fn clear_preview(&mut self, stdout: &mut impl Write) -> io::Result<()> {
        if self.preview.is_empty() {
            return Ok(());
        }
        self.preview.clear();
        self.render(stdout)
    }

    fn commit(&mut self, text: &str, stdout: &mut impl Write) -> io::Result<()> {
        if text.is_empty() {
            return self.clear_preview(stdout);
        }
        self.confirmed.push_str(text);
        self.preview.clear();
        self.render(stdout)
    }

    fn finish(&mut self, stdout: &mut impl Write) -> io::Result<()> {
        self.preview.clear();
        self.render(stdout)?;
        writeln!(stdout)
    }

    fn render(&self, stdout: &mut impl Write) -> io::Result<()> {
        write!(stdout, "\r\x1b[2K{}{}", self.confirmed, self.preview)?;
        stdout.flush()
    }
}

#[derive(Debug)]
struct Args {
    models_dir: PathBuf,
    device: Option<String>,
    list_devices: bool,
    verbose: bool,
}

impl Args {
    fn parse(args: impl IntoIterator<Item = OsString>) -> Result<Self, Box<dyn Error>> {
        let mut models_dir = PathBuf::from("models");
        let mut device = None;
        let mut list_devices = false;
        let mut verbose = false;
        let mut positional = Vec::new();
        let mut args = args.into_iter();

        while let Some(arg) = args.next() {
            let arg = arg.to_string_lossy();
            match arg.as_ref() {
                "--help" | "-h" => {
                    print_usage();
                    std::process::exit(0);
                }
                "--list-devices" => list_devices = true,
                "--verbose" | "-v" => verbose = true,
                "--models" => {
                    let value = args.next().ok_or("--models requires a directory")?;
                    models_dir = PathBuf::from(value);
                }
                "--device" => {
                    let value = args
                        .next()
                        .ok_or("--device requires an index or device name")?;
                    device = Some(value.to_string_lossy().into_owned());
                }
                value if value.starts_with('-') => {
                    return Err(format!("unknown option: {value}").into());
                }
                _ => positional.push(OsString::from(arg.as_ref())),
            }
        }

        if let Some(first) = positional.first() {
            models_dir = PathBuf::from(first);
        }
        if positional.len() > 1 {
            return Err("too many positional arguments".into());
        }

        Ok(Self {
            models_dir,
            device,
            list_devices,
            verbose,
        })
    }
}

fn print_usage() {
    eprintln!(
        "usage: cargo run --example microphone_realtime -- [--models DIR] [--device INDEX_OR_NAME]\n\
         \n\
         options:\n\
           --list-devices           list available input devices\n\
           --device INDEX_OR_NAME   choose an input device by list index or name substring\n\
           --models DIR             model directory, default: models\n\
           --verbose, -v            print input levels and endpoint diagnostics to stderr\n\
         \n\
         legacy: a single positional argument is treated as the models directory"
    );
}

fn list_input_devices(host: &cpal::Host) -> Result<(), Box<dyn Error>> {
    let default_name = host
        .default_input_device()
        .and_then(|device| device.name().ok());
    let devices = input_devices(host)?;
    if devices.is_empty() {
        println!("no input devices found");
        return Ok(());
    }

    for (index, device) in devices.iter().enumerate() {
        let name = device
            .name()
            .unwrap_or_else(|_| "<unavailable name>".to_string());
        let marker = if Some(name.as_str()) == default_name.as_deref() {
            " default"
        } else {
            ""
        };
        match device.default_input_config() {
            Ok(config) => println!(
                "[{index}] {name}{marker} - {} Hz, {} channels, {:?}",
                config.sample_rate().0,
                config.channels(),
                config.sample_format()
            ),
            Err(error) => println!("[{index}] {name}{marker} - config unavailable: {error}"),
        }
    }
    Ok(())
}

fn select_input_device(
    host: &cpal::Host,
    selector: Option<&str>,
) -> Result<cpal::Device, Box<dyn Error>> {
    let devices = input_devices(host)?;
    if let Some(selector) = selector {
        if let Ok(index) = selector.parse::<usize>() {
            return devices
                .into_iter()
                .nth(index)
                .ok_or_else(|| format!("no input device at index {index}").into());
        }

        let selector = selector.to_ascii_lowercase();
        return devices
            .into_iter()
            .find(|device| {
                device
                    .name()
                    .map(|name| name.to_ascii_lowercase().contains(&selector))
                    .unwrap_or(false)
            })
            .ok_or_else(|| format!("no input device matching {selector:?}").into());
    }

    host.default_input_device()
        .ok_or_else(|| "no default input device available".into())
}

fn input_devices(host: &cpal::Host) -> Result<Vec<cpal::Device>, Box<dyn Error>> {
    Ok(host.input_devices()?.collect())
}

#[derive(Clone, Copy, Debug, Default)]
struct Level {
    rms: f32,
    peak: f32,
}

impl Level {
    fn from_samples(samples: &[f32]) -> Self {
        if samples.is_empty() {
            return Self::default();
        }

        let mut sum_squares = 0.0_f64;
        let mut peak = 0.0_f32;
        for sample in samples {
            let sample = sample.abs();
            peak = peak.max(sample);
            sum_squares += f64::from(sample * sample);
        }
        Self {
            rms: (sum_squares / samples.len() as f64).sqrt() as f32,
            peak,
        }
    }

    fn sounds_like_speech(self) -> bool {
        self.rms >= SPEECH_RMS_THRESHOLD || self.peak >= SPEECH_PEAK_THRESHOLD
    }
}

#[derive(Debug, Default)]
struct EndpointState {
    heard_speech: bool,
    silence_samples: usize,
    unflushed_samples: usize,
}

impl EndpointState {
    fn new() -> Self {
        Self::default()
    }

    fn observe(&mut self, sample_count: usize, level: Level) -> bool {
        if level.sounds_like_speech() {
            self.heard_speech = true;
            self.silence_samples = 0;
        } else if self.heard_speech {
            self.silence_samples += sample_count;
        }

        if self.heard_speech {
            self.unflushed_samples += sample_count;
        }

        self.heard_speech
            && (self.silence_samples >= AUTO_FLUSH_SILENCE
                || self.unflushed_samples >= MAX_UNFLUSHED_AUDIO)
    }

    fn reset(&mut self) {
        *self = Self::default();
    }

    fn heard_speech(&self) -> bool {
        self.heard_speech
    }
}

fn build_input_stream(
    device: &cpal::Device,
    config: &cpal::StreamConfig,
    sample_format: cpal::SampleFormat,
    input_sample_rate: u32,
    channels: usize,
    tx: mpsc::Sender<Vec<f32>>,
) -> Result<cpal::Stream, Box<dyn Error>> {
    match sample_format {
        cpal::SampleFormat::F32 => build_typed_input_stream::<f32, _>(
            device,
            config,
            input_sample_rate,
            channels,
            tx,
            |sample| sample,
        ),
        cpal::SampleFormat::F64 => build_typed_input_stream::<f64, _>(
            device,
            config,
            input_sample_rate,
            channels,
            tx,
            |sample| sample.clamp(-1.0, 1.0) as f32,
        ),
        cpal::SampleFormat::I8 => build_typed_input_stream::<i8, _>(
            device,
            config,
            input_sample_rate,
            channels,
            tx,
            |sample| (f32::from(sample) / f32::from(i8::MAX)).clamp(-1.0, 1.0),
        ),
        cpal::SampleFormat::I16 => build_typed_input_stream::<i16, _>(
            device,
            config,
            input_sample_rate,
            channels,
            tx,
            |sample| (f32::from(sample) / f32::from(i16::MAX)).clamp(-1.0, 1.0),
        ),
        cpal::SampleFormat::I32 => build_typed_input_stream::<i32, _>(
            device,
            config,
            input_sample_rate,
            channels,
            tx,
            |sample| (sample as f32 / i32::MAX as f32).clamp(-1.0, 1.0),
        ),
        cpal::SampleFormat::I64 => build_typed_input_stream::<i64, _>(
            device,
            config,
            input_sample_rate,
            channels,
            tx,
            |sample| (sample as f64 / i64::MAX as f64).clamp(-1.0, 1.0) as f32,
        ),
        cpal::SampleFormat::U8 => build_typed_input_stream::<u8, _>(
            device,
            config,
            input_sample_rate,
            channels,
            tx,
            |sample| (f32::from(sample) - 128.0) / 128.0,
        ),
        cpal::SampleFormat::U16 => build_typed_input_stream::<u16, _>(
            device,
            config,
            input_sample_rate,
            channels,
            tx,
            |sample| (f32::from(sample) - 32_768.0) / 32_768.0,
        ),
        cpal::SampleFormat::U32 => build_typed_input_stream::<u32, _>(
            device,
            config,
            input_sample_rate,
            channels,
            tx,
            |sample| ((sample as f64 - 2_147_483_648.0) / 2_147_483_648.0) as f32,
        ),
        cpal::SampleFormat::U64 => build_typed_input_stream::<u64, _>(
            device,
            config,
            input_sample_rate,
            channels,
            tx,
            |sample| {
                ((sample as f64 - 9_223_372_036_854_775_808.0) / 9_223_372_036_854_775_808.0) as f32
            },
        ),
        other => Err(format!("unsupported input sample format: {other:?}").into()),
    }
}

fn build_typed_input_stream<T, F>(
    device: &cpal::Device,
    config: &cpal::StreamConfig,
    input_sample_rate: u32,
    channels: usize,
    tx: mpsc::Sender<Vec<f32>>,
    convert: F,
) -> Result<cpal::Stream, Box<dyn Error>>
where
    T: cpal::SizedSample + Copy,
    F: Fn(T) -> f32 + Send + Copy + 'static,
{
    let mut resampler = DownmixResampler::new(f64::from(input_sample_rate), channels);
    let stream = device.build_input_stream(
        config,
        move |data: &[T], _| {
            let samples = resampler.push_interleaved(data, convert);
            if !samples.is_empty() {
                let _ = tx.send(samples);
            }
        },
        |error| eprintln!("audio input stream error: {error}"),
        None,
    )?;
    Ok(stream)
}

struct DownmixResampler {
    input_sample_rate: f64,
    channels: usize,
    mono: Vec<f32>,
    next_input_pos: f64,
}

impl DownmixResampler {
    fn new(input_sample_rate: f64, channels: usize) -> Self {
        Self {
            input_sample_rate,
            channels: channels.max(1),
            mono: Vec::new(),
            next_input_pos: 0.0,
        }
    }

    fn push_interleaved<T>(&mut self, input: &[T], convert: impl Fn(T) -> f32) -> Vec<f32>
    where
        T: Copy,
    {
        for frame in input.chunks_exact(self.channels) {
            let sum = frame
                .iter()
                .map(|sample| convert(*sample))
                .fold(0.0_f32, |acc, sample| acc + sample);
            self.mono.push(sum / self.channels as f32);
        }

        let mut output = Vec::new();
        let step = self.input_sample_rate / OUTPUT_SAMPLE_RATE;
        while self.next_input_pos + 1.0 < self.mono.len() as f64 {
            let index = self.next_input_pos.floor() as usize;
            let frac = (self.next_input_pos - index as f64) as f32;
            let current = self.mono[index];
            let next = self.mono[index + 1];
            output.push(current + (next - current) * frac);
            self.next_input_pos += step;
        }

        let len = self.mono.len();
        let consumed = if self.next_input_pos >= len as f64 {
            len
        } else {
            self.next_input_pos.floor() as usize
        };
        if consumed > 0 {
            self.mono.drain(..consumed);
            self.next_input_pos -= consumed as f64;
        }

        output
    }
}

#[cfg(test)]
mod tests {
    use super::{
        AUTO_FLUSH_SILENCE, Args, DownmixResampler, EndpointState, Level, TranscriptPrinter,
        append_with_limit,
    };
    use std::ffi::OsString;
    use std::path::PathBuf;

    #[test]
    fn resampler_does_not_panic_on_44100_stereo_512_frame_chunks() {
        let mut resampler = DownmixResampler::new(44_100.0, 2);
        let input = vec![0.0_f32; 512 * 2];

        for _ in 0..20 {
            let samples = resampler.push_interleaved(&input, |sample| sample);
            assert!(!samples.is_empty());
        }
    }

    #[test]
    fn endpoint_flushes_after_speech_then_silence() {
        let mut endpoint = EndpointState::new();

        assert!(!endpoint.observe(
            1_600,
            Level {
                rms: 0.02,
                peak: 0.1
            }
        ));
        assert!(endpoint.observe(
            AUTO_FLUSH_SILENCE,
            Level {
                rms: 0.0,
                peak: 0.0
            }
        ));
    }

    #[test]
    fn parses_device_and_models_options() {
        let args = Args::parse([
            OsString::from("--models"),
            OsString::from("custom-models"),
            OsString::from("--device"),
            OsString::from("1"),
        ])
        .unwrap();

        assert_eq!(args.models_dir, PathBuf::from("custom-models"));
        assert_eq!(args.device.as_deref(), Some("1"));
        assert!(!args.list_devices);
        assert!(!args.verbose);
    }

    #[test]
    fn parses_legacy_positional_models_dir() {
        let args = Args::parse([OsString::from("alt-models")]).unwrap();

        assert_eq!(args.models_dir, PathBuf::from("alt-models"));
        assert!(args.device.is_none());
        assert!(!args.list_devices);
        assert!(!args.verbose);
    }

    #[test]
    fn parses_verbose_flag() {
        let args = Args::parse([OsString::from("--verbose")]).unwrap();

        assert!(args.verbose);
    }

    #[test]
    fn append_with_limit_keeps_newest_samples() {
        let mut buffer = vec![1.0, 2.0, 3.0];

        append_with_limit(&mut buffer, &[4.0, 5.0, 6.0], 4);

        assert_eq!(buffer, vec![3.0, 4.0, 5.0, 6.0]);
    }

    #[test]
    fn transcript_printer_replaces_preview_and_commits_text() {
        let mut printer = TranscriptPrinter::new();
        let mut output = Vec::new();

        printer.set_preview("你好", &mut output).unwrap();
        printer.set_preview("你好世界", &mut output).unwrap();
        printer.commit("你好世界", &mut output).unwrap();

        let output = String::from_utf8(output).unwrap();
        assert!(output.ends_with("\r\u{1b}[2K你好世界"));
    }
}
