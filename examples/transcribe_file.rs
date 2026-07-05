use std::env;
use std::error::Error;

use sensevoice::Recognizer;

fn main() -> Result<(), Box<dyn Error>> {
    let audio_path = env::args_os()
        .nth(1)
        .ok_or("usage: cargo run --example transcribe_file -- <audio.wav>")?;
    let mut recognizer = Recognizer::from_models_dir("models")?;
    let text = recognizer.transcribe_file(audio_path)?;
    println!("{text}");
    Ok(())
}
