use std::error::Error;
use std::io::{self, Read, Write};

use sensevoice::RealtimeRecognizer;

const CHUNK_SAMPLES: usize = 1600;

fn main() -> Result<(), Box<dyn Error>> {
    let mut recognizer = RealtimeRecognizer::from_models_dir("models")?;
    let mut stdin = io::stdin().lock();
    let mut stdout = io::stdout().lock();
    let mut bytes = vec![0_u8; CHUNK_SAMPLES * 4];

    loop {
        let read = stdin.read(&mut bytes)?;
        if read == 0 {
            break;
        }

        let mut samples = Vec::with_capacity(read / 4);
        for chunk in bytes[..read].chunks_exact(4) {
            samples.push(f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]));
        }

        if let Some(text) = recognizer.accept_pcm_16k(&samples)? {
            write!(stdout, "{text}")?;
            stdout.flush()?;
        }
    }

    if let Some(text) = recognizer.flush()? {
        write!(stdout, "{text}")?;
    }
    writeln!(stdout)?;
    Ok(())
}
