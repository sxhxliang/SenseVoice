// funasr_audio.h — load any audio file (WAV/MP3/FLAC/OGG...) as 16 kHz mono f32,
// via miniaudio's decoder (handles arbitrary sample rate, channels, bit depth).
// One TU must define FUNASR_AUDIO_IMPLEMENTATION before including.
#ifndef FUNASR_AUDIO_H
#define FUNASR_AUDIO_H
#include <vector>
// returns true on success; out = mono 16 kHz float samples in [-1,1]
inline bool funasr_load_audio_16k_mono(const char * path, std::vector<float> & out);
#endif

#ifdef FUNASR_AUDIO_IMPLEMENTATION
// keep only the decoder + data conversion (resampler/channel mixer); drop playback/capture
#define MA_NO_DEVICE_IO
#define MA_NO_ENGINE
#define MA_NO_GENERATION
#define MA_NO_THREADING
#define MINIAUDIO_IMPLEMENTATION
#include "miniaudio.h"
#include <cstdio>
#ifdef _WIN32
#include <windows.h>
#endif

#ifdef _WIN32
inline bool funasr_utf8_to_wide(const char * path, std::vector<wchar_t> & wide_path) {
    int length = MultiByteToWideChar(CP_UTF8, MB_ERR_INVALID_CHARS, path, -1, nullptr, 0);
    if (length <= 0) {
        return false;
    }
    wide_path.resize((size_t)length);
    return MultiByteToWideChar(CP_UTF8, MB_ERR_INVALID_CHARS, path, -1, wide_path.data(), length) > 0;
}
#endif

inline bool funasr_load_audio_16k_mono(const char * path, std::vector<float> & out) {
    if (!path) { fprintf(stderr, "audio: null path\n"); return false; }
    ma_decoder_config cfg = ma_decoder_config_init(ma_format_f32, 1, 16000); // f32, mono, 16k
    ma_decoder dec;
    ma_result init_result;
#ifdef _WIN32
    std::vector<wchar_t> wide_path;
    if (!funasr_utf8_to_wide(path, wide_path)) {
        fprintf(stderr, "audio: path is not valid UTF-8\n");
        return false;
    }
    init_result = ma_decoder_init_file_w(wide_path.data(), &cfg, &dec);
#else
    init_result = ma_decoder_init_file(path, &cfg, &dec);
#endif
    if (init_result != MA_SUCCESS) {
        fprintf(stderr, "audio: failed to open/decode %s (supported: wav/mp3/flac)\n", path);
        return false;
    }
    out.clear();
    ma_uint64 nframes = 0;
    if (ma_decoder_get_length_in_pcm_frames(&dec, &nframes) == MA_SUCCESS && nframes > 0) {
        out.resize(nframes);
        ma_uint64 got = 0;
        ma_result r = ma_decoder_read_pcm_frames(&dec, out.data(), nframes, &got);
        if (r != MA_SUCCESS && r != MA_AT_END) { ma_decoder_uninit(&dec); fprintf(stderr, "audio: decode error\n"); return false; }
        out.resize(got);
    } else {
        // length unknown (e.g. some mp3 streams): read in chunks until EOF
        std::vector<float> buf(16000);
        for (;;) {
            ma_uint64 got = 0;
            ma_result r = ma_decoder_read_pcm_frames(&dec, buf.data(), buf.size(), &got);
            out.insert(out.end(), buf.begin(), buf.begin() + got);
            if (got < buf.size() || (r != MA_SUCCESS && r != MA_AT_END)) break;
        }
    }
    ma_decoder_uninit(&dec);
    return !out.empty();
}
#endif
