#include "funasr_c_api.h"

#define FUNASR_AUDIO_IMPLEMENTATION
#include "funasr_audio.h"
#include "funasr_sensevoice.h"
#include "funasr_vad.h"

#include <algorithm>
#include <cstddef>
#include <cstdlib>
#include <cstring>
#include <exception>
#include <string>
#include <utility>
#include <vector>

struct sv_recognizer {
    funasr_sensevoice::Model asr;
    std::string vad_model;
    int n_threads = 8;
    int vad_max_seg_ms = 30000;
    int realtime_min_silence_ms = 600;
    int realtime_step_ms = 500;
    bool keep_tags = false;

    std::vector<float> stream;
    int committed_until_ms = 0;
    size_t last_analyzed_samples = 0;
    std::string last_error;
};

namespace {

constexpr int SAMPLE_RATE = 16000;
constexpr int STREAM_KEEP_BEHIND_MS = 5000;
constexpr int STREAM_PRUNE_AFTER_MS = 65000;

int samples_to_ms(size_t samples) {
    return (int)((samples * 1000) / SAMPLE_RATE);
}

size_t ms_to_samples(int ms) {
    if (ms <= 0) {
        return 0;
    }
    return (size_t)(((int64_t)ms * SAMPLE_RATE) / 1000);
}

int set_error(sv_recognizer * recognizer, const std::string & message) {
    if (recognizer) {
        recognizer->last_error = message;
    }
    return -1;
}

int alloc_string(const std::string & text, char ** out_text) {
    if (!out_text) {
        return -1;
    }
    char * ptr = (char *)malloc(text.size() + 1);
    if (!ptr) {
        *out_text = nullptr;
        return -1;
    }
    memcpy(ptr, text.data(), text.size());
    ptr[text.size()] = '\0';
    *out_text = ptr;
    return 0;
}

bool transcribe_slice(sv_recognizer * recognizer, const std::vector<float> & wav, int start_ms, int end_ms,
                      std::string & out_text, std::string & err) {
    int start = (int)ms_to_samples(start_ms);
    int end = (int)ms_to_samples(end_ms);
    if (start < 0) {
        start = 0;
    }
    if (end > (int)wav.size()) {
        end = (int)wav.size();
    }
    if (end - start < funasr_sensevoice::WINLEN) {
        return true;
    }
    std::vector<float> slice(wav.begin() + start, wav.begin() + end);
    std::string piece;
    if (!recognizer->asr.transcribe_pcm(slice, piece, recognizer->keep_tags, recognizer->n_threads, &err)) {
        return false;
    }
    out_text += piece;
    return true;
}

bool recognize_wav(sv_recognizer * recognizer, const std::vector<float> & wav, std::string & text,
                   std::string & err) {
    text.clear();
    if (recognizer->vad_model.empty()) {
        return recognizer->asr.transcribe_pcm(wav, text, recognizer->keep_tags, recognizer->n_threads, &err);
    }

    std::vector<std::pair<int, int>> segments;
    if (!funasr_vad_segments(recognizer->vad_model, wav, recognizer->vad_max_seg_ms, segments,
                             recognizer->n_threads)) {
        err = "VAD failed";
        return false;
    }
    for (const auto & segment : segments) {
        if (!transcribe_slice(recognizer, wav, segment.first, segment.second, text, err)) {
            return false;
        }
    }
    return true;
}

void prune_stream(sv_recognizer * recognizer) {
    if (recognizer->committed_until_ms <= STREAM_PRUNE_AFTER_MS) {
        return;
    }
    int drop_ms = recognizer->committed_until_ms - STREAM_KEEP_BEHIND_MS;
    size_t drop_samples = std::min(ms_to_samples(drop_ms), recognizer->stream.size());
    recognizer->stream.erase(recognizer->stream.begin(), recognizer->stream.begin() + (ptrdiff_t)drop_samples);
    recognizer->committed_until_ms -= drop_ms;
    recognizer->last_analyzed_samples =
        recognizer->last_analyzed_samples > drop_samples ? recognizer->last_analyzed_samples - drop_samples : 0;
}

bool process_ready_stream_segments(sv_recognizer * recognizer, bool final, std::string & text, std::string & err) {
    text.clear();
    if (recognizer->stream.empty()) {
        return true;
    }
    if (recognizer->vad_model.empty()) {
        if (!final) {
            return true;
        }
        bool ok = recognizer->asr.transcribe_pcm(recognizer->stream, text, recognizer->keep_tags,
                                                 recognizer->n_threads, &err);
        recognizer->stream.clear();
        recognizer->committed_until_ms = 0;
        recognizer->last_analyzed_samples = 0;
        return ok;
    }

    std::vector<std::pair<int, int>> segments;
    if (!funasr_vad_segments(recognizer->vad_model, recognizer->stream, recognizer->vad_max_seg_ms, segments,
                             recognizer->n_threads)) {
        err = "VAD failed";
        return false;
    }

    int total_ms = samples_to_ms(recognizer->stream.size());
    int ready_until_ms = final ? total_ms : total_ms - std::max(0, recognizer->realtime_min_silence_ms);
    for (const auto & segment : segments) {
        if (segment.second <= recognizer->committed_until_ms) {
            continue;
        }
        if (!final && segment.second > ready_until_ms) {
            continue;
        }

        int start_ms = std::max(segment.first, recognizer->committed_until_ms);
        int end_ms = std::min(segment.second, total_ms);
        if (end_ms <= start_ms) {
            continue;
        }
        if (!transcribe_slice(recognizer, recognizer->stream, start_ms, end_ms, text, err)) {
            return false;
        }
        recognizer->committed_until_ms = std::max(recognizer->committed_until_ms, segment.second);
    }

    if (final) {
        recognizer->stream.clear();
        recognizer->committed_until_ms = 0;
        recognizer->last_analyzed_samples = 0;
    } else {
        prune_stream(recognizer);
    }
    return true;
}

} // namespace

int sv_recognizer_new(const sv_config * config, sv_recognizer ** out) {
    if (!out) {
        return -1;
    }
    *out = nullptr;
    if (!config || !config->sensevoice_model) {
        return -1;
    }

    sv_recognizer * recognizer = nullptr;
    try {
        recognizer = new sv_recognizer();
        recognizer->vad_model = config->vad_model ? config->vad_model : "";
        recognizer->n_threads = config->n_threads > 0 ? config->n_threads : 8;
        recognizer->vad_max_seg_ms = config->vad_max_seg_ms > 0 ? config->vad_max_seg_ms : 30000;
        recognizer->realtime_min_silence_ms =
            config->realtime_min_silence_ms > 0 ? config->realtime_min_silence_ms : 600;
        recognizer->realtime_step_ms = config->realtime_step_ms > 0 ? config->realtime_step_ms : 500;
        recognizer->keep_tags = config->keep_tags != 0;

        std::string err;
        if (!recognizer->asr.load(config->sensevoice_model, &err)) {
            recognizer->last_error = err;
            delete recognizer;
            return -1;
        }
        *out = recognizer;
        return 0;
    } catch (const std::exception & e) {
        if (recognizer) {
            delete recognizer;
        }
        return -1;
    }
}

void sv_recognizer_free(sv_recognizer * recognizer) {
    delete recognizer;
}

const char * sv_last_error(const sv_recognizer * recognizer) {
    if (!recognizer) {
        return "null recognizer";
    }
    return recognizer->last_error.c_str();
}

int sv_recognize_file(sv_recognizer * recognizer, const char * audio_path, char ** out_text) {
    if (!recognizer || !audio_path) {
        return -1;
    }
    try {
        std::vector<float> wav;
        if (!funasr_load_audio_16k_mono(audio_path, wav)) {
            return set_error(recognizer, "failed to decode audio file");
        }
        std::string text;
        std::string err;
        if (!recognize_wav(recognizer, wav, text, err)) {
            return set_error(recognizer, err);
        }
        return alloc_string(text, out_text);
    } catch (const std::exception & e) {
        return set_error(recognizer, e.what());
    }
}

int sv_recognize_pcm(sv_recognizer * recognizer, const float * samples, size_t sample_count, char ** out_text) {
    if (!recognizer || (!samples && sample_count > 0)) {
        return -1;
    }
    try {
        std::vector<float> wav;
        if (sample_count > 0) {
            wav.assign(samples, samples + sample_count);
        }
        std::string text;
        std::string err;
        if (!recognize_wav(recognizer, wav, text, err)) {
            return set_error(recognizer, err);
        }
        return alloc_string(text, out_text);
    } catch (const std::exception & e) {
        return set_error(recognizer, e.what());
    }
}

int sv_vad_segments(sv_recognizer * recognizer, const float * samples, size_t sample_count,
                    sv_segment ** out_segments, size_t * out_count) {
    if (!recognizer || (!samples && sample_count > 0) || !out_segments || !out_count) {
        return -1;
    }
    *out_segments = nullptr;
    *out_count = 0;
    if (recognizer->vad_model.empty()) {
        return set_error(recognizer, "VAD model is not configured");
    }

    try {
        std::vector<float> wav;
        if (sample_count > 0) {
            wav.assign(samples, samples + sample_count);
        }
        std::vector<std::pair<int, int>> segments;
        if (!funasr_vad_segments(recognizer->vad_model, wav, recognizer->vad_max_seg_ms, segments,
                                 recognizer->n_threads)) {
            return set_error(recognizer, "VAD failed");
        }
        if (segments.empty()) {
            return 0;
        }
        sv_segment * out = (sv_segment *)malloc(sizeof(sv_segment) * segments.size());
        if (!out) {
            return set_error(recognizer, "failed to allocate VAD segment buffer");
        }
        for (size_t i = 0; i < segments.size(); i++) {
            out[i].start_ms = segments[i].first;
            out[i].end_ms = segments[i].second;
        }
        *out_segments = out;
        *out_count = segments.size();
        return 0;
    } catch (const std::exception & e) {
        return set_error(recognizer, e.what());
    }
}

int sv_realtime_accept_pcm(sv_recognizer * recognizer, const float * samples, size_t sample_count,
                           char ** out_text) {
    if (!recognizer || (!samples && sample_count > 0)) {
        return -1;
    }
    try {
        if (sample_count > 0) {
            recognizer->stream.insert(recognizer->stream.end(), samples, samples + sample_count);
        }
        size_t step_samples = ms_to_samples(recognizer->realtime_step_ms);
        if (recognizer->stream.size() - recognizer->last_analyzed_samples < step_samples) {
            return alloc_string("", out_text);
        }
        recognizer->last_analyzed_samples = recognizer->stream.size();

        std::string text;
        std::string err;
        if (!process_ready_stream_segments(recognizer, false, text, err)) {
            return set_error(recognizer, err);
        }
        return alloc_string(text, out_text);
    } catch (const std::exception & e) {
        return set_error(recognizer, e.what());
    }
}

int sv_realtime_flush(sv_recognizer * recognizer, char ** out_text) {
    if (!recognizer) {
        return -1;
    }
    try {
        std::string text;
        std::string err;
        if (!process_ready_stream_segments(recognizer, true, text, err)) {
            return set_error(recognizer, err);
        }
        return alloc_string(text, out_text);
    } catch (const std::exception & e) {
        return set_error(recognizer, e.what());
    }
}

void sv_realtime_reset(sv_recognizer * recognizer) {
    if (!recognizer) {
        return;
    }
    recognizer->stream.clear();
    recognizer->committed_until_ms = 0;
    recognizer->last_analyzed_samples = 0;
}

void sv_string_free(char * text) {
    free(text);
}

void sv_segments_free(sv_segment * segments) {
    free(segments);
}
