// funasr-sensevoice: SenseVoiceSmall (SAN-M encoder + CTC) on ggml.
// This CLI is a thin wrapper around funasr_sensevoice::Model, which is also used
// by the Rust C ABI bridge.

#define FUNASR_AUDIO_IMPLEMENTATION
#include "funasr_audio.h"
#include "funasr_sensevoice.h"
#include "funasr_vad.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <utility>
#include <vector>

static void print_ids(const std::vector<int> & ids) {
    for (int id : ids) {
        printf("%d ", id);
    }
}

int main(int argc, char ** argv) {
    std::string gguf_path;
    std::string fbank_path;
    std::string wav_path;
    std::string vad_path;
    int vad_maxseg = 30000;
    bool ids_mode = false;
    bool keep_tags = false;

    for (int i = 1; i < argc; i++) {
        if (!strcmp(argv[i], "-m") && i + 1 < argc) {
            gguf_path = argv[++i];
        } else if (!strcmp(argv[i], "-f") && i + 1 < argc) {
            fbank_path = argv[++i];
        } else if (!strcmp(argv[i], "-a") && i + 1 < argc) {
            wav_path = argv[++i];
        } else if (!strcmp(argv[i], "--vad") && i + 1 < argc) {
            vad_path = argv[++i];
        } else if (!strcmp(argv[i], "--vad-maxseg") && i + 1 < argc) {
            vad_maxseg = atoi(argv[++i]);
        } else if (!strcmp(argv[i], "--ids")) {
            ids_mode = true;
        } else if (!strcmp(argv[i], "--keep-tags")) {
            keep_tags = true;
        } else {
            fprintf(stderr,
                    "usage: %s -m sensevoice.gguf (-a audio.wav | -f fbank.bin) "
                    "[--vad fsmn-vad.gguf [--vad-maxseg ms]] [--ids] [--keep-tags]\n",
                    argv[0]);
            return 1;
        }
    }
    if (gguf_path.empty() || (fbank_path.empty() && wav_path.empty())) {
        fprintf(stderr, "missing args\n");
        return 1;
    }

    funasr_sensevoice::Model model;
    std::string err;
    if (!model.load(gguf_path, &err)) {
        fprintf(stderr, "%s\n", err.c_str());
        return 1;
    }

    int64_t t0 = ggml_time_us();
    if (!vad_path.empty()) {
        std::vector<float> wav;
        if (!funasr_load_audio_16k_mono(wav_path.c_str(), wav)) {
            fprintf(stderr, "read audio failed\n");
            return 1;
        }
        std::vector<std::pair<int, int>> segs;
        if (!funasr_vad_segments(vad_path, wav, vad_maxseg, segs)) {
            fprintf(stderr, "vad failed\n");
            return 1;
        }
        for (auto & segment : segs) {
            int off = (int)((int64_t)segment.first * funasr_sensevoice::FS / 1000);
            int end = (int)((int64_t)segment.second * funasr_sensevoice::FS / 1000);
            if (end > (int)wav.size()) {
                end = (int)wav.size();
            }
            if (end - off < funasr_sensevoice::WINLEN) {
                continue;
            }
            std::vector<float> slice(wav.begin() + off, wav.begin() + end);
            if (ids_mode || !model.has_vocab()) {
                std::vector<int> ids;
                if (!model.transcribe_pcm_ids(slice, ids, 8, &err)) {
                    fprintf(stderr, "%s\n", err.c_str());
                    return 1;
                }
                print_ids(ids);
            } else {
                std::string text;
                if (!model.transcribe_pcm(slice, text, keep_tags, 8, &err)) {
                    fprintf(stderr, "%s\n", err.c_str());
                    return 1;
                }
                printf("%s", text.c_str());
            }
        }
        fprintf(stderr, "[sensevoice] %zu vad segments\n", segs.size());
    } else {
        int32_t frames = 0;
        int32_t feature_dim = 560;
        std::vector<float> fb;
        if (!wav_path.empty()) {
            std::vector<float> wav;
            if (!funasr_load_audio_16k_mono(wav_path.c_str(), wav)) {
                fprintf(stderr, "read audio failed\n");
                return 1;
            }
            fb = funasr_sensevoice::compute_fbank(wav, frames);
        } else {
            FILE * f = fopen(fbank_path.c_str(), "rb");
            if (!f) {
                fprintf(stderr, "open fbank\n");
                return 1;
            }
            if (fread(&frames, 4, 1, f) != 1 || fread(&feature_dim, 4, 1, f) != 1) {
                fclose(f);
                return 1;
            }
            fb.resize((size_t)frames * feature_dim);
            if ((int)fread(fb.data(), 4, fb.size(), f) != (int)fb.size()) {
                fclose(f);
                return 1;
            }
            fclose(f);
        }

        if (ids_mode || !model.has_vocab()) {
            std::vector<int> ids;
            if (!model.transcribe_fbank_ids(fb, frames, ids, 8, &err)) {
                fprintf(stderr, "%s\n", err.c_str());
                return 1;
            }
            print_ids(ids);
        } else {
            std::string text;
            if (!model.transcribe_fbank(fb, frames, text, keep_tags, 8, &err)) {
                fprintf(stderr, "%s\n", err.c_str());
                return 1;
            }
            printf("%s", text.c_str());
        }
    }

    printf("\n");
    fprintf(stderr, "[sensevoice] done %.2fs\n", (ggml_time_us() - t0) / 1e6);
    return 0;
}
