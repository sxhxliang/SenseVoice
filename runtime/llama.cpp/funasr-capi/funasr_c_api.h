#pragma once

#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct sv_recognizer sv_recognizer;

typedef struct sv_config {
    const char * sensevoice_model;
    const char * vad_model;
    int n_threads;
    int vad_max_seg_ms;
    int realtime_min_silence_ms;
    int realtime_step_ms;
    int keep_tags;
} sv_config;

typedef struct sv_segment {
    int start_ms;
    int end_ms;
} sv_segment;

int sv_recognizer_new(const sv_config * config, sv_recognizer ** out);
void sv_recognizer_free(sv_recognizer * recognizer);

const char * sv_last_error(const sv_recognizer * recognizer);

int sv_recognize_file(sv_recognizer * recognizer, const char * audio_path, char ** out_text);
int sv_recognize_pcm(sv_recognizer * recognizer, const float * samples, size_t sample_count, char ** out_text);
int sv_vad_segments(sv_recognizer * recognizer, const float * samples, size_t sample_count,
                    sv_segment ** out_segments, size_t * out_count);

int sv_realtime_accept_pcm(sv_recognizer * recognizer, const float * samples, size_t sample_count,
                           char ** out_text);
int sv_realtime_flush(sv_recognizer * recognizer, char ** out_text);
void sv_realtime_reset(sv_recognizer * recognizer);

void sv_string_free(char * text);
void sv_segments_free(sv_segment * segments);

#ifdef __cplusplus
}
#endif
