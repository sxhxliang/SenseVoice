#pragma once

#include <stddef.h>

#if defined(_WIN32)
#  if defined(FUNASR_RS_EXPORTS)
#    define FUNASR_RS_API __declspec(dllexport)
#  else
#    define FUNASR_RS_API __declspec(dllimport)
#  endif
#elif defined(__GNUC__) || defined(__clang__)
#  define FUNASR_RS_API __attribute__((visibility("default")))
#else
#  define FUNASR_RS_API
#endif

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

FUNASR_RS_API int sv_recognizer_new(const sv_config * config, sv_recognizer ** out);
FUNASR_RS_API void sv_recognizer_free(sv_recognizer * recognizer);

FUNASR_RS_API const char * sv_last_error(const sv_recognizer * recognizer);

FUNASR_RS_API int sv_recognize_file(sv_recognizer * recognizer, const char * audio_path, char ** out_text);
FUNASR_RS_API int sv_recognize_pcm(sv_recognizer * recognizer, const float * samples, size_t sample_count,
                                   char ** out_text);
FUNASR_RS_API int sv_vad_segments(sv_recognizer * recognizer, const float * samples, size_t sample_count,
                                  sv_segment ** out_segments, size_t * out_count);

FUNASR_RS_API int sv_realtime_accept_pcm(sv_recognizer * recognizer, const float * samples, size_t sample_count,
                                         char ** out_text);
FUNASR_RS_API int sv_realtime_flush(sv_recognizer * recognizer, char ** out_text);
FUNASR_RS_API void sv_realtime_reset(sv_recognizer * recognizer);

FUNASR_RS_API void sv_string_free(char * text);
FUNASR_RS_API void sv_segments_free(sv_segment * segments);

#ifdef __cplusplus
}
#endif
