// funasr_sensevoice.h - reusable SenseVoiceSmall ggml runtime.
// Input is 16 kHz mono f32 PCM in [-1, 1]; output is greedy-CTC detokenized text.
#pragma once

#include "ggml.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml-cpu.h"
#include "gguf.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <map>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

namespace funasr_sensevoice {

static constexpr float LN_EPS = 1e-5f;
static constexpr int FS = 16000;
static constexpr int WINLEN = 400;
static constexpr int SHIFT = 160;
static constexpr int NFFT = 512;
static constexpr int NMEL = 80;
static constexpr int LFR_M = 7;
static constexpr int LFR_N = 6;
static constexpr float PREEMPH = 0.97f;
static constexpr float LOWF = 20.0f;
static constexpr float HIGHF = 8000.0f;

struct Config {
    int d_model = 512;
    int n_head = 4;
    int num_blocks = 50;
    int tp_blocks = 20;
    int kernel = 11;
    int vocab = 25055;
    int blank = 0;
};

inline void set_error(std::string * error, const std::string & message) {
    if (error) {
        *error = message;
    }
}

inline float melf(float f) {
    return 1127.0f * logf(1.0f + f / 700.0f);
}

inline void fftc(std::vector<float> & re, std::vector<float> & im, int n) {
    for (int i = 1, j = 0; i < n; i++) {
        int b = n >> 1;
        for (; j & b; b >>= 1) {
            j ^= b;
        }
        j ^= b;
        if (i < j) {
            std::swap(re[i], re[j]);
            std::swap(im[i], im[j]);
        }
    }
    for (int len = 2; len <= n; len <<= 1) {
        double a = -2.0 * M_PI / len;
        float wr = cosf(a);
        float wi = sinf(a);
        for (int i = 0; i < n; i += len) {
            float cr = 1.0f;
            float ci = 0.0f;
            for (int k = 0; k < len / 2; k++) {
                float ur = re[i + k];
                float ui = im[i + k];
                float vr = re[i + k + len / 2] * cr - im[i + k + len / 2] * ci;
                float vi = re[i + k + len / 2] * ci + im[i + k + len / 2] * cr;
                re[i + k] = ur + vr;
                im[i + k] = ui + vi;
                re[i + k + len / 2] = ur - vr;
                im[i + k + len / 2] = ui - vi;
                float nc = cr * wr - ci * wi;
                ci = cr * wi + ci * wr;
                cr = nc;
            }
        }
    }
}

inline std::vector<float> compute_fbank(std::vector<float> wav, int & frames_out) {
    frames_out = 0;
    if ((int)wav.size() < WINLEN) {
        return {};
    }

    for (float & v : wav) {
        v *= 32768.0f;
    }

    std::vector<float> win(WINLEN);
    for (int i = 0; i < WINLEN; i++) {
        win[i] = 0.54f - 0.46f * cosf(2.0f * M_PI * i / (WINLEN - 1));
    }

    const int nbin = NFFT / 2 + 1;
    float bw = (float)FS / NFFT;
    float ml = melf(LOWF);
    float mh = melf(HIGHF);
    float dm = (mh - ml) / (NMEL + 1);
    std::vector<std::vector<float>> fb(NMEL, std::vector<float>(nbin, 0.0f));
    for (int m = 0; m < NMEL; m++) {
        float left = ml + m * dm;
        float center = ml + (m + 1) * dm;
        float right = ml + (m + 2) * dm;
        for (int k = 0; k < nbin; k++) {
            float mf = melf(bw * k);
            if (mf > left && mf < right) {
                fb[m][k] = mf <= center ? (mf - left) / (center - left) : (right - mf) / (right - center);
            }
        }
    }

    int n = (int)wav.size();
    int t_frames = (n - WINLEN) / SHIFT + 1;
    if (t_frames < 1) {
        return {};
    }

    std::vector<std::vector<float>> feat(t_frames, std::vector<float>(NMEL));
    std::vector<float> re(NFFT);
    std::vector<float> im(NFFT);
    std::vector<float> fr(WINLEN);
    const float floor = 1.1920929e-07f;
    for (int t = 0; t < t_frames; t++) {
        const float * s = wav.data() + t * SHIFT;
        double mean = 0.0;
        for (int i = 0; i < WINLEN; i++) {
            mean += s[i];
        }
        mean /= WINLEN;

        for (int i = 0; i < WINLEN; i++) {
            fr[i] = s[i] - (float)mean;
        }
        for (int i = WINLEN - 1; i > 0; i--) {
            fr[i] -= PREEMPH * fr[i - 1];
        }
        fr[0] -= PREEMPH * fr[0];

        for (int i = 0; i < NFFT; i++) {
            re[i] = i < WINLEN ? fr[i] * win[i] : 0.0f;
            im[i] = 0.0f;
        }
        fftc(re, im, NFFT);

        for (int m = 0; m < NMEL; m++) {
            float e = 0.0f;
            for (int k = 0; k < nbin; k++) {
                if (fb[m][k] > 0.0f) {
                    e += fb[m][k] * (re[k] * re[k] + im[k] * im[k]);
                }
            }
            feat[t][m] = logf(e > floor ? e : floor);
        }
    }

    const int pad = (LFR_M - 1) / 2;
    int lfr_frames = (t_frames + LFR_N - 1) / LFR_N;
    std::vector<std::vector<float>> padded;
    padded.reserve(t_frames + pad + LFR_M);
    for (int i = 0; i < pad; i++) {
        padded.push_back(feat[0]);
    }
    for (int t = 0; t < t_frames; t++) {
        padded.push_back(feat[t]);
    }
    while ((int)padded.size() < (lfr_frames - 1) * LFR_N + LFR_M) {
        padded.push_back(feat[t_frames - 1]);
    }

    int dims = LFR_M * NMEL;
    std::vector<float> out((size_t)lfr_frames * dims);
    for (int i = 0; i < lfr_frames; i++) {
        for (int j = 0; j < LFR_M; j++) {
            memcpy(&out[(size_t)i * dims + j * NMEL], padded[i * LFR_N + j].data(), NMEL * sizeof(float));
        }
    }
    frames_out = lfr_frames;
    return out;
}

inline std::string trim(const std::string & s) {
    size_t a = s.find_first_not_of(' ');
    if (a == std::string::npos) {
        return "";
    }
    size_t b = s.find_last_not_of(' ');
    return s.substr(a, b - a + 1);
}

inline std::string detok(const std::vector<int> & ids, const std::vector<std::string> & vocab, bool keep_tags) {
    std::string text;
    for (int id : ids) {
        if (id < 0 || id >= (int)vocab.size()) {
            continue;
        }
        const std::string & piece = vocab[id];
        if (!keep_tags && piece.size() >= 2 && piece[0] == '<' && piece[1] == '|') {
            continue;
        }
        text += piece;
    }
    const std::string marker = "\xe2\x96\x81";
    size_t pos = 0;
    while ((pos = text.find(marker)) != std::string::npos) {
        text.replace(pos, marker.size(), " ");
    }
    return trim(text);
}

class Model {
public:
    Model() = default;

    ~Model() {
        reset();
    }

    Model(const Model &) = delete;
    Model & operator=(const Model &) = delete;

    Model(Model && other) noexcept {
        move_from(std::move(other));
    }

    Model & operator=(Model && other) noexcept {
        if (this != &other) {
            reset();
            move_from(std::move(other));
        }
        return *this;
    }

    bool load(const std::string & gguf_path, std::string * error = nullptr) {
        reset();
        gguf_init_params gp = { false, &ctx_w_ };
        gguf_context * gg = gguf_init_from_file(gguf_path.c_str(), gp);
        if (!gg) {
            set_error(error, "failed to load SenseVoice GGUF: " + gguf_path);
            ctx_w_ = nullptr;
            return false;
        }

        auto rd = [&](const char * key, int fallback) {
            int i = gguf_find_key(gg, key);
            return i < 0 ? fallback : (int)gguf_get_val_u32(gg, i);
        };
        cfg_.d_model = rd("sv.output_size", 512);
        cfg_.n_head = rd("sv.attention_heads", 4);
        cfg_.num_blocks = rd("sv.num_blocks", 50);
        cfg_.tp_blocks = rd("sv.tp_blocks", 20);
        cfg_.kernel = rd("sv.kernel_size", 11);
        cfg_.vocab = rd("sv.vocab_size", 25055);
        cfg_.blank = rd("sv.blank_id", 0);

        int qi = gguf_find_key(gg, "sv.query_tokens");
        int nq = qi < 0 ? 0 : (int)gguf_get_arr_n(gg, qi);
        query_tokens_.resize(nq);
        for (int i = 0; i < nq; i++) {
            query_tokens_[i] = ((const int32_t *)gguf_get_arr_data(gg, qi))[i];
        }

        int vi = gguf_find_key(gg, "sv.vocab");
        if (vi >= 0) {
            int nv = (int)gguf_get_arr_n(gg, vi);
            vocab_.resize(nv);
            for (int i = 0; i < nv; i++) {
                const char * s = gguf_get_arr_str(gg, vi, i);
                vocab_[i] = s ? s : "";
            }
        }

        for (int i = 0; i < gguf_get_n_tensors(gg); i++) {
            const char * name = gguf_get_tensor_name(gg, i);
            tensors_[name] = ggml_get_tensor(ctx_w_, name);
        }
        gguf_free(gg);

        if (!tensor_or_null("embed.weight")) {
            reset();
            set_error(error, "SenseVoice GGUF is missing embed.weight");
            return false;
        }
        loaded_ = true;
        return true;
    }

    bool is_loaded() const {
        return loaded_;
    }

    bool has_vocab() const {
        return !vocab_.empty();
    }

    bool transcribe_pcm(const std::vector<float> & wav, std::string & text, bool keep_tags = false,
                        int n_threads = 8, std::string * error = nullptr) {
        int frames = 0;
        std::vector<float> fb = compute_fbank(wav, frames);
        return transcribe_fbank(fb, frames, text, keep_tags, n_threads, error);
    }

    bool transcribe_pcm_ids(const std::vector<float> & wav, std::vector<int> & ids, int n_threads = 8,
                            std::string * error = nullptr) {
        int frames = 0;
        std::vector<float> fb = compute_fbank(wav, frames);
        return transcribe_fbank_ids(fb, frames, ids, n_threads, error);
    }

    bool transcribe_fbank(const std::vector<float> & fb, int frames, std::string & text, bool keep_tags = false,
                          int n_threads = 8, std::string * error = nullptr) {
        std::vector<int> ids;
        if (!decode_fbank(fb, frames, ids, n_threads, error)) {
            return false;
        }
        text = detok(ids, vocab_, keep_tags);
        return true;
    }

    bool transcribe_fbank_ids(const std::vector<float> & fb, int frames, std::vector<int> & ids, int n_threads = 8,
                              std::string * error = nullptr) {
        return decode_fbank(fb, frames, ids, n_threads, error);
    }

private:
    struct BackendGuard {
        ggml_backend_t ptr = nullptr;
        explicit BackendGuard(ggml_backend_t p) : ptr(p) {}
        ~BackendGuard() {
            if (ptr) {
                ggml_backend_free(ptr);
            }
        }
        operator ggml_backend_t() const {
            return ptr;
        }
    };

    struct ContextGuard {
        ggml_context * ptr = nullptr;
        explicit ContextGuard(ggml_context * p) : ptr(p) {}
        ~ContextGuard() {
            if (ptr) {
                ggml_free(ptr);
            }
        }
        operator ggml_context *() const {
            return ptr;
        }
    };

    struct AllocGuard {
        ggml_gallocr_t ptr = nullptr;
        explicit AllocGuard(ggml_gallocr_t p) : ptr(p) {}
        ~AllocGuard() {
            if (ptr) {
                ggml_gallocr_free(ptr);
            }
        }
        operator ggml_gallocr_t() const {
            return ptr;
        }
    };

    void reset() {
        if (ctx_w_) {
            ggml_free(ctx_w_);
            ctx_w_ = nullptr;
        }
        tensors_.clear();
        vocab_.clear();
        query_tokens_.clear();
        cfg_ = Config{};
        loaded_ = false;
    }

    void move_from(Model && other) {
        ctx_w_ = other.ctx_w_;
        tensors_ = std::move(other.tensors_);
        vocab_ = std::move(other.vocab_);
        query_tokens_ = std::move(other.query_tokens_);
        cfg_ = other.cfg_;
        loaded_ = other.loaded_;
        other.ctx_w_ = nullptr;
        other.loaded_ = false;
    }

    ggml_tensor * tensor_or_null(const std::string & name) const {
        auto it = tensors_.find(name);
        if (it == tensors_.end()) {
            return nullptr;
        }
        return it->second;
    }

    ggml_tensor * tensor(const std::string & name) const {
        ggml_tensor * t = tensor_or_null(name);
        if (!t) {
            throw std::runtime_error("SenseVoice GGUF is missing tensor: " + name);
        }
        return t;
    }

    static ggml_tensor * lin(ggml_context * ctx, ggml_tensor * w, ggml_tensor * b, ggml_tensor * x) {
        ggml_tensor * y = ggml_mul_mat(ctx, w, x);
        return b ? ggml_add(ctx, y, b) : y;
    }

    ggml_tensor * lnorm(ggml_context * ctx, ggml_tensor * x, const std::string & g, const std::string & b) const {
        return ggml_add(ctx, ggml_mul(ctx, ggml_norm(ctx, x, LN_EPS), tensor(g)), tensor(b));
    }

    ggml_tensor * sanm_attn(ggml_context * ctx, const std::string & prefix, ggml_tensor * x, int frames) const {
        const int d = cfg_.d_model;
        const int h = cfg_.n_head;
        const int dk = d / h;
        const int kernel = cfg_.kernel;
        ggml_tensor * qkv = lin(ctx, tensor(prefix + "linear_q_k_v.weight"), tensor(prefix + "linear_q_k_v.bias"), x);
        size_t nb1 = qkv->nb[1];
        ggml_tensor * q = ggml_cont(ctx, ggml_view_2d(ctx, qkv, d, frames, nb1, 0));
        ggml_tensor * k = ggml_cont(ctx, ggml_view_2d(ctx, qkv, d, frames, nb1, (size_t)d * sizeof(float)));
        ggml_tensor * v = ggml_cont(ctx, ggml_view_2d(ctx, qkv, d, frames, nb1, (size_t)2 * d * sizeof(float)));

        const int pad = (kernel - 1) / 2;
        ggml_tensor * fk = tensor(prefix + "fsmn_block.weight");
        ggml_tensor * vp = ggml_pad_ext(ctx, v, 0, 0, pad, pad, 0, 0, 0, 0);
        ggml_tensor * fsmn = v;
        for (int j = 0; j < kernel; j++) {
            ggml_tensor * sl = ggml_view_2d(ctx, vp, d, frames, vp->nb[1], (size_t)j * vp->nb[1]);
            ggml_tensor * wj = ggml_view_1d(ctx, fk, d, (size_t)j * fk->nb[1]);
            fsmn = ggml_add(ctx, fsmn, ggml_mul(ctx, ggml_cont(ctx, sl), wj));
        }

        q = ggml_permute(ctx, ggml_reshape_3d(ctx, q, dk, h, frames), 0, 2, 1, 3);
        k = ggml_permute(ctx, ggml_reshape_3d(ctx, k, dk, h, frames), 0, 2, 1, 3);
        ggml_tensor * vh = ggml_cont(ctx, ggml_permute(ctx, ggml_reshape_3d(ctx, v, dk, h, frames), 1, 2, 0, 3));
        ggml_tensor * kq = ggml_soft_max(ctx, ggml_scale(ctx, ggml_mul_mat(ctx, k, q), 1.0f / sqrtf((float)dk)));
        ggml_tensor * o = ggml_cont_2d(ctx, ggml_permute(ctx, ggml_mul_mat(ctx, vh, kq), 0, 2, 1, 3), d, frames);
        return ggml_add(ctx, lin(ctx, tensor(prefix + "linear_out.weight"), tensor(prefix + "linear_out.bias"), o), fsmn);
    }

    ggml_tensor * sanm_layer(ggml_context * ctx, const std::string & prefix, ggml_tensor * x, int frames,
                             bool residual) const {
        ggml_tensor * r = x;
        ggml_tensor * h = lnorm(ctx, x, prefix + "norm1.weight", prefix + "norm1.bias");
        ggml_tensor * sa = sanm_attn(ctx, prefix + "self_attn.", h, frames);
        x = residual ? ggml_add(ctx, r, sa) : sa;
        r = x;
        h = lnorm(ctx, x, prefix + "norm2.weight", prefix + "norm2.bias");
        h = lin(ctx, tensor(prefix + "feed_forward.w_1.weight"), tensor(prefix + "feed_forward.w_1.bias"), h);
        h = ggml_relu(ctx, h);
        h = lin(ctx, tensor(prefix + "feed_forward.w_2.weight"), tensor(prefix + "feed_forward.w_2.bias"), h);
        return ggml_add(ctx, r, h);
    }

    static void add_posenc(std::vector<float> & x, int frames, int depth) {
        double inc = log(10000.0) / (depth / 2.0 - 1.0);
        for (int t = 0; t < frames; t++) {
            double pos = t + 1;
            for (int i = 0; i < depth / 2; i++) {
                double its = exp(i * -inc);
                double st = pos * its;
                x[(size_t)t * depth + i] += (float)sin(st);
                x[(size_t)t * depth + depth / 2 + i] += (float)cos(st);
            }
        }
    }

    bool decode_fbank(const std::vector<float> & fb, int frames, std::vector<int> & ids, int n_threads,
                      std::string * error) {
        ids.clear();
        if (!loaded_) {
            set_error(error, "SenseVoice model is not loaded");
            return false;
        }
        if (frames < 1) {
            return true;
        }
        if ((int)fb.size() < frames * 560) {
            set_error(error, "fbank buffer is smaller than frames * 560");
            return false;
        }

        try {
            const int feature_dim = 560;
            const int d_model = cfg_.d_model;
            const int vocab = cfg_.vocab;
            ggml_tensor * embed_tensor = tensor("embed.weight");
            float * emb = (float *)embed_tensor->data;

            int query_count = (int)query_tokens_.size();
            int n = query_count + frames;
            std::vector<float> input((size_t)n * feature_dim);
            for (int i = 0; i < query_count; i++) {
                memcpy(&input[(size_t)i * feature_dim], &emb[(size_t)query_tokens_[i] * feature_dim],
                       feature_dim * sizeof(float));
            }
            memcpy(&input[(size_t)query_count * feature_dim], fb.data(), (size_t)frames * feature_dim * sizeof(float));

            float scale = sqrtf((float)d_model);
            for (float & v : input) {
                v *= scale;
            }
            add_posenc(input, n, feature_dim);

            BackendGuard backend(ggml_backend_cpu_init());
            if (!backend.ptr) {
                set_error(error, "failed to initialize ggml CPU backend");
                return false;
            }
            ggml_init_params cp = { (size_t)1024 * 1024 * 1024, nullptr, true };
            ContextGuard ctx(ggml_init(cp));
            if (!ctx.ptr) {
                set_error(error, "failed to initialize ggml context");
                return false;
            }

            ggml_tensor * x = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, feature_dim, n);
            ggml_set_input(x);
            ggml_tensor * h = sanm_layer(ctx, "encoder.encoders0.0.", x, n, false);
            for (int i = 0; i < cfg_.num_blocks - 1; i++) {
                h = sanm_layer(ctx, "encoder.encoders." + std::to_string(i) + ".", h, n, true);
            }
            h = lnorm(ctx, h, "encoder.after_norm.weight", "encoder.after_norm.bias");
            for (int i = 0; i < cfg_.tp_blocks; i++) {
                h = sanm_layer(ctx, "encoder.tp_encoders." + std::to_string(i) + ".", h, n, true);
            }
            h = lnorm(ctx, h, "encoder.tp_norm.weight", "encoder.tp_norm.bias");
            ggml_tensor * logits = lin(ctx, tensor("ctc.ctc_lo.weight"), tensor("ctc.ctc_lo.bias"), h);
            ggml_set_output(logits);

            ggml_cgraph * graph = ggml_new_graph_custom(ctx, 32768, false);
            ggml_build_forward_expand(graph, logits);
            AllocGuard alloc(ggml_gallocr_new(ggml_backend_cpu_buffer_type()));
            ggml_gallocr_alloc_graph(alloc, graph);
            ggml_backend_tensor_set(x, input.data(), 0, ggml_nbytes(x));
            ggml_backend_cpu_set_n_threads(backend, std::max(1, n_threads));
            if (ggml_backend_graph_compute(backend, graph) != GGML_STATUS_SUCCESS) {
                set_error(error, "ggml graph compute failed");
                return false;
            }

            std::vector<float> lg((size_t)vocab * n);
            ggml_backend_tensor_get(logits, lg.data(), 0, ggml_nbytes(logits));
            int prev = -1;
            for (int frame = 0; frame < n; frame++) {
                const float * col = &lg[(size_t)frame * vocab];
                int argmax = 0;
                float best = col[0];
                for (int v = 1; v < vocab; v++) {
                    if (col[v] > best) {
                        best = col[v];
                        argmax = v;
                    }
                }
                if (argmax != prev && argmax != cfg_.blank) {
                    ids.push_back(argmax);
                }
                prev = argmax;
            }
            return true;
        } catch (const std::exception & e) {
            set_error(error, e.what());
            return false;
        }
    }

    Config cfg_;
    ggml_context * ctx_w_ = nullptr;
    std::map<std::string, ggml_tensor *> tensors_;
    std::vector<std::string> vocab_;
    std::vector<int> query_tokens_;
    bool loaded_ = false;
};

} // namespace funasr_sensevoice
