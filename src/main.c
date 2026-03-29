#define _POSIX_C_SOURCE 200809L
#include <ctype.h>
#include <math.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <sys/stat.h>

typedef enum {
    MODEL_MLP = 0,
    MODEL_LSTM = 1,
    MODEL_TRANSFORMER = 2,
    MODEL_MAMBA = 3,
    MODEL_ALL = 4
} ModelType;

typedef struct {
    int *tokens;
    int length;
    int vocab_size;
    int idx_to_char[256];
    int char_to_idx[256];
} Dataset;

typedef struct {
    ModelType model;
    const char *data_path;
    int epochs;
    int steps_per_epoch;
    int context_len;
    int d_model;
    int hidden;
    float lr;
    uint32_t seed;
    const char *benchmark_path;
    int param_budget;
    bool match_params;
} Config;

typedef struct {
    long long params;
    int used_d_model;
    int used_hidden;
    long long target_params;
    float train_loss;
    float val_loss;
    float val_acc;
    double seconds;
} RunResult;

typedef struct {
    int d_model;
    int hidden;
    long long params;
} ModelShape;

typedef struct {
    uint32_t state;
} Rng;

static void fatal(const char *msg) {
    fprintf(stderr, "error: %s\n", msg);
    exit(1);
}

static double now_seconds(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec + (double)ts.tv_nsec * 1e-9;
}

static void rng_seed(Rng *rng, uint32_t seed) {
    rng->state = seed ? seed : 1u;
}

static uint32_t rng_next_u32(Rng *rng) {
    uint32_t x = rng->state;
    x ^= x << 13;
    x ^= x >> 17;
    x ^= x << 5;
    rng->state = x;
    return x;
}

static float rng_uniform(Rng *rng) {
    return (rng_next_u32(rng) / (float)UINT32_MAX);
}

static float rng_sym(Rng *rng, float scale) {
    return (rng_uniform(rng) * 2.0f - 1.0f) * scale;
}

static int argmax(const float *x, int n) {
    int idx = 0;
    float best = x[0];
    for (int i = 1; i < n; ++i) {
        if (x[i] > best) {
            best = x[i];
            idx = i;
        }
    }
    return idx;
}

static float softmax_cross_entropy(const float *logits, int n, int target, float *probs_out) {
    float maxv = logits[0];
    for (int i = 1; i < n; ++i) {
        if (logits[i] > maxv) {
            maxv = logits[i];
        }
    }

    float sum = 0.0f;
    for (int i = 0; i < n; ++i) {
        probs_out[i] = expf(logits[i] - maxv);
        sum += probs_out[i];
    }

    if (sum < 1e-12f) {
        sum = 1e-12f;
    }

    for (int i = 0; i < n; ++i) {
        probs_out[i] /= sum;
    }

    float p = probs_out[target];
    if (p < 1e-12f) {
        p = 1e-12f;
    }
    return -logf(p);
}

static Dataset load_dataset(const char *path) {
    Dataset ds;
    memset(&ds, 0, sizeof(ds));

    for (int i = 0; i < 256; ++i) {
        ds.char_to_idx[i] = -1;
        ds.idx_to_char[i] = -1;
    }

    FILE *f = fopen(path, "rb");
    if (!f) {
        fatal("could not open dataset file");
    }

    if (fseek(f, 0, SEEK_END) != 0) {
        fclose(f);
        fatal("fseek failed");
    }
    long sz = ftell(f);
    if (sz <= 0) {
        fclose(f);
        fatal("dataset file empty");
    }
    if (fseek(f, 0, SEEK_SET) != 0) {
        fclose(f);
        fatal("fseek reset failed");
    }

    unsigned char *raw = (unsigned char *)malloc((size_t)sz);
    if (!raw) {
        fclose(f);
        fatal("out of memory reading dataset");
    }

    size_t got = fread(raw, 1, (size_t)sz, f);
    fclose(f);
    if (got != (size_t)sz) {
        free(raw);
        fatal("failed to read full dataset");
    }

    int vocab = 0;
    for (long i = 0; i < sz; ++i) {
        unsigned char c = raw[i];
        if (ds.char_to_idx[c] < 0) {
            ds.char_to_idx[c] = vocab;
            ds.idx_to_char[vocab] = (int)c;
            vocab += 1;
        }
    }

    int *tokens = (int *)malloc(sizeof(int) * (size_t)sz);
    if (!tokens) {
        free(raw);
        fatal("out of memory encoding dataset");
    }

    for (long i = 0; i < sz; ++i) {
        tokens[i] = ds.char_to_idx[raw[i]];
    }

    free(raw);
    ds.tokens = tokens;
    ds.length = (int)sz;
    ds.vocab_size = vocab;
    return ds;
}

static void free_dataset(Dataset *ds) {
    free(ds->tokens);
    ds->tokens = NULL;
}

static ModelType parse_model(const char *name) {
    if (strcmp(name, "mlp") == 0) return MODEL_MLP;
    if (strcmp(name, "lstm") == 0) return MODEL_LSTM;
    if (strcmp(name, "transformer") == 0) return MODEL_TRANSFORMER;
    if (strcmp(name, "mamba") == 0) return MODEL_MAMBA;
    if (strcmp(name, "all") == 0) return MODEL_ALL;
    fatal("unknown model; expected mlp|lstm|transformer|mamba|all");
    return MODEL_MLP;
}

static const char *model_name(ModelType t) {
    switch (t) {
        case MODEL_MLP: return "mlp";
        case MODEL_LSTM: return "lstm";
        case MODEL_TRANSFORMER: return "transformer";
        case MODEL_MAMBA: return "mamba";
        case MODEL_ALL: return "all";
        default: return "unknown";
    }
}

static Config default_config(void) {
    Config c;
    c.model = MODEL_ALL;
    c.data_path = "data/tinyshakespeare.txt";
    c.epochs = 3;
    c.steps_per_epoch = 600;
    c.context_len = 24;
    c.d_model = 24;
    c.hidden = 48;
    c.lr = 0.03f;
    c.seed = 42;
    c.benchmark_path = "results/benchmark.csv";
    c.param_budget = 0;
    c.match_params = true;
    return c;
}

static void print_help(void) {
    puts("Usage: ./bin/train [options]");
    puts("");
    puts("Options:");
    puts("  --model NAME           mlp|lstm|transformer|mamba|all");
    puts("  --data PATH            input text file");
    puts("  --epochs N             epochs (default 3)");
    puts("  --steps N              steps per epoch (default 600)");
    puts("  --ctx N                context length (default 24)");
    puts("  --dmodel N             model embedding dim (default 24)");
    puts("  --hidden N             hidden/state dim (default 48)");
    puts("  --lr F                 learning rate (default 0.03)");
    puts("  --seed N               RNG seed (default 42)");
    puts("  --benchmark PATH       output CSV path (default results/benchmark.csv)");
    puts("  --param-budget N       force target trainable params for every model");
    puts("  --no-match-params      disable automatic parameter matching");
    puts("  --help                 show this message");
}

static Config parse_args(int argc, char **argv) {
    Config c = default_config();
    for (int i = 1; i < argc; ++i) {
        if (strcmp(argv[i], "--help") == 0 || strcmp(argv[i], "-h") == 0) {
            print_help();
            exit(0);
        } else if (strcmp(argv[i], "--model") == 0 && i + 1 < argc) {
            c.model = parse_model(argv[++i]);
        } else if (strcmp(argv[i], "--data") == 0 && i + 1 < argc) {
            c.data_path = argv[++i];
        } else if (strcmp(argv[i], "--epochs") == 0 && i + 1 < argc) {
            c.epochs = atoi(argv[++i]);
        } else if (strcmp(argv[i], "--steps") == 0 && i + 1 < argc) {
            c.steps_per_epoch = atoi(argv[++i]);
        } else if (strcmp(argv[i], "--ctx") == 0 && i + 1 < argc) {
            c.context_len = atoi(argv[++i]);
        } else if (strcmp(argv[i], "--dmodel") == 0 && i + 1 < argc) {
            c.d_model = atoi(argv[++i]);
        } else if (strcmp(argv[i], "--hidden") == 0 && i + 1 < argc) {
            c.hidden = atoi(argv[++i]);
        } else if (strcmp(argv[i], "--lr") == 0 && i + 1 < argc) {
            c.lr = (float)atof(argv[++i]);
        } else if (strcmp(argv[i], "--seed") == 0 && i + 1 < argc) {
            c.seed = (uint32_t)atoi(argv[++i]);
        } else if (strcmp(argv[i], "--benchmark") == 0 && i + 1 < argc) {
            c.benchmark_path = argv[++i];
        } else if (strcmp(argv[i], "--param-budget") == 0 && i + 1 < argc) {
            c.param_budget = atoi(argv[++i]);
        } else if (strcmp(argv[i], "--no-match-params") == 0) {
            c.match_params = false;
        } else {
            fprintf(stderr, "unknown arg: %s\n", argv[i]);
            print_help();
            exit(1);
        }
    }

    if (c.epochs < 1 || c.steps_per_epoch < 1 || c.context_len < 2 || c.d_model < 2 || c.hidden < 2 || c.lr <= 0.0f || c.param_budget < 0) {
        fatal("invalid hyperparameters");
    }

    return c;
}

static long long llabs_diff(long long a, long long b) {
    return (a > b) ? (a - b) : (b - a);
}

static long long params_mlp(int vocab, int ctx, int d, int h) {
    return (long long)vocab * d + (long long)ctx * d * h + h + (long long)h * vocab + vocab;
}

static long long params_lstm(int vocab, int d, int h) {
    return (long long)vocab * d + 4LL * h * (h + d) + 4LL * h + (long long)h * vocab + vocab;
}

static long long params_transformer(int vocab, int d) {
    return (long long)vocab * d + 3LL * d * d + (long long)d * vocab + vocab;
}

static long long params_mamba(int vocab, int d, int h) {
    return (long long)vocab * d + 2LL * d * h + 2LL * h + (long long)h * vocab + vocab;
}

static int solve_best_h(ModelType model, int vocab, int ctx, int d, long long target) {
    int best_h = 2;
    long long best_delta = (1LL << 62);
    for (int h = 2; h <= 2048; ++h) {
        long long p = 0;
        if (model == MODEL_MLP) p = params_mlp(vocab, ctx, d, h);
        if (model == MODEL_LSTM) p = params_lstm(vocab, d, h);
        if (model == MODEL_MAMBA) p = params_mamba(vocab, d, h);
        long long delta = llabs_diff(p, target);
        if (delta < best_delta) {
            best_delta = delta;
            best_h = h;
        }
    }
    return best_h;
}

static int solve_best_d_for_transformer(int vocab, long long target) {
    int best_d = 2;
    long long best_delta = (1LL << 62);
    for (int d = 2; d <= 1024; ++d) {
        long long p = params_transformer(vocab, d);
        long long delta = llabs_diff(p, target);
        if (delta < best_delta) {
            best_delta = delta;
            best_d = d;
        }
    }
    return best_d;
}

static ModelShape build_shape(const Config *cfg, const Dataset *ds, ModelType model, long long target) {
    ModelShape s;
    s.d_model = cfg->d_model;
    s.hidden = cfg->hidden;
    s.params = 0;

    int vocab = ds->vocab_size;
    int ctx = cfg->context_len;

    if (!cfg->match_params) {
        if (model == MODEL_TRANSFORMER) s.hidden = 0;
        if (model == MODEL_MLP) s.params = params_mlp(vocab, ctx, s.d_model, s.hidden);
        if (model == MODEL_LSTM) s.params = params_lstm(vocab, s.d_model, s.hidden);
        if (model == MODEL_TRANSFORMER) s.params = params_transformer(vocab, s.d_model);
        if (model == MODEL_MAMBA) s.params = params_mamba(vocab, s.d_model, s.hidden);
        return s;
    }

    if (model == MODEL_TRANSFORMER) {
        s.d_model = solve_best_d_for_transformer(vocab, target);
        s.hidden = 0;
        s.params = params_transformer(vocab, s.d_model);
    } else {
        s.hidden = solve_best_h(model, vocab, ctx, s.d_model, target);
        if (model == MODEL_MLP) s.params = params_mlp(vocab, ctx, s.d_model, s.hidden);
        if (model == MODEL_LSTM) s.params = params_lstm(vocab, s.d_model, s.hidden);
        if (model == MODEL_MAMBA) s.params = params_mamba(vocab, s.d_model, s.hidden);
    }

    return s;
}

typedef struct {
    int vocab, ctx, d, h;
    float *E;
    float *W1, *b1;
    float *W2, *b2;
} MLPModel;

static void mlp_init(MLPModel *m, int vocab, int ctx, int d, int h, Rng *rng) {
    m->vocab = vocab;
    m->ctx = ctx;
    m->d = d;
    m->h = h;
    m->E = (float *)malloc(sizeof(float) * (size_t)vocab * (size_t)d);
    m->W1 = (float *)malloc(sizeof(float) * (size_t)(ctx * d) * (size_t)h);
    m->b1 = (float *)calloc((size_t)h, sizeof(float));
    m->W2 = (float *)malloc(sizeof(float) * (size_t)h * (size_t)vocab);
    m->b2 = (float *)calloc((size_t)vocab, sizeof(float));
    if (!m->E || !m->W1 || !m->b1 || !m->W2 || !m->b2) fatal("oom mlp init");

    for (int i = 0; i < vocab * d; ++i) m->E[i] = rng_sym(rng, 0.08f);
    for (int i = 0; i < ctx * d * h; ++i) m->W1[i] = rng_sym(rng, 0.08f);
    for (int i = 0; i < h * vocab; ++i) m->W2[i] = rng_sym(rng, 0.08f);
}

static void mlp_free(MLPModel *m) {
    free(m->E); free(m->W1); free(m->b1); free(m->W2); free(m->b2);
    memset(m, 0, sizeof(*m));
}

static float mlp_step(MLPModel *m, const int *ctx_tokens, int target, float lr, bool train, int *pred_out) {
    int xdim = m->ctx * m->d;
    float *x = (float *)malloc(sizeof(float) * (size_t)xdim);
    float *h = (float *)malloc(sizeof(float) * (size_t)m->h);
    float *logits = (float *)calloc((size_t)m->vocab, sizeof(float));
    float *probs = (float *)calloc((size_t)m->vocab, sizeof(float));
    if (!x || !h || !logits || !probs) fatal("oom mlp step");

    for (int t = 0; t < m->ctx; ++t) {
        int idx = ctx_tokens[t];
        memcpy(&x[t * m->d], &m->E[idx * m->d], sizeof(float) * (size_t)m->d);
    }

    for (int j = 0; j < m->h; ++j) {
        float s = m->b1[j];
        for (int i = 0; i < xdim; ++i) {
            s += x[i] * m->W1[i * m->h + j];
        }
        h[j] = tanhf(s);
    }

    for (int k = 0; k < m->vocab; ++k) {
        float s = m->b2[k];
        for (int j = 0; j < m->h; ++j) {
            s += h[j] * m->W2[j * m->vocab + k];
        }
        logits[k] = s;
    }

    float loss = softmax_cross_entropy(logits, m->vocab, target, probs);
    int pred = argmax(probs, m->vocab);
    if (pred_out) *pred_out = pred;

    if (train) {
        float *dlogits = probs;
        dlogits[target] -= 1.0f;

        float *dh = (float *)calloc((size_t)m->h, sizeof(float));
        float *dz1 = (float *)calloc((size_t)m->h, sizeof(float));
        float *dx = (float *)calloc((size_t)xdim, sizeof(float));
        float *dE = (float *)calloc((size_t)m->vocab * (size_t)m->d, sizeof(float));
        if (!dh || !dz1 || !dx || !dE) fatal("oom mlp backprop");

        for (int j = 0; j < m->h; ++j) {
            for (int k = 0; k < m->vocab; ++k) {
                m->W2[j * m->vocab + k] -= lr * (h[j] * dlogits[k]);
                dh[j] += m->W2[j * m->vocab + k] * dlogits[k];
            }
        }
        for (int k = 0; k < m->vocab; ++k) {
            m->b2[k] -= lr * dlogits[k];
        }

        for (int j = 0; j < m->h; ++j) {
            dz1[j] = dh[j] * (1.0f - h[j] * h[j]);
        }

        for (int i = 0; i < xdim; ++i) {
            float xi = x[i];
            for (int j = 0; j < m->h; ++j) {
                m->W1[i * m->h + j] -= lr * (xi * dz1[j]);
                dx[i] += m->W1[i * m->h + j] * dz1[j];
            }
        }
        for (int j = 0; j < m->h; ++j) {
            m->b1[j] -= lr * dz1[j];
        }

        for (int t = 0; t < m->ctx; ++t) {
            int tok = ctx_tokens[t];
            for (int d = 0; d < m->d; ++d) {
                dE[tok * m->d + d] += dx[t * m->d + d];
            }
        }

        for (int i = 0; i < m->vocab * m->d; ++i) {
            m->E[i] -= lr * dE[i];
        }

        free(dh); free(dz1); free(dx); free(dE);
    }

    free(x); free(h); free(logits); free(probs);
    return loss;
}

typedef struct {
    int vocab, ctx, d, h;
    float *E;
    float *Wf, *Wi, *Wo, *Wg;
    float *bf, *bi, *bo, *bg;
    float *Wy, *by;
} LSTMModel;

static float sigmoidf(float x) {
    if (x > 20.0f) return 1.0f;
    if (x < -20.0f) return 0.0f;
    return 1.0f / (1.0f + expf(-x));
}

static void lstm_init(LSTMModel *m, int vocab, int ctx, int d, int h, Rng *rng) {
    int u = h + d;
    m->vocab = vocab; m->ctx = ctx; m->d = d; m->h = h;
    m->E = (float *)malloc(sizeof(float) * (size_t)vocab * (size_t)d);
    m->Wf = (float *)malloc(sizeof(float) * (size_t)h * (size_t)u);
    m->Wi = (float *)malloc(sizeof(float) * (size_t)h * (size_t)u);
    m->Wo = (float *)malloc(sizeof(float) * (size_t)h * (size_t)u);
    m->Wg = (float *)malloc(sizeof(float) * (size_t)h * (size_t)u);
    m->bf = (float *)calloc((size_t)h, sizeof(float));
    m->bi = (float *)calloc((size_t)h, sizeof(float));
    m->bo = (float *)calloc((size_t)h, sizeof(float));
    m->bg = (float *)calloc((size_t)h, sizeof(float));
    m->Wy = (float *)malloc(sizeof(float) * (size_t)h * (size_t)vocab);
    m->by = (float *)calloc((size_t)vocab, sizeof(float));
    if (!m->E || !m->Wf || !m->Wi || !m->Wo || !m->Wg || !m->bf || !m->bi || !m->bo || !m->bg || !m->Wy || !m->by) {
        fatal("oom lstm init");
    }

    for (int i = 0; i < vocab * d; ++i) m->E[i] = rng_sym(rng, 0.08f);
    for (int i = 0; i < h * u; ++i) {
        m->Wf[i] = rng_sym(rng, 0.08f);
        m->Wi[i] = rng_sym(rng, 0.08f);
        m->Wo[i] = rng_sym(rng, 0.08f);
        m->Wg[i] = rng_sym(rng, 0.08f);
    }
    for (int i = 0; i < h * vocab; ++i) m->Wy[i] = rng_sym(rng, 0.08f);
}

static void lstm_free(LSTMModel *m) {
    free(m->E); free(m->Wf); free(m->Wi); free(m->Wo); free(m->Wg);
    free(m->bf); free(m->bi); free(m->bo); free(m->bg);
    free(m->Wy); free(m->by);
    memset(m, 0, sizeof(*m));
}

static float lstm_step(LSTMModel *m, const int *ctx_tokens, int target, float lr, bool train, int *pred_out) {
    int T = m->ctx;
    int H = m->h;
    int D = m->d;
    int U = H + D;

    float *x = (float *)malloc(sizeof(float) * (size_t)T * (size_t)D);
    float *u = (float *)malloc(sizeof(float) * (size_t)T * (size_t)U);
    float *f = (float *)malloc(sizeof(float) * (size_t)T * (size_t)H);
    float *ii = (float *)malloc(sizeof(float) * (size_t)T * (size_t)H);
    float *o = (float *)malloc(sizeof(float) * (size_t)T * (size_t)H);
    float *g = (float *)malloc(sizeof(float) * (size_t)T * (size_t)H);
    float *h = (float *)malloc(sizeof(float) * (size_t)T * (size_t)H);
    float *c = (float *)malloc(sizeof(float) * (size_t)T * (size_t)H);
    float *logits = (float *)calloc((size_t)m->vocab, sizeof(float));
    float *probs = (float *)calloc((size_t)m->vocab, sizeof(float));
    if (!x || !u || !f || !ii || !o || !g || !h || !c || !logits || !probs) fatal("oom lstm step");

    for (int t = 0; t < T; ++t) {
        memcpy(&x[t * D], &m->E[ctx_tokens[t] * D], sizeof(float) * (size_t)D);
    }

    for (int t = 0; t < T; ++t) {
        float *ut = &u[t * U];
        float *ft = &f[t * H];
        float *it = &ii[t * H];
        float *ot = &o[t * H];
        float *gt = &g[t * H];
        float *ht = &h[t * H];
        float *ct = &c[t * H];

        const float *hprev = (t == 0) ? NULL : &h[(t - 1) * H];
        const float *cprev = (t == 0) ? NULL : &c[(t - 1) * H];

        for (int j = 0; j < H; ++j) {
            ut[j] = hprev ? hprev[j] : 0.0f;
        }
        memcpy(&ut[H], &x[t * D], sizeof(float) * (size_t)D);

        for (int j = 0; j < H; ++j) {
            float zf = m->bf[j], zi = m->bi[j], zo = m->bo[j], zg = m->bg[j];
            for (int k = 0; k < U; ++k) {
                float uk = ut[k];
                zf += m->Wf[j * U + k] * uk;
                zi += m->Wi[j * U + k] * uk;
                zo += m->Wo[j * U + k] * uk;
                zg += m->Wg[j * U + k] * uk;
            }
            ft[j] = sigmoidf(zf);
            it[j] = sigmoidf(zi);
            ot[j] = sigmoidf(zo);
            gt[j] = tanhf(zg);
            float cp = cprev ? cprev[j] : 0.0f;
            ct[j] = ft[j] * cp + it[j] * gt[j];
            ht[j] = ot[j] * tanhf(ct[j]);
        }
    }

    const float *hT = &h[(T - 1) * H];
    for (int v = 0; v < m->vocab; ++v) {
        float s = m->by[v];
        for (int j = 0; j < H; ++j) {
            s += hT[j] * m->Wy[j * m->vocab + v];
        }
        logits[v] = s;
    }

    float loss = softmax_cross_entropy(logits, m->vocab, target, probs);
    int pred = argmax(probs, m->vocab);
    if (pred_out) *pred_out = pred;

    if (train) {
        float *dlogits = probs;
        dlogits[target] -= 1.0f;

        float *dh = (float *)calloc((size_t)T * (size_t)H, sizeof(float));
        float *dc = (float *)calloc((size_t)T * (size_t)H, sizeof(float));
        float *dE = (float *)calloc((size_t)m->vocab * (size_t)D, sizeof(float));
        if (!dh || !dc || !dE) fatal("oom lstm grads");

        for (int j = 0; j < H; ++j) {
            float hj = hT[j];
            for (int v = 0; v < m->vocab; ++v) {
                m->Wy[j * m->vocab + v] -= lr * hj * dlogits[v];
                dh[(T - 1) * H + j] += m->Wy[j * m->vocab + v] * dlogits[v];
            }
        }
        for (int v = 0; v < m->vocab; ++v) {
            m->by[v] -= lr * dlogits[v];
        }

        for (int t = T - 1; t >= 0; --t) {
            float *ct = &c[t * H];
            float *ft = &f[t * H];
            float *it = &ii[t * H];
            float *ot = &o[t * H];
            float *gt = &g[t * H];
            float *ut = &u[t * U];

            const float *cprev = (t == 0) ? NULL : &c[(t - 1) * H];
            float *dhprev = (t == 0) ? NULL : &dh[(t - 1) * H];
            float *dcprev = (t == 0) ? NULL : &dc[(t - 1) * H];

            float *dht = &dh[t * H];
            float *dct = &dc[t * H];

            float *dztf = (float *)calloc((size_t)H, sizeof(float));
            float *dzti = (float *)calloc((size_t)H, sizeof(float));
            float *dzto = (float *)calloc((size_t)H, sizeof(float));
            float *dztg = (float *)calloc((size_t)H, sizeof(float));
            float *du = (float *)calloc((size_t)U, sizeof(float));
            if (!dztf || !dzti || !dzto || !dztg || !du) fatal("oom lstm gate grad");

            for (int j = 0; j < H; ++j) {
                float tanhc = tanhf(ct[j]);
                float do_ = dht[j] * tanhc;
                float dc_total = dht[j] * ot[j] * (1.0f - tanhc * tanhc) + dct[j];
                float cp = cprev ? cprev[j] : 0.0f;
                float df_ = dc_total * cp;
                float di_ = dc_total * gt[j];
                float dg_ = dc_total * it[j];

                if (dcprev) dcprev[j] += dc_total * ft[j];

                dztf[j] = df_ * ft[j] * (1.0f - ft[j]);
                dzti[j] = di_ * it[j] * (1.0f - it[j]);
                dzto[j] = do_ * ot[j] * (1.0f - ot[j]);
                dztg[j] = dg_ * (1.0f - gt[j] * gt[j]);
            }

            for (int j = 0; j < H; ++j) {
                for (int k = 0; k < U; ++k) {
                    float uk = ut[k];
                    m->Wf[j * U + k] -= lr * dztf[j] * uk;
                    m->Wi[j * U + k] -= lr * dzti[j] * uk;
                    m->Wo[j * U + k] -= lr * dzto[j] * uk;
                    m->Wg[j * U + k] -= lr * dztg[j] * uk;

                    du[k] += m->Wf[j * U + k] * dztf[j]
                           + m->Wi[j * U + k] * dzti[j]
                           + m->Wo[j * U + k] * dzto[j]
                           + m->Wg[j * U + k] * dztg[j];
                }
                m->bf[j] -= lr * dztf[j];
                m->bi[j] -= lr * dzti[j];
                m->bo[j] -= lr * dzto[j];
                m->bg[j] -= lr * dztg[j];
            }

            if (dhprev) {
                for (int j = 0; j < H; ++j) {
                    dhprev[j] += du[j];
                }
            }

            int tok = ctx_tokens[t];
            for (int d = 0; d < D; ++d) {
                dE[tok * D + d] += du[H + d];
            }

            free(dztf); free(dzti); free(dzto); free(dztg); free(du);
        }

        for (int i = 0; i < m->vocab * D; ++i) {
            m->E[i] -= lr * dE[i];
        }

        free(dh); free(dc); free(dE);
    }

    free(x); free(u); free(f); free(ii); free(o); free(g); free(h); free(c); free(logits); free(probs);
    return loss;
}

typedef struct {
    int vocab, ctx, d;
    float *E;
    float *Wq, *Wk, *Wv;
    float *Wo, *by;
} TransformerModel;

static void transformer_init(TransformerModel *m, int vocab, int ctx, int d, Rng *rng) {
    m->vocab = vocab; m->ctx = ctx; m->d = d;
    m->E = (float *)malloc(sizeof(float) * (size_t)vocab * (size_t)d);
    m->Wq = (float *)malloc(sizeof(float) * (size_t)d * (size_t)d);
    m->Wk = (float *)malloc(sizeof(float) * (size_t)d * (size_t)d);
    m->Wv = (float *)malloc(sizeof(float) * (size_t)d * (size_t)d);
    m->Wo = (float *)malloc(sizeof(float) * (size_t)d * (size_t)vocab);
    m->by = (float *)calloc((size_t)vocab, sizeof(float));
    if (!m->E || !m->Wq || !m->Wk || !m->Wv || !m->Wo || !m->by) fatal("oom transformer init");

    for (int i = 0; i < vocab * d; ++i) m->E[i] = rng_sym(rng, 0.08f);
    for (int i = 0; i < d * d; ++i) {
        m->Wq[i] = rng_sym(rng, 0.08f);
        m->Wk[i] = rng_sym(rng, 0.08f);
        m->Wv[i] = rng_sym(rng, 0.08f);
    }
    for (int i = 0; i < d * vocab; ++i) m->Wo[i] = rng_sym(rng, 0.08f);
}

static void transformer_free(TransformerModel *m) {
    free(m->E); free(m->Wq); free(m->Wk); free(m->Wv); free(m->Wo); free(m->by);
    memset(m, 0, sizeof(*m));
}

static float transformer_step(TransformerModel *m, const int *ctx_tokens, int target, float lr, bool train, int *pred_out) {
    int T = m->ctx;
    int D = m->d;
    int V = m->vocab;
    float scale = 1.0f / sqrtf((float)D);

    float *x = (float *)malloc(sizeof(float) * (size_t)T * (size_t)D);
    float *k = (float *)calloc((size_t)T * (size_t)D, sizeof(float));
    float *v = (float *)calloc((size_t)T * (size_t)D, sizeof(float));
    float *q = (float *)calloc((size_t)D, sizeof(float));
    float *scores = (float *)calloc((size_t)T, sizeof(float));
    float *attn = (float *)calloc((size_t)T, sizeof(float));
    float *ctx = (float *)calloc((size_t)D, sizeof(float));
    float *logits = (float *)calloc((size_t)V, sizeof(float));
    float *probs = (float *)calloc((size_t)V, sizeof(float));
    if (!x || !k || !v || !q || !scores || !attn || !ctx || !logits || !probs) fatal("oom transformer step");

    for (int t = 0; t < T; ++t) {
        memcpy(&x[t * D], &m->E[ctx_tokens[t] * D], sizeof(float) * (size_t)D);
    }

    float *x_last = &x[(T - 1) * D];
    for (int i = 0; i < D; ++i) {
        for (int j = 0; j < D; ++j) {
            q[j] += x_last[i] * m->Wq[i * D + j];
        }
    }

    for (int t = 0; t < T; ++t) {
        float *xt = &x[t * D];
        float *kt = &k[t * D];
        float *vt = &v[t * D];
        for (int i = 0; i < D; ++i) {
            for (int j = 0; j < D; ++j) {
                kt[j] += xt[i] * m->Wk[i * D + j];
                vt[j] += xt[i] * m->Wv[i * D + j];
            }
        }
    }

    for (int t = 0; t < T; ++t) {
        float dot = 0.0f;
        float *kt = &k[t * D];
        for (int j = 0; j < D; ++j) dot += q[j] * kt[j];
        scores[t] = dot * scale;
    }

    float maxs = scores[0];
    for (int t = 1; t < T; ++t) if (scores[t] > maxs) maxs = scores[t];
    float sum = 0.0f;
    for (int t = 0; t < T; ++t) {
        attn[t] = expf(scores[t] - maxs);
        sum += attn[t];
    }
    if (sum < 1e-12f) sum = 1e-12f;
    for (int t = 0; t < T; ++t) attn[t] /= sum;

    for (int t = 0; t < T; ++t) {
        float *vt = &v[t * D];
        for (int j = 0; j < D; ++j) {
            ctx[j] += attn[t] * vt[j];
        }
    }

    for (int out = 0; out < V; ++out) {
        float s = m->by[out];
        for (int j = 0; j < D; ++j) s += ctx[j] * m->Wo[j * V + out];
        logits[out] = s;
    }

    float loss = softmax_cross_entropy(logits, V, target, probs);
    int pred = argmax(probs, V);
    if (pred_out) *pred_out = pred;

    if (train) {
        float *dlogits = probs;
        dlogits[target] -= 1.0f;

        float *dctx = (float *)calloc((size_t)D, sizeof(float));
        float *dq = (float *)calloc((size_t)D, sizeof(float));
        float *dk = (float *)calloc((size_t)T * (size_t)D, sizeof(float));
        float *dv = (float *)calloc((size_t)T * (size_t)D, sizeof(float));
        float *da = (float *)calloc((size_t)T, sizeof(float));
        float *ds = (float *)calloc((size_t)T, sizeof(float));
        float *dx = (float *)calloc((size_t)T * (size_t)D, sizeof(float));
        float *dE = (float *)calloc((size_t)V * (size_t)D, sizeof(float));
        if (!dctx || !dq || !dk || !dv || !da || !ds || !dx || !dE) fatal("oom transformer grad");

        for (int j = 0; j < D; ++j) {
            for (int out = 0; out < V; ++out) {
                m->Wo[j * V + out] -= lr * ctx[j] * dlogits[out];
                dctx[j] += m->Wo[j * V + out] * dlogits[out];
            }
        }
        for (int out = 0; out < V; ++out) m->by[out] -= lr * dlogits[out];

        for (int t = 0; t < T; ++t) {
            float *vt = &v[t * D];
            for (int j = 0; j < D; ++j) {
                dv[t * D + j] += attn[t] * dctx[j];
                da[t] += dctx[j] * vt[j];
            }
        }

        float dot_da = 0.0f;
        for (int t = 0; t < T; ++t) dot_da += da[t] * attn[t];
        for (int t = 0; t < T; ++t) ds[t] = attn[t] * (da[t] - dot_da);

        for (int t = 0; t < T; ++t) {
            float *kt = &k[t * D];
            for (int j = 0; j < D; ++j) {
                dq[j] += ds[t] * kt[j] * scale;
                dk[t * D + j] += ds[t] * q[j] * scale;
            }
        }

        for (int i = 0; i < D; ++i) {
            for (int j = 0; j < D; ++j) {
                m->Wq[i * D + j] -= lr * x_last[i] * dq[j];
                dx[(T - 1) * D + i] += m->Wq[i * D + j] * dq[j];
            }
        }

        for (int t = 0; t < T; ++t) {
            float *xt = &x[t * D];
            for (int i = 0; i < D; ++i) {
                for (int j = 0; j < D; ++j) {
                    m->Wk[i * D + j] -= lr * xt[i] * dk[t * D + j];
                    m->Wv[i * D + j] -= lr * xt[i] * dv[t * D + j];
                    dx[t * D + i] += m->Wk[i * D + j] * dk[t * D + j]
                                  + m->Wv[i * D + j] * dv[t * D + j];
                }
            }
        }

        for (int t = 0; t < T; ++t) {
            int tok = ctx_tokens[t];
            for (int j = 0; j < D; ++j) {
                dE[tok * D + j] += dx[t * D + j];
            }
        }

        for (int i = 0; i < V * D; ++i) {
            m->E[i] -= lr * dE[i];
        }

        free(dctx); free(dq); free(dk); free(dv); free(da); free(ds); free(dx); free(dE);
    }

    free(x); free(k); free(v); free(q); free(scores); free(attn); free(ctx); free(logits); free(probs);
    return loss;
}

typedef struct {
    int vocab, ctx, d, h;
    float *E;
    float *Wg, *bg;
    float *Wu, *bu;
    float *Wy, *by;
} MambaModel;

static void mamba_init(MambaModel *m, int vocab, int ctx, int d, int h, Rng *rng) {
    m->vocab = vocab; m->ctx = ctx; m->d = d; m->h = h;
    m->E = (float *)malloc(sizeof(float) * (size_t)vocab * (size_t)d);
    m->Wg = (float *)malloc(sizeof(float) * (size_t)d * (size_t)h);
    m->bg = (float *)calloc((size_t)h, sizeof(float));
    m->Wu = (float *)malloc(sizeof(float) * (size_t)d * (size_t)h);
    m->bu = (float *)calloc((size_t)h, sizeof(float));
    m->Wy = (float *)malloc(sizeof(float) * (size_t)h * (size_t)vocab);
    m->by = (float *)calloc((size_t)vocab, sizeof(float));
    if (!m->E || !m->Wg || !m->bg || !m->Wu || !m->bu || !m->Wy || !m->by) fatal("oom mamba init");

    for (int i = 0; i < vocab * d; ++i) m->E[i] = rng_sym(rng, 0.08f);
    for (int i = 0; i < d * h; ++i) {
        m->Wg[i] = rng_sym(rng, 0.08f);
        m->Wu[i] = rng_sym(rng, 0.08f);
    }
    for (int i = 0; i < h * vocab; ++i) m->Wy[i] = rng_sym(rng, 0.08f);
}

static void mamba_free(MambaModel *m) {
    free(m->E); free(m->Wg); free(m->bg); free(m->Wu); free(m->bu); free(m->Wy); free(m->by);
    memset(m, 0, sizeof(*m));
}

static float mamba_step(MambaModel *m, const int *ctx_tokens, int target, float lr, bool train, int *pred_out) {
    int T = m->ctx;
    int D = m->d;
    int H = m->h;
    int V = m->vocab;

    float *x = (float *)malloc(sizeof(float) * (size_t)T * (size_t)D);
    float *g = (float *)calloc((size_t)T * (size_t)H, sizeof(float));
    float *u = (float *)calloc((size_t)T * (size_t)H, sizeof(float));
    float *s = (float *)calloc((size_t)T * (size_t)H, sizeof(float));
    float *logits = (float *)calloc((size_t)V, sizeof(float));
    float *probs = (float *)calloc((size_t)V, sizeof(float));
    if (!x || !g || !u || !s || !logits || !probs) fatal("oom mamba step");

    for (int t = 0; t < T; ++t) {
        memcpy(&x[t * D], &m->E[ctx_tokens[t] * D], sizeof(float) * (size_t)D);
    }

    for (int t = 0; t < T; ++t) {
        const float *sprev = (t == 0) ? NULL : &s[(t - 1) * H];
        for (int j = 0; j < H; ++j) {
            float zg = m->bg[j];
            float zu = m->bu[j];
            for (int d = 0; d < D; ++d) {
                float xd = x[t * D + d];
                zg += xd * m->Wg[d * H + j];
                zu += xd * m->Wu[d * H + j];
            }
            float gj = sigmoidf(zg);
            float uj = tanhf(zu);
            g[t * H + j] = gj;
            u[t * H + j] = uj;
            float sp = sprev ? sprev[j] : 0.0f;
            s[t * H + j] = (1.0f - gj) * sp + gj * uj;
        }
    }

    float *sT = &s[(T - 1) * H];
    for (int out = 0; out < V; ++out) {
        float y = m->by[out];
        for (int j = 0; j < H; ++j) y += sT[j] * m->Wy[j * V + out];
        logits[out] = y;
    }

    float loss = softmax_cross_entropy(logits, V, target, probs);
    int pred = argmax(probs, V);
    if (pred_out) *pred_out = pred;

    if (train) {
        float *dlogits = probs;
        dlogits[target] -= 1.0f;

        float *ds = (float *)calloc((size_t)T * (size_t)H, sizeof(float));
        float *dE = (float *)calloc((size_t)V * (size_t)D, sizeof(float));
        if (!ds || !dE) fatal("oom mamba grad");

        for (int j = 0; j < H; ++j) {
            for (int out = 0; out < V; ++out) {
                m->Wy[j * V + out] -= lr * sT[j] * dlogits[out];
                ds[(T - 1) * H + j] += m->Wy[j * V + out] * dlogits[out];
            }
        }
        for (int out = 0; out < V; ++out) m->by[out] -= lr * dlogits[out];

        for (int t = T - 1; t >= 0; --t) {
            const float *sprev = (t == 0) ? NULL : &s[(t - 1) * H];

            float *dpre_g = (float *)calloc((size_t)H, sizeof(float));
            float *dpre_u = (float *)calloc((size_t)H, sizeof(float));
            float *dx = (float *)calloc((size_t)D, sizeof(float));
            if (!dpre_g || !dpre_u || !dx) fatal("oom mamba gate grad");

            for (int j = 0; j < H; ++j) {
                float ds_t = ds[t * H + j];
                float sp = sprev ? sprev[j] : 0.0f;
                float gj = g[t * H + j];
                float uj = u[t * H + j];

                float dg = ds_t * (uj - sp);
                float du = ds_t * gj;
                float ds_prev = ds_t * (1.0f - gj);
                if (t > 0) ds[(t - 1) * H + j] += ds_prev;

                dpre_g[j] = dg * gj * (1.0f - gj);
                dpre_u[j] = du * (1.0f - uj * uj);
            }

            for (int d = 0; d < D; ++d) {
                float xd = x[t * D + d];
                for (int j = 0; j < H; ++j) {
                    m->Wg[d * H + j] -= lr * xd * dpre_g[j];
                    m->Wu[d * H + j] -= lr * xd * dpre_u[j];
                    dx[d] += m->Wg[d * H + j] * dpre_g[j] + m->Wu[d * H + j] * dpre_u[j];
                }
            }
            for (int j = 0; j < H; ++j) {
                m->bg[j] -= lr * dpre_g[j];
                m->bu[j] -= lr * dpre_u[j];
            }

            int tok = ctx_tokens[t];
            for (int d = 0; d < D; ++d) dE[tok * D + d] += dx[d];

            free(dpre_g); free(dpre_u); free(dx);
        }

        for (int i = 0; i < V * D; ++i) m->E[i] -= lr * dE[i];

        free(ds); free(dE);
    }

    free(x); free(g); free(u); free(s); free(logits); free(probs);
    return loss;
}

static int sample_position(Rng *rng, int low, int high) {
    if (high <= low) return low;
    uint32_t r = rng_next_u32(rng);
    return low + (int)(r % (uint32_t)(high - low));
}

static void make_context(const Dataset *ds, int pos, int ctx_len, int *ctx_out, int *target_out) {
    for (int i = 0; i < ctx_len; ++i) {
        ctx_out[i] = ds->tokens[pos - ctx_len + i];
    }
    *target_out = ds->tokens[pos];
}

static void ensure_csv_header(const char *path) {
    FILE *f = fopen(path, "r");
    if (f) {
        fclose(f);
        return;
    }
    f = fopen(path, "w");
    if (!f) fatal("could not create benchmark csv");
    fprintf(f, "model,epochs,steps,ctx,d_model,hidden,target_params,params,lr,seed,train_loss,val_loss,val_acc,seconds\n");
    fclose(f);
}

static void append_result_csv(const char *path, const Config *cfg, const char *model, const RunResult *r) {
    FILE *f = fopen(path, "a");
    if (!f) fatal("could not append benchmark csv");
    fprintf(f, "%s,%d,%d,%d,%d,%d,%lld,%lld,%.6f,%u,%.6f,%.6f,%.6f,%.6f\n",
            model, cfg->epochs, cfg->steps_per_epoch, cfg->context_len, r->used_d_model, r->used_hidden,
            r->target_params, r->params, cfg->lr, cfg->seed, r->train_loss, r->val_loss, r->val_acc, r->seconds);
    fclose(f);
}

static void eval_model(ModelType model, void *model_ptr, const Dataset *ds, int split, int ctx_len, float *loss_out, float *acc_out) {
    int *ctx = (int *)malloc(sizeof(int) * (size_t)ctx_len);
    if (!ctx) fatal("oom eval ctx");

    int n = 0;
    double loss_sum = 0.0;
    int correct = 0;
    int start = split;
    if (start < ctx_len) start = ctx_len;

    int max_eval = 2000;
    int end = ds->length;
    for (int pos = start; pos < end && n < max_eval; ++pos) {
        int target = 0;
        make_context(ds, pos, ctx_len, ctx, &target);
        int pred = -1;
        float loss = 0.0f;

        switch (model) {
            case MODEL_MLP:
                loss = mlp_step((MLPModel *)model_ptr, ctx, target, 0.0f, false, &pred);
                break;
            case MODEL_LSTM:
                loss = lstm_step((LSTMModel *)model_ptr, ctx, target, 0.0f, false, &pred);
                break;
            case MODEL_TRANSFORMER:
                loss = transformer_step((TransformerModel *)model_ptr, ctx, target, 0.0f, false, &pred);
                break;
            case MODEL_MAMBA:
                loss = mamba_step((MambaModel *)model_ptr, ctx, target, 0.0f, false, &pred);
                break;
            default:
                fatal("invalid model eval");
        }

        loss_sum += loss;
        correct += (pred == target) ? 1 : 0;
        n += 1;
    }

    if (n == 0) {
        *loss_out = 0.0f;
        *acc_out = 0.0f;
    } else {
        *loss_out = (float)(loss_sum / (double)n);
        *acc_out = (float)correct / (float)n;
    }

    free(ctx);
}

static RunResult train_one(const Config *cfg, const Dataset *ds, ModelType model, const ModelShape *shape, long long target_params) {
    if (ds->length < cfg->context_len + 2) {
        fatal("dataset too small for context length");
    }

    Rng rng;
    rng_seed(&rng, cfg->seed + (uint32_t)model * 17u);

    int split = (int)((double)ds->length * 0.9);
    if (split <= cfg->context_len + 1) split = cfg->context_len + 1;
    if (split >= ds->length - 1) split = ds->length - 1;

    int *ctx = (int *)malloc(sizeof(int) * (size_t)cfg->context_len);
    if (!ctx) fatal("oom train ctx");

    printf("\n=== Training %s ===\n", model_name(model));
    printf("dataset: %d chars, vocab: %d, train split: %d, val split: %d\n",
           ds->length, ds->vocab_size, split, ds->length - split);
    if (model == MODEL_TRANSFORMER) {
        printf("shape: d_model=%d params=%lld target=%lld\n", shape->d_model, shape->params, target_params);
    } else {
        printf("shape: d_model=%d hidden=%d params=%lld target=%lld\n",
               shape->d_model, shape->hidden, shape->params, target_params);
    }

    double t0 = now_seconds();
    double train_loss_sum = 0.0;
    int train_count = 0;

    MLPModel mlp;
    LSTMModel lstm;
    TransformerModel tr;
    MambaModel ma;
    memset(&mlp, 0, sizeof(mlp));
    memset(&lstm, 0, sizeof(lstm));
    memset(&tr, 0, sizeof(tr));
    memset(&ma, 0, sizeof(ma));

    if (model == MODEL_MLP) mlp_init(&mlp, ds->vocab_size, cfg->context_len, shape->d_model, shape->hidden, &rng);
    if (model == MODEL_LSTM) lstm_init(&lstm, ds->vocab_size, cfg->context_len, shape->d_model, shape->hidden, &rng);
    if (model == MODEL_TRANSFORMER) transformer_init(&tr, ds->vocab_size, cfg->context_len, shape->d_model, &rng);
    if (model == MODEL_MAMBA) mamba_init(&ma, ds->vocab_size, cfg->context_len, shape->d_model, shape->hidden, &rng);

    for (int epoch = 0; epoch < cfg->epochs; ++epoch) {
        double epoch_loss = 0.0;
        for (int step = 0; step < cfg->steps_per_epoch; ++step) {
            int pos = sample_position(&rng, cfg->context_len, split);
            int target = 0;
            make_context(ds, pos, cfg->context_len, ctx, &target);

            float loss = 0.0f;
            switch (model) {
                case MODEL_MLP:
                    loss = mlp_step(&mlp, ctx, target, cfg->lr, true, NULL);
                    break;
                case MODEL_LSTM:
                    loss = lstm_step(&lstm, ctx, target, cfg->lr, true, NULL);
                    break;
                case MODEL_TRANSFORMER:
                    loss = transformer_step(&tr, ctx, target, cfg->lr, true, NULL);
                    break;
                case MODEL_MAMBA:
                    loss = mamba_step(&ma, ctx, target, cfg->lr, true, NULL);
                    break;
                default:
                    fatal("invalid model train");
            }

            epoch_loss += loss;
            train_loss_sum += loss;
            train_count += 1;
        }

        printf("epoch %d/%d train_loss=%.4f\n", epoch + 1, cfg->epochs, epoch_loss / (double)cfg->steps_per_epoch);
    }

    float val_loss = 0.0f;
    float val_acc = 0.0f;
    switch (model) {
        case MODEL_MLP:
            eval_model(model, &mlp, ds, split, cfg->context_len, &val_loss, &val_acc);
            break;
        case MODEL_LSTM:
            eval_model(model, &lstm, ds, split, cfg->context_len, &val_loss, &val_acc);
            break;
        case MODEL_TRANSFORMER:
            eval_model(model, &tr, ds, split, cfg->context_len, &val_loss, &val_acc);
            break;
        case MODEL_MAMBA:
            eval_model(model, &ma, ds, split, cfg->context_len, &val_loss, &val_acc);
            break;
        default:
            fatal("invalid model eval dispatch");
    }

    double t1 = now_seconds();

    RunResult rr;
    rr.params = shape->params;
    rr.used_d_model = shape->d_model;
    rr.used_hidden = shape->hidden;
    rr.target_params = target_params;
    rr.train_loss = (train_count > 0) ? (float)(train_loss_sum / (double)train_count) : 0.0f;
    rr.val_loss = val_loss;
    rr.val_acc = val_acc;
    rr.seconds = t1 - t0;

    printf("result model=%s train_loss=%.4f val_loss=%.4f val_acc=%.4f seconds=%.2f\n",
           model_name(model), rr.train_loss, rr.val_loss, rr.val_acc, rr.seconds);

    if (model == MODEL_MLP) mlp_free(&mlp);
    if (model == MODEL_LSTM) lstm_free(&lstm);
    if (model == MODEL_TRANSFORMER) transformer_free(&tr);
    if (model == MODEL_MAMBA) mamba_free(&ma);
    free(ctx);
    return rr;
}

static void ensure_dir_results(void) {
#if defined(_WIN32)
    _mkdir("results");
#else
    mkdir("results", 0777);
#endif
}

int main(int argc, char **argv) {
    Config cfg = parse_args(argc, argv);
    Dataset ds = load_dataset(cfg.data_path);
    ensure_dir_results();
    ensure_csv_header(cfg.benchmark_path);

    long long target_params = 0;
    if (cfg.param_budget > 0) {
        target_params = cfg.param_budget;
    } else {
        target_params = params_mamba(ds.vocab_size, cfg.d_model, cfg.hidden);
    }
    printf("parameter policy: %s, target=%lld\n", cfg.match_params ? "matched" : "raw", target_params);

    if (cfg.model == MODEL_ALL) {
        ModelType models[4] = {MODEL_MLP, MODEL_LSTM, MODEL_TRANSFORMER, MODEL_MAMBA};
        for (int i = 0; i < 4; ++i) {
            ModelShape shape = build_shape(&cfg, &ds, models[i], target_params);
            RunResult rr = train_one(&cfg, &ds, models[i], &shape, target_params);
            append_result_csv(cfg.benchmark_path, &cfg, model_name(models[i]), &rr);
        }
    } else {
        ModelShape shape = build_shape(&cfg, &ds, cfg.model, target_params);
        RunResult rr = train_one(&cfg, &ds, cfg.model, &shape, target_params);
        append_result_csv(cfg.benchmark_path, &cfg, model_name(cfg.model), &rr);
    }

    free_dataset(&ds);
    return 0;
}
