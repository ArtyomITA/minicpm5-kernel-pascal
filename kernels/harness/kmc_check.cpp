// kmc_check: drives single ggml ops with MiniCPM5-2B shapes on the CUDA backend, for bit-exact comparison between
// ggml-cuda.dll variants (or valve on/off) and for nvprof kernel timing.
//
//   kmc_check mm    M N K        out = mul_mat(W[K,M] q8_0, x[K,N])
//   kmc_check glu   M N K        out = swiglu(mul_mat(Wgate,x), mul_mat(Wup,x))   (llama.cpp node order: up, gate, glu)
//   kmc_check mmadd M N K        out = mul_mat(W,x) + r
//   kmc_check qkv   N            q,k,v projections of one layer (2048/256/256 rows, K 2048), outputs concatenated
//   kmc_check fa    KV NB        flash_attn_ext, head 128, 16 Q heads / 2 KV heads, q8_0 K/V as cache views, causal mask
//   options: --iters I (default 20)  --out file.bin (raw f32 of the last run)
// Inputs are deterministic (fixed LCG), so two processes with different DLLs see identical data.
#include "ggml.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"

#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>
#include <fstream>

static uint64_t g_rng = 0x9E3779B97F4A7C15ull;
static float frand() {  // uniform in [-1, 1)
    g_rng ^= g_rng << 13; g_rng ^= g_rng >> 7; g_rng ^= g_rng << 17;
    return (float) ((g_rng >> 40) & 0xFFFFFF) / 8388608.0f - 1.0f;
}

static void fill(ggml_tensor * t) {
    const int64_t n = ggml_nelements(t);
    std::vector<float> f(n);
    for (auto & v : f) { v = frand(); }
    if (t->type == GGML_TYPE_F32) {
        ggml_backend_tensor_set(t, f.data(), 0, n * sizeof(float));
    } else if (t->type == GGML_TYPE_F16) {
        std::vector<ggml_fp16_t> h(n);
        ggml_fp32_to_fp16_row(f.data(), h.data(), n);
        ggml_backend_tensor_set(t, h.data(), 0, n * sizeof(ggml_fp16_t));
    } else {
        std::vector<uint8_t> q(ggml_nbytes(t));
        ggml_quantize_chunk(t->type, f.data(), q.data(), 0, n / t->ne[0], t->ne[0], nullptr);
        ggml_backend_tensor_set(t, q.data(), 0, q.size());
    }
}

int main(int argc, char ** argv) {
    if (argc < 3) {
        fprintf(stderr, "usage: kmc_check mm|glu|mmadd M N K | qkv N | fa KV NB [--iters I] [--out f.bin]\n");
        return 1;
    }
    const std::string mode = argv[1];
    std::vector<int64_t> a;
    int iters = 20;
    const char * out_path = nullptr;
    for (int i = 2; i < argc; ++i) {
        if (!strcmp(argv[i], "--iters")) { iters = atoi(argv[++i]); }
        else if (!strcmp(argv[i], "--out")) { out_path = argv[++i]; }
        else { a.push_back(atoll(argv[i])); }
    }

    ggml_backend_load_all();
    {   // print which libggml-cuda.so this process really loaded (A/B by LD_LIBRARY_PATH must be provable)
        std::ifstream maps("/proc/self/maps");
        std::string line;
        while (std::getline(maps, line)) {
            const size_t at = line.find("libggml-cuda.so");
            if (at != std::string::npos) {
                fprintf(stderr, "lib %s\n", line.substr(line.find('/')).c_str());
                break;
            }
        }
    }
    ggml_backend_t backend = ggml_backend_init_by_type(GGML_BACKEND_DEVICE_TYPE_GPU, nullptr);
    if (!backend) { fprintf(stderr, "no GPU backend\n"); return 2; }

    ggml_init_params ip = { 64 * ggml_tensor_overhead() + ggml_graph_overhead(), nullptr, true };
    ggml_context * cw = ggml_init(ip);   // weights and inputs (backend buffer)
    ggml_context * cg = ggml_init(ip);   // graph ops
    std::vector<ggml_tensor *> inputs;
    ggml_tensor * out = nullptr;
    double wbytes = 0;

    auto T2 = [&](ggml_type t, int64_t n0, int64_t n1) { ggml_tensor * x = ggml_new_tensor_2d(cw, t, n0, n1); inputs.push_back(x); return x; };

    if (mode == "mm" || mode == "glu" || mode == "mmadd") {
        const int64_t M = a[0], N = a[1], K = a[2];
        ggml_tensor * x = T2(GGML_TYPE_F32, K, N);
        if (mode == "glu") {
            ggml_tensor * wu = T2(GGML_TYPE_Q8_0, K, M);
            ggml_tensor * wg = T2(GGML_TYPE_Q8_0, K, M);
            ggml_tensor * up = ggml_mul_mat(cg, wu, x);
            ggml_tensor * gt = ggml_mul_mat(cg, wg, x);
            out = ggml_swiglu_split(cg, gt, up);
            wbytes = 2.0 * ggml_nbytes(wu);
        } else {
            ggml_tensor * w = T2(GGML_TYPE_Q8_0, K, M);
            out = ggml_mul_mat(cg, w, x);
            if (mode == "mmadd") {
                out = ggml_add(cg, out, T2(GGML_TYPE_F32, M, N));
            }
            wbytes = ggml_nbytes(w);
        }
    } else if (mode == "qkv") {
        const int64_t N = a[0];
        ggml_tensor * x  = T2(GGML_TYPE_F32, 2048, N);
        ggml_tensor * wq = T2(GGML_TYPE_Q8_0, 2048, 2048);
        ggml_tensor * wk = T2(GGML_TYPE_Q8_0, 2048, 256);
        ggml_tensor * wv = T2(GGML_TYPE_Q8_0, 2048, 256);
        ggml_tensor * q = ggml_mul_mat(cg, wq, x);
        ggml_tensor * k = ggml_mul_mat(cg, wk, x);
        ggml_tensor * v = ggml_mul_mat(cg, wv, x);
        out = ggml_concat(cg, ggml_concat(cg, q, k, 0), v, 0);
        wbytes = ggml_nbytes(wq) + ggml_nbytes(wk) + ggml_nbytes(wv);
    } else if (mode == "fa") {
        const int64_t KV = a[0], NB = a[1];
        ggml_tensor * q  = ggml_new_tensor_3d(cw, GGML_TYPE_F32, 128, NB, 16); inputs.push_back(q);
        ggml_tensor * kc = T2(GGML_TYPE_Q8_0, 256, KV);  // cache rows: 2 KV heads x 128 per cell
        ggml_tensor * vc = T2(GGML_TYPE_Q8_0, 256, KV);
        ggml_tensor * m  = T2(GGML_TYPE_F16, KV, NB);
        ggml_tensor * k = ggml_view_3d(cg, kc, 128, KV, 2, kc->nb[1], ggml_row_size(kc->type, 128), 0);
        ggml_tensor * v = ggml_view_3d(cg, vc, 128, KV, 2, vc->nb[1], ggml_row_size(vc->type, 128), 0);
        out = ggml_flash_attn_ext(cg, q, k, v, m, 1.0f / sqrtf(128.0f), 0.0f, 0.0f);
        ggml_flash_attn_ext_set_prec(out, GGML_PREC_F32);
        wbytes = ggml_nbytes(kc) + ggml_nbytes(vc);
    } else {
        fprintf(stderr, "unknown mode\n"); return 1;
    }

    ggml_backend_buffer_t buf = ggml_backend_alloc_ctx_tensors(cw, backend);
    for (ggml_tensor * t : inputs) {
        if (t->type == GGML_TYPE_F16 && mode == "fa") {  // causal mask: row j sees cells <= KV - NB + j
            const int64_t KV = t->ne[0], NB = t->ne[1];
            std::vector<ggml_fp16_t> h(KV * NB);
            for (int64_t j = 0; j < NB; ++j)
                for (int64_t i = 0; i < KV; ++i)
                    h[j * KV + i] = ggml_fp32_to_fp16(i <= KV - NB + j ? 0.0f : -INFINITY);
            ggml_backend_tensor_set(t, h.data(), 0, h.size() * sizeof(ggml_fp16_t));
        } else {
            fill(t);
        }
    }

    ggml_cgraph * gf = ggml_new_graph(cg);
    ggml_build_forward_expand(gf, out);
    ggml_gallocr_t ga = ggml_gallocr_new(ggml_backend_get_default_buffer_type(backend));
    ggml_gallocr_alloc_graph(ga, gf);

    ggml_backend_graph_compute(backend, gf);  // warmup (and CUDA graph capture)
    const auto t0 = std::chrono::steady_clock::now();
    for (int i = 0; i < iters; ++i) {
        ggml_backend_graph_compute(backend, gf);
    }
    const double us = std::chrono::duration<double, std::micro>(std::chrono::steady_clock::now() - t0).count() / iters;

    std::vector<float> res(ggml_nelements(out));
    ggml_backend_tensor_get(out, res.data(), 0, res.size() * sizeof(float));
    uint64_t h = 1469598103934665603ull;
    const uint8_t * p = (const uint8_t *) res.data();
    for (size_t i = 0; i < res.size() * sizeof(float); ++i) { h = (h ^ p[i]) * 1099511628211ull; }
    printf("%s", mode.c_str());
    for (auto v : a) { printf(" %lld", (long long) v); }
    printf(" | %.1f us/graph (host timed) | %.1f GB/s | fnv %016llx | out[0..2] %.9g %.9g %.9g\n",
           us, wbytes / us / 1e3, (unsigned long long) h, res[0], res.size() > 1 ? res[1] : 0.0f, res.size() > 2 ? res[2] : 0.0f);
    if (out_path) {
        FILE * f = fopen(out_path, "wb");
        fwrite(res.data(), sizeof(float), res.size(), f);
        fclose(f);
    }
    ggml_gallocr_free(ga);
    ggml_backend_buffer_free(buf);
    ggml_backend_free(backend);
    ggml_free(cg);
    ggml_free(cw);
    return 0;
}
