# Pascal (sm_61) kernels for MiniCPM5-2B in llama.cpp

Patches against llama.cpp **b10549** (`b2e5e9b28`). Written for the GTX 10xx series (compute capability 6.1: no tensor
cores, no fast FP16). Every adopted change produces **bit-identical outputs**: same operations, same order of sums,
same rounding. Only the way data is loaded and work is split changes.

## pascal-sm61-kmcG-B02-B04.patch (adopted)

| part | file | what | valve (set the variable to disable) |
|---|---|---|---|
| kmcG Q8K | `mmvq.cu`, `mmvq.cuh` | Q8_0 matrix-vector for 1 to 8 columns: same thread mapping and sum order as upstream, constant pointers, `__ldg` on aligned words, funnel shifts; 4 rows per block at 1 column with >= 4096 rows, 2 otherwise | `GGML_CUDA_DISABLE_KMC_Q8K` |
| kmcG ROWWARP | `mmvq.cu` | one row per warp when K <= 256 (the DSpark Markov layer, 256 x 130560), same reduction | `GGML_CUDA_DISABLE_KMC_ROWWARP` |
| kmcG QKV | `ggml-cuda.cu`, `mmvq.cu` | gate and up projections at 2 to 8 columns in one launch with a shared quantized input (one quantization less per layer) | `GGML_CUDA_DISABLE_KMC_QKV` |
| B02 | `ggml-cuda.cu` | allows CUDA graphs on cc 6.1 (upstream disables them below Volta) | upstream `GGML_CUDA_DISABLE_GRAPHS` |
| B04 | `convert.cu` | Q8_0 to f16 conversion in float on Pascal instead of `__hmul2` (one rounding either way, exact product) | `GGML_CUDA_DISABLE_B04` |

Valves are read once. **Any value disables** (even `0`): to turn a part on, unset the variable. With all valves set the
build behaves like upstream b10549 (checked: 95/95 harness cases identical, server outputs identical).

Only Q8_0 weights and cc 6.x take the new paths; every other shape falls back to upstream code.

Measured on a GTX 1080, MiniCPM5-2B Q8_0 + DSpark 5: **+5.5%** decode (two ABBA pairs, four prompt types),
identical outputs 8/8. CUDA graphs: +0% on Linux, so the recipe sets `GGML_CUDA_DISABLE_GRAPHS=1` there.

## experimental-H5-tileq8-on-top.patch (not adopted)

Apply on top of the first patch. The flash attention tile kernel reads a q8_0 KV cache directly, without the
conversion of the whole cache to f16 before every call (`fattn-tile.cuh`, `fattn-common.cuh`). Bit-identical
(18/18 harness cases), -46 us per layer at 8k context. Not adopted because an f16 KV cache removes the same
conversion and is faster; useful only if you must keep a q8_0 cache. Known issue: as written it is also dispatched
in prefill, where it is 21% slower; limit it to `Q->ne[1] <= 8` (verification of speculative drafts).
Valve: `GGML_CUDA_DISABLE_KMC_TILEQ8`.

## Apply and build

```bash
git clone https://github.com/ggml-org/llama.cpp && cd llama.cpp
git checkout b2e5e9b28
patch -p1 < ../kernels/patches/pascal-sm61-kmcG-B02-B04.patch
cmake -B build -DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=61 -DLLAMA_BUILD_UI=OFF -DLLAMA_USE_PREBUILT_UI=OFF
cmake --build build --config Release -j 2 --target llama-server llama-bench
```

## Verify on your card

`harness/kmc_check.cpp` runs the MiniCPM5 shapes (matrix-vector 1 to 8 columns, gate/up, matmul + residual, QKV,
output head at 1/5/6 columns, Markov layer, flash attention 512 to 12288 x 1 to 7 tokens, prefill attention) on
deterministic random data and writes the raw outputs. `harness/harness.py` runs it against two builds (or one build
with and without a valve) and fails on any difference.

```bash
g++ -O2 -std=c++17 -I llama.cpp/ggml/include harness/kmc_check.cpp -L llama.cpp/build/bin -lggml -lggml-base \
    -o harness/kmc_check
# usage: harness.py <tag> <libdirA>[@ENV=V...] <libdirB>[@ENV=V...] [--quick | --fa]
python3 harness/harness.py upstream-vs-patched upstream/build/bin patched/build/bin
python3 harness/harness.py q8k-valve patched/build/bin@GGML_CUDA_DISABLE_KMC_Q8K=1 patched/build/bin
```

`kmc_check` is linked without rpath: `LD_LIBRARY_PATH` (set by the harness) selects which build is loaded.
