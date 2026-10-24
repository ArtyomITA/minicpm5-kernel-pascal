# Results

GTX 1080 8 GB, Pop!_OS, NVIDIA driver 580.173.02, CUDA 12.0, llama.cpp b10549 (`b2e5e9b28`). Target
MiniCPM5-2B Q8_0, DSpark draft with 5 tokens, temperature 1.0 (the model's recommended sampling). All arms of a table
ran in the same window, in ABBA order, first round discarded. "ms per step" is one speculative step (draft plus
verification); it is the robust number when acceptance varies by chance.

## 1. Dead slots in the unified KV cache

Replay of 66 real agent turns (Italian), max 1024 output tokens, reference llama.cpp build.

| arm | decode tok/s | acceptance | prefill tok/s | ms per step |
|---|---|---|---|---|
| A, clean slots | 59.7 | 0.280 | 907 | 40.3 |
| B, three dead 10k-token sequences in the cache | 29.6 (x0.50) | 0.275 | 503 (x0.55) | 80.3 |
| C, after erasing the slots | 57.8 (x0.97) | 0.267 | 902 | 40.5 |

Fit over 61 requests: ms per step = 29.9 + 1.26 x (thousands of tokens in the cache, dead ones included).

## 2. Kernels (bit-identical)

Profile of one speculative step on Linux (nvprof, 675 steps): verification 25.8 ms of GPU work (quantized
matrix-vector 12, flash attention tile 7, KV conversion 3, output head 1.4), draft 6.0, draft KV injection 2, host
gaps about 3, gaps between kernels 2.

| lever | result | identical | decision |
|---|---|---|---|
| kmcG: Q8K matrix-vector, row-per-warp Markov layer, shared-input gate+up | +5.5% (geometric mean of 4 prompt types, two ABBA pairs +4.7 to +6.1%) | 95/95 harness cases, 10/10 server outputs | adopted |
| CUDA graphs on Pascal (B02) | DSpark 1.00; no draft x0.996 / x0.997 / x1.003 | yes | not on Linux (Windows: see RECIPE) |
| B04 (q8_0 dequant in float on Pascal) | no speed change | yes | kept, harmless |
| H5: attention reads the q8_0 KV without conversion | -46 us per layer at 8k; partial replay +1.9% decode, prefill -21% (dispatched in prefill too, should be limited to <= 8 query columns) | yes (18/18) | not adopted; f16 KV removes the same conversion |
| H5 v3 / v4 (faster loaders) | v4 223 us vs 239 | no (1e-9 differences) | dropped |

## 3. KV cache f16 (changes numbers, so gated on quality)

KL divergence of the logits on 80k English tokens (`llama-perplexity --kl-divergence`), reference = Q8_0 weights +
f16 KV (the most faithful configuration on this card), perplexity 3.5909.

| candidate | PPL ratio | mean KLD | KLD 99th pct | same top-1 token |
|---|---|---|---|---|
| reference with another ubatch (noise) | 1.00026 | 0.000000 | 0.00004 | 99.995% |
| KV q8_0 (what we ran before) | 1.00067 | 0.00092 | 0.0062 | 98.67% |
| Q6_K weights + KV f16 | 1.00526 | 0.00643 | 0.043 | 96.36% |

Gate: mean KLD <= 0.001 and top-1 >= 99.5%. f16 KV is the reference, so it passes by definition and the q8_0 cache
we used before was the less faithful one. Q6_K weights fail (7x the threshold) and were dropped without a speed test.

Speed, English replay of 30 turns: q8_0 69.7 / 63.7 tok/s, f16 79.0 / 79.2 tok/s; ms per step 38.8 vs 34.15 in both
pairs (-12%, +13.6% at equal acceptance). The second q8_0 arm had acceptance 0.293 instead of 0.339 by chance, which
is why we report ms per step. VRAM 5341 to 6285 MiB. No runaway generations, no stray `</think>`.

## 4. Draft

| lever | result | decision |
|---|---|---|
| FR-Spec 32k draft | 79.6 / 79.8 to 83.5 / 83.4 tok/s (+4.7%), 33.5 to 31.9 ms per step, acceptance 0.333 to 0.331, +122 MiB | adopted |
| n-gram + DSpark (`ngram-map-k,draft-dspark`) | first single pass looked like +6.5%; with a fixed seed x1.003 | not adopted |

Lesson: at temperature 1.0 without a fixed seed, two arms of the same configuration differ by up to 10%. Every
replay comparison after that used seed 42 (noise about 0.2%).

## 5. Prefill

`llama-bench -p 8192`, f16 KV, ABBA: ubatch 256 1151.9, 512 1198.8, 1024 1263.6, 2048 1269.0 tok/s. Our 1500 tok/s
goal was not reached: on this card prefill is flash attention tile (about 50%) and Q8_0 matrix-matrix (about 42%).

## 6. Not reached, next

- 90 tok/s decode (we stopped at 83.5). Left on the table: about 3 ms per step of host work between steps
  (sampling, accept, checkpoint restore), a fused RMS norm + q8_1 quantization (-0.5 to -1 ms), better acceptance on
  English prose (0.21 to 0.33).
- 1500 tok/s prefill: needs Pascal-specific matrix-matrix and attention tile kernels.

## Decider (System One) results

62 held-out English decisions from the agent (tool choice 24, UI element 22, page state 16), never used for tuning:

| decider | accuracy | latency |
|---|---|---|
| MiniCPM5-2B, plain prompt + batch calibration | 75.8% | 0.15 s |
| MiniCPM5-2B + rejev LoRA (joe-q), best per category | 83.9% | 0.2 s |
| LFM2.5-2.6B, plain prompt + batch calibration | 85.5% | 0.23 s |
| LFM2.5-2.6B, full reasoning | 91.9% | 7.8 s |
| pjev-2b (Kailune-AI), merged model | 91.9% | 0.16 s |
| **MiniCPM5-2B + pjev LoRA (extracted, rank 64)** | **93.5%** | 0.22 s |

## 7. Follow-up (room 13, same day): what else was tried

All measured against the adopted configuration above, same window, ABBA, seed 42.

| test | result | identical | decision |
|---|---|---|---|
| Host-side work per step (timers in the server) | 31.47 ms step: target verification 24.03, draft 5.58, target sampling + accept 0.93, draft feature transfer 0.73, rest 0.20. Pure host work is 1.86 ms (5.9%) | - | no lever: the ceiling is +6%, a realistic one (top-k on GPU) +1.5 to +3.8% with hard identity constraints (grammar, reasoning budget) |
| CPU power profile `performance` vs `balanced` | 83.3 vs 83.4 tok/s (x0.999) | - | no effect: the CPU already ran near 4 GHz |
| Fused RMS norm + q8_1 quantization, quantized input shared by Q/K/V | 83.0 / 82.9 to 83.5 / 83.6 tok/s (+0.7%), 22 quantize launches per layer to 0 | yes (151/151 harness, 10/10 server) | below the +1.5% bar, not adopted |
| Q8_0 matrix-matrix with a 128-column tile (prefill) | kernel -10 to -15% at 1024 columns, end-to-end prefill not measured | yes (10/10 per tile size, 10/10 server) | open |
| Flash attention tile occupancy variants | only two bit-identical variants exist on sm_61, estimated below +4% | - | not measured |

Note on noise: with a fixed seed, two runs of the same configuration produce the same sampled text in only about a
third of the turns (8/24); tiny run-to-run numeric differences change samples at temperature 1.0. This is why every
comparison reports ms per step at equal acceptance and uses paired ABBA arms.
