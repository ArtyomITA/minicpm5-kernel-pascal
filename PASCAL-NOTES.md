# Local LLMs on a GTX 1080 (Pascal, sm_61): what helps and what does not

By **Adrian Ciprian Iancu**. Notes from about two months of A/B tests on my own GTX 1080 (8 GB, PCIe x8 gen3, Windows 11, CUDA 12.4) with llama.cpp. Every "yes" below kept the output identical (same perplexity, same greedy tokens); every number is a median of alternated runs (A-B-B-A) unless noted. Things that changed numbers are marked.

## Worth doing

| Change | Gain | Notes |
|---|---|---|
| Build for the card: `-DCMAKE_CUDA_ARCHITECTURES=61-real` instead of the default PTX | +4.6% generation | the DLL drops from 538 MB to 69.5 MB and startup from 43.3 s to 20.4 s; output identical |
| Allow CUDA graphs on Pascal (`cc < GGML_CUDA_CC_VOLTA` to `cc < GGML_CUDA_CC_DP4A` in `ggml-cuda.cu`, [PR #27721](https://github.com/ggml-org/llama.cpp/pull/27721)) | +39.8% on Ling-3.0-tiny (41.2 to 57.6 tok/s); on six models from +5.6% (dense 9B) to +53.3% (Granite 3B MoE) | perplexity identical 6/6, greedy decode identical; the gain grows with the number of graph nodes, because decoding on this card is launch-bound |
| Erase finished slots (`--slot-save-path` + slot erase) when you serve several sequences with a unified KV cache | x2 decode on my agent replays (29.6 to 59.7 tok/s) | dead sequences still cost time in the attention; see [RESULTS.md](RESULTS.md) |
| KV cache in f16 instead of q8_0, if it fits | +13.6% on MiniCPM5-2B | f16 is also the reference for quality |
| Speculative decoding | verifying 4 draft tokens costs about the same as 1 on this card | the card is idle most of the time per token, so drafts are almost free |

## Not worth it (measured)

| Change | Result |
|---|---|
| Windows hardware-accelerated GPU scheduling (HAGS) | about 0 gain and +514 MiB of VRAM: I turned it back off |
| Fan fixed at 85-90% | same steady clock and temperature as auto (1632 MHz, 88-89 C): the throttling is thermal, not power, and only new thermal paste would fix it |
| Lowering the power limit | useless, for the same reason |
| `fast_fp16_available` instead of `fp16_available` in the conversion path | -2.5%, within noise: correct but not faster |
| n-gram speculation | +6.5% in a quick screen, x1.003 with a fixed seed: noise |

## Measurement traps I fell into

- "The server prefill is 3.5x slower than llama-bench" was an artifact: measured properly it is 1.23x.
- Long runs heat the card and the clock drops from about 1835 to 1632 MHz: compare A and B only at the same temperature, alternating them.
- On Windows with 8 GB, PyTorch training that fragments memory spills into system RAM and gets twice as slow; one `torch.cuda.empty_cache()` per step fixed it.

## Software versions that still support Pascal

- PyTorch 2.14 with CUDA 12.6 is the last line with Pascal wheels.
- CUDA 13 drops Pascal: stay on CUDA 12.x toolkits for llama.cpp builds.
- fp16 compute runs at 1/64 of fp32 on GP104, so train and compute in fp32; fp16/bf16 are fine as storage.
