# Local LLMs on a GTX 1080 (Pascal, sm_61): what helps and what does not

By **Adrian Ciprian Iancu**. Notes from about two months of A/B tests on my own GTX 1080 (8 GB, PCIe x8 gen3) with llama.cpp, on Windows 11 (CUDA 12.4) and on Pop!_OS (CUDA 12). Numbers are medians of alternated runs (A-B-B-A). Kernel and build changes were kept only with identical output (same perplexity, same greedy tokens).

## Worth doing

| Change | Gain | Notes |
|---|---|---|
| Build for the card: `-DCMAKE_CUDA_ARCHITECTURES=61-real` instead of the default PTX | +4.6% generation (Windows) | the DLL drops from 538 MB to 69.5 MB and startup from 43.3 s to 20.4 s; output identical |
| Allow CUDA graphs on Pascal (`cc < GGML_CUDA_CC_VOLTA` to `cc < GGML_CUDA_CC_DP4A` in `ggml-cuda.cu`, [PR #27721](https://github.com/ggml-org/llama.cpp/pull/27721)) | +39.8% on Ling-3.0-tiny (41.2 to 57.6 tok/s); on six models from +5.6% (dense 9B) to +53.3% (Granite 3B MoE), Windows | perplexity identical 6/6; the gain grows with the number of graph nodes, because decoding on this card is launch-bound. On MiniCPM5-2B with a DSpark draft on Linux it was x1.00, so measure it on your model |
| Erase finished slots (`--slot-save-path` + slot erase) when you serve several sequences with a unified KV cache | x2 decode on my agent replays (29.6 to 59.7 tok/s, Linux) | dead sequences still cost time in the attention; see [RESULTS.md](RESULTS.md) |
| KV cache in f16 instead of q8_0, if it fits | +13.6% on MiniCPM5-2B (Linux) | changes numbers, but toward the reference: f16 is what q8_0 is compared against |
| Speculative decoding | verifying 4 draft tokens costs about the same as 1 on this card | the card waits on launches most of the time per token, so a short draft is almost free |

## Not worth it (measured)

| Change | Result |
|---|---|
| Windows hardware-accelerated GPU scheduling (HAGS) | about 0 gain and +514 MiB of VRAM: I turned it back off |
| Fan fixed at 90% | same steady clock as auto under 4 minutes of prefill (1632 MHz, 88 C against 89 C): the throttling is thermal, and only new thermal paste would fix it |
| n-gram speculation on top of the DSpark draft | +6.5% in a quick screen without a fixed seed, x1.003 with seed 42: noise |

## Measurement traps I fell into

- "The server prefill is 3.5x slower than llama-bench" was an artifact: measured properly it is 1.23x.
- Long runs heat the card and the clock drops from about 1733 to 1632 MHz: compare A and B only in the same thermal state, alternating them.
- At temperature 1.0 without a fixed seed, two runs of the same build differed by up to 10%: fix the seed before you believe a small gain.
- On Windows with 8 GB, PyTorch training that fragments memory spills into system RAM and gets twice as slow; one `torch.cuda.empty_cache()` per step fixed it.

## Software that still supports Pascal

- CUDA 13 drops Pascal: stay on CUDA 12.x toolkits for llama.cpp builds.
- For training I use `torch==2.14` with CUDA 12.6, the last PyTorch line with Pascal wheels.
- Native fp16 arithmetic runs at 1/64 of fp32 on GP104: my training runs in fp32.
