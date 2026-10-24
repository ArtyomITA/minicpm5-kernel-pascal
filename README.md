# MiniCPM5-2B on a GTX 1080: from 33 to 83 tokens/s

By **Adrian Ciprian Iancu** · [GitHub](https://github.com/ArtyomITA) · [Hugging Face](https://huggingface.co/adriandj3) · [LinkedIn](https://www.linkedin.com/in/adrian-ciprian-iancu-a2a477391)

How the local assistant [Vergilius](https://github.com/ArtyomITA/vergilius) got **2.5x faster decode** on an
eight-year-old Pascal card, measured on replays of real agent turns, with a quality check on every step that
changes numbers.

| step | what changed | decode | measured on | quality |
|---|---|---|---|---|
| start | dead sequences left in the unified KV cache | 29.6 to 33.4 tok/s | agent replay (Italian turns) | |
| 1 | erase finished slots (`--slot-save-path` + slot erase) | 59.7 tok/s (x2.0) | same replay, clean slots | identical tokens |
| 2 | Pascal kernels **kmcG** (patch in `kernels/`) | +5.5% | server, 4 English prompt types, ABBA | bit-identical (95/95 harness, 10/10 server) |
| 3 | KV cache **f16** instead of q8_0 | 69.7 to 79.0 tok/s, +13.6% at equal acceptance | English agent replay, 30 turns | **better**: f16 is the reference, q8_0 differs from it |
| 4 | **FR-Spec** draft, 32k vocabulary ([on Hugging Face](https://huggingface.co/adriandj3/MiniCPM5-2B-DSpark-FRSpec32k-GGUF)) | +4.7% | English agent replay, 24 turns, fixed seed | same output distribution |
| **total** | | **83.5 tok/s**, 31.9 ms per speculative step | English agent replay | |

General notes on what helps and what does not on Pascal (build flags, CUDA graphs, slots, HAGS, thermals): [PASCAL-NOTES.md](PASCAL-NOTES.md).

Prefill on an 8k prompt: 1264 tok/s. VRAM: 6.4 GB of 8 with a 49k context.

Bonus, for agents that need fast decisions: [pjev-2b as a LoRA](https://huggingface.co/adriandj3/MiniCPM5-2B-pjev-LoRA)
over the same MiniCPM5-2B, 93.5% on 62 held-out agent decisions in 0.22 s, for about 200 MB of VRAM.

## The recipe

Hardware: GTX 1080 8 GB (sm_61), Linux, CUDA 12. Software: llama.cpp b10549 (`b2e5e9b28`) plus the Pascal patch.

```bash
GGML_CUDA_DISABLE_GRAPHS=1 llama-server \
  -m MiniCPM5-2B-Q8_0.gguf \
  -md MiniCPM5-DSpark-fr32k-Q8_0.gguf -ngld 999 \
  --spec-type draft-dspark --spec-draft-n-max 5 --spec-draft-n-min 0 \
  --cache-type-k f16 --cache-type-v f16 \
  -c 49152 -b 2048 -ub 1024 -ngl 999 -fa on --jinja \
  --slot-save-path ./slots \
  --temp 1.0 --top-p 0.95 --min-p 0.0
```

Then make sure your client **erases a slot when its conversation ends** (`POST /slots/<id>?action=erase`). With a
unified KV cache, sequences of finished conversations stay in memory and every attention step walks over them: in
our replays that alone halved decode speed (59.7 to 29.6 tok/s). The cost model we measured: milliseconds per
speculative step = 29.9 + 1.26 x (thousands of context tokens, dead ones included).

Details of every flag and why: [RECIPE.md](RECIPE.md). Every number, method and what did **not** work:
[RESULTS.md](RESULTS.md).

## What is in this repository

| path | what |
|---|---|
| `kernels/patches/pascal-sm61-kmcG-B02-B04.patch` | the adopted kernel patch against llama.cpp b10549, bit-identical outputs |
| `kernels/patches/experimental-H5-tileq8-on-top.patch` | flash attention reading the q8_0 KV cache directly; bit-identical, not adopted |
| `kernels/harness/` | bit-for-bit comparison of two builds on MiniCPM5 shapes (`kmc_check.cpp`, `harness.py`) |
| `tools/frspec.py` | builds the FR-Spec draft from your own token statistics |
| `tools/extract_lora.py` | re-expresses a merged fine-tune as a LoRA by truncated SVD (used for pjev-2b) |
| `RECIPE.md`, `RESULTS.md` | the configuration and the measurements |

See [kernels/README.md](kernels/README.md) to apply the patch and verify it on your card.

## How we measured

- Replays of real turns of the agent (tool calls, answers, long JSON tool outputs), English, one stream.
- ABBA order, first round discarded, fixed sampling seed (noise between arms about 0.2%; without a fixed seed it was
  up to 10%), GPU start temperature at or below 60 C, clock in every table.
- Kernel changes had to be bit-identical before any speed number. Changes that alter numbers (KV type, weight
  quantization) had to pass a KL-divergence gate against the most faithful configuration we could run.
- The GTX 1080 throttles itself after two to three minutes of load (91 to 94 C, 1640 MHz instead of 1936): absolute
  numbers on your card will differ, the ratios between arms are what we report.

## License

Code and patches: MIT (see `LICENSE`; the patches modify llama.cpp, MIT, copyright the ggml authors). Models on
Hugging Face: Apache-2.0, see their cards. Not affiliated with OpenBMB, llama.cpp, or the authors of pjev-2b.

## Citation

```bibtex
@misc{iancu2026minicpm5pascal,
  author       = {Iancu, Adrian Ciprian},
  title        = {MiniCPM5-2B on a GTX 1080: from 33 to 83 tokens/s},
  year         = {2026},
  howpublished = {\url{https://github.com/ArtyomITA/minicpm5-kernel-pascal}}
}
```

Keywords: llama.cpp, CUDA, GTX 1080, Pascal, sm_61, MiniCPM5, speculative decoding, DSpark, FR-Spec, KV cache, local LLM, decode speed, older NVIDIA GPUs
