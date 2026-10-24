# Recipe: MiniCPM5-2B with speculative decoding on a GTX 1080

Target: one user, one stream, an agent that calls tools and reads long JSON results. GTX 1080 8 GB (Pascal, sm_61),
llama.cpp b10549 (`b2e5e9b28`) built with the Pascal patch from `kernels/`.

## Files

| file | where |
|---|---|
| `MiniCPM5-2B-Q8_0.gguf` (target) | [openbmb/MiniCPM5-2B-GGUF](https://huggingface.co/openbmb/MiniCPM5-2B-GGUF) |
| `MiniCPM5-DSpark-fr32k-Q8_0.gguf` (draft) | [adriandj3/MiniCPM5-2B-DSpark-FRSpec32k-GGUF](https://huggingface.co/adriandj3/MiniCPM5-2B-DSpark-FRSpec32k-GGUF) |
| `pjev-lora-r64-f16.gguf` (optional decider) | [adriandj3/MiniCPM5-2B-pjev-LoRA](https://huggingface.co/adriandj3/MiniCPM5-2B-pjev-LoRA) |

## Server

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

| flag | why | measured |
|---|---|---|
| `-md ...fr32k...` | FR-Spec draft: scores 32,768 rows instead of 130,560 at every draft position | +4.7% decode, acceptance -0.2 points |
| `--spec-draft-n-max 5` | DSpark with 5 draft tokens; verification runs 6 columns | in a real agent replay the draft is worth about +20% over no draft (not +50% as in synthetic benches) |
| `--cache-type-k/v f16` | no q8_0 to f16 conversion of the whole KV before every attention call (about 3 ms per step) | +13.6% at equal acceptance; +944 MiB VRAM at 49k context |
| `-ub 1024` | prefill micro-batch | 256: 1152, 512: 1199, 1024: 1264, 2048: 1269 tok/s on an 8k prompt; 2048 costs +534 MiB for +0.4% |
| `GGML_CUDA_DISABLE_GRAPHS=1` | the patch enables CUDA graphs on Pascal; on Linux they give nothing, so they are left off as measured | 0.0% with and without draft (three measurements within 0.5%) |
| `--slot-save-path` | enables the slot actions endpoint, needed to erase slots | |

## The one change that matters most: erase finished slots

With a unified KV cache, llama.cpp keeps the sequences of finished conversations until the slot is reused or erased.
Attention then walks over them on every step. In our replays three dead 10k-token sequences halved decode speed
(59.7 to 29.6 tok/s) and prefill (907 to 503 tok/s). Erasing restored 57.8 tok/s.

```bash
curl -X POST "http://127.0.0.1:8080/slots/0?action=erase"
```

Call it when a conversation is closed, or before reusing a slot for an unrelated task.

## Optional: the decider LoRA

Load `--lora pjev-lora-r64-f16.gguf --lora-init-without-apply` and send decision requests with
`"lora": [{"id": 0, "scale": 1.0}]`. In b10549 a LoRA scale set by a request stays on the slot: send
`"lora": [{"id": 0, "scale": 0.0}]` on every chat request, or give the decider its own slot. Prompt format and
results on the model card.

## Windows

The same flags apply. On Windows (WDDM driver) CUDA graphs do help, because every kernel launch is expensive there:
on a MoE model with 1,794 graph nodes they were worth +40%. For MiniCPM5-2B on Windows we have not measured it; try
both. Numbers in this repository are from Linux.

## VRAM budget at 49k context (measured)

| configuration | peak VRAM |
|---|---|
| KV q8_0, official draft | 5341 MiB |
| KV f16, official draft | 6285 MiB |
| **KV f16, FR-Spec draft (recipe)** | **6407 MiB** |

Short on VRAM: `-c 32768` frees about 0.7 GB with the f16 cache and does not change speed on short prompts.
