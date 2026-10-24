# One-token decisions with a 2B model: calibration first, then a LoRA

By **Adrian Ciprian Iancu**. My assistant [Vergilius](https://github.com/ArtyomITA/vergilius) asks a small local model many quick questions before it answers: which tool fits this message, which element on the screen is the one the user means, whether a page has finished loading. Each question is multiple choice. The model writes its reasoning (or none), the prompt closes the thinking block, the server generates one token, and I read the probabilities of the option letters (`n_probs` on `/completion`, temperature 0). One decision costs about 0.2 s on a GTX 1080.

These are the numbers from my bench and what moved them.

## The bench

- 100 multiple-choice questions in three categories (tool choice, UI element, page state), split once into a development half and a test half before any tuning.
- A separate held-out set of 62 questions, Italian and English, written from real situations (real tool descriptions, UI fixtures, pages modelled on real cases), used **once** per final configuration. Everything below is on that set unless noted. With 62 questions one question is 1.6 points.
- Every method, threshold and prior was chosen on the development half only, then frozen.

## What worked, at zero cost

Two changes, same latency:

1. **Prefill `Answer:` after the closed thinking block**, so the next token is the letter (it arrives as `" A"`, with the space).
2. **Batch calibration** ([Zhou et al., ICLR 2024](https://arxiv.org/abs/2309.17249)): per category, the prior is the mean of the letter probabilities over the development inputs (no labels needed); at decision time I subtract its log from the log probabilities. The prior is estimated once on a balanced set and frozen per model, language and prompt. Never estimate it on live traffic.

| model | language | before | `Answer:` + calibration | full reasoning (7 to 9 s per decision) |
|---|---|---|---|---|
| LFM2.5-2.6B | Italian | 66.1 | **79.0** | 90.3 |
| LFM2.5-2.6B | English | 71.0 | **85.5** | 91.9 |
| MiniCPM5-2B | Italian | 46.8 | 64.5 | 85.5 |
| MiniCPM5-2B | English | 51.6 | **77.4** | 87.1 |

MiniCPM5 had a strong bias toward the "none of these" option (prior 0.96 in the tool category in Italian): calibration removes most of it.

## A LoRA for English decisions

[pjev-2b](https://huggingface.co/Kailune-AI/pjev-2b) is a MiniCPM5-2B fine-tuned for agent decisions. I checked that its 87 non-linear tensors are identical to the base, so it is the base plus an adapter merged into the 7 linear matrices; the singular values of the difference fall off by rank 64. I extracted it back as a rank 64 LoRA with truncated SVD ([`tools/extract_lora.py`](tools/extract_lora.py)), 201 MB in f16, and published it as [MiniCPM5-2B-pjev-LoRA](https://huggingface.co/adriandj3/MiniCPM5-2B-pjev-LoRA).

| configuration, English held-out set | accuracy | latency |
|---|---|---|
| MiniCPM5-2B, `Answer:` + calibration | 75.8 | |
| pjev-2b merged, its own prompt format | 91.9 (57/62) | median 0.16 s |
| **pjev as LoRA over MiniCPM5-2B Q8_0, same format** | **93.5 (58/62)** | median 0.22 s |
| LFM2.5-2.6B with full reasoning | 91.9 | 7.8 s mean |

The LoRA and the merged model make the same choice on 60 of 62 questions. It is English only: on the Italian set it reaches 69.4.

## What did not work

- Putting "none" as the first option: worse.
- NoThinking (an empty thinking block): no gain.
- PriDe debiasing: worse than batch calibration.
- Averaging over all option permutations: 6 to 9 forward passes, and in English worse without calibration.
- Batch calibration with a gamma tuned on the development labels: overfits (71.4 on the test half against 83.7).
- A thinking budget below 256 tokens on MiniCPM5.

## Two traps for production

- **The adapter scale stays on the slot.** With llama.cpp b10549, a LoRA scale set by one request stayed on that slot: later requests on the same slot without a `lora` field ran with the adapter on (Italian held-out set, that slot 19% against 75-76% on the others). Pass `"lora": [{"id": 0, "scale": ...}]` explicitly in every request, chat included, or give the decisions their own slot.
- **Decisions compete with the chat for the GPU.** 20 one-token decisions during a 512-token chat answer took the chat from 76.5 to 50.1 tok/s. A reasoning decision holds the GPU for 5 to 9 s.

## Cite

```bibtex
@misc{iancu2026fastdecisions,
  author       = {Iancu, Adrian Ciprian},
  title        = {One-token decisions with a 2B model: calibration first, then a LoRA},
  year         = {2026},
  howpublished = {\url{https://github.com/ArtyomITA/minicpm5-kernel-pascal/blob/main/FAST-DECISIONS.md}}
}
```
