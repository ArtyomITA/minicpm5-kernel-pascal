# Full-parameter SFT of MiniCPM5-2B on an 8 GB GTX 1080 with BAdam

By **Adrian Ciprian Iancu**. Before a long fine-tuning run I measured, one lever at a time, how fast full-parameter training of MiniCPM5-2B (2.5 B parameters, 42 layers) can go on my GTX 1080 (8 GB, Pascal: no bf16, no Triton, no flash attention) with [BAdam](https://github.com/Ledzy/BAdam), which updates one block of layers at a time with AdamW.

Setup: Windows 11, `torch==2.14.1+cu126`, `transformers==5.18.0`, `badam==1.4`, weights in fp16, BAdam `switch_block_every=2`, random block order, batch 1, packed sequences of chat turns, same seed and same batches in every run so the losses can be compared step by step. 12 steps per run, the first one discarded as warm-up, median seconds per step.

## The biggest lever: let backward stop at the active block

BAdam trains one block; the layers below it are frozen. Backward still walked through all of them in my baseline, because `gradient_checkpointing_enable()` in transformers 5 calls `enable_input_require_grads()` for every model with `input_ids`, which makes the embedding output require grad. With nothing trainable below the active block, those gradients are thrown away. One line after enabling checkpointing fixes it:

```python
model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
model.disable_input_require_grads()   # backward stops at the active block
```

Use non-reentrant checkpointing with it: reentrant checkpointing gives no gradients to a checkpointed layer whose input does not require grad, which is exactly what the active block would become.

| sequence 1024 | s / step | tokens / s | peak VRAM | loss vs baseline |
|---|---|---|---|---|
| baseline (reentrant checkpointing, standard loss) | 2.89 | 354 | 6.9 GB | |
| non-reentrant checkpointing | 2.89 | 354 | 6.9 GB | identical |
| **+ `disable_input_require_grads()`** | **1.93** | **531** | 6.9 GB | **identical** (12/12 steps) |
| + fused cross-entropy (below) | 1.97 | 521 | 7.1 GB | max difference 0.0004 |
| + checkpoint only the MLP, keep attention activations | 1.82 | 563 | 7.0 GB | identical |

## Longer sequences: a fused cross-entropy

With a 130,560-token vocabulary the logits are the memory problem: at 2048 tokens the standard loss peaks at 8.5 GB and spills into system RAM. My fused loss computes the loss and the input gradient chunk by chunk in the forward pass (the head is frozen), so the full `(N, V)` logits never exist and backward costs nothing extra:

```python
class FusedCE(torch.autograd.Function):
    @staticmethod
    def forward(ctx, h, W, y, chunk):
        n = h.shape[0]
        tot = h.new_zeros((), dtype=torch.float32)
        gh = torch.empty_like(h)
        for i in range(0, n, chunk):
            lg = (h[i:i + chunk] @ W.T).float()
            yy = y[i:i + chunk]
            lse = torch.logsumexp(lg, -1)
            tot += (lse - lg.gather(1, yy[:, None])[:, 0]).sum()
            p = (lg - lse[:, None]).exp_()
            p[torch.arange(p.shape[0], device=p.device), yy] -= 1
            gh[i:i + chunk] = (p / n).to(W.dtype) @ W
        ctx.save_for_backward(gh)
        return tot / n

    @staticmethod
    def backward(ctx, g):
        (gh,) = ctx.saved_tensors
        return gh * g.to(gh.dtype), None, None, None
```

Use it on the last hidden state of the labelled positions: `FusedCE.apply(h, model.lm_head.weight, y, 512)`. PyTorch's own `F.linear_cross_entropy` saves more memory, but on this card it is slower.

| with `disable_input_require_grads()` | s / step | tokens / s | peak VRAM |
|---|---|---|---|
| 2048, standard loss | 4.84 | 423 | 8.5 GB, spills into system RAM |
| 2048, fused loss | 4.48 | 457 | 6.4 GB |
| 4096, fused loss | 11.02 | 372 | 6.6 GB |
| 8192, fused loss | 34.60 | 237 | 7.2 GB |
| 8192, `F.linear_cross_entropy` + embedding on CPU + checkpoint offload + MLP in chunks of 512 | 31.48 | 260 | 6.3 GB |

## What did not help

- Adam 8-bit (bitsandbytes) as the base optimizer: 3.10 s against 2.89 s per step.
- Adam-mini as the base optimizer: does not run under BAdam 1.4 (KeyError).
- Eager attention instead of SDPA: about the same at 1024, out of memory at 4096.
- A legacy flash attention build for Pascal: 15x slower.
- Resetting position ids per packed example: 2.21 s against 1.93 s at 1024.
- Embedding on CPU, checkpoint offload or MLP chunking alone at 4096: within 2% of each other; they only matter at 8192 for memory.

One more Windows lesson: `torch.cuda.set_per_process_memory_fraction(0.97)` makes a run fail with out of memory instead of silently spilling into system RAM (a value of 1.0 does not cap anything).

## Cite

```bibtex
@misc{iancu2026badam8gb,
  author       = {Iancu, Adrian Ciprian},
  title        = {Full-parameter SFT of MiniCPM5-2B on an 8 GB GTX 1080 with BAdam},
  year         = {2026},
  howpublished = {\url{https://github.com/ArtyomITA/minicpm5-kernel-pascal/blob/main/TRAINING-8GB.md}}
}
```
