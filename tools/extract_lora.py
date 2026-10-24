"""Extract pjev-2b as a LoRA over openbmb/MiniCPM5-2B: delta = W_pjev - W_base per linear weight, truncated SVD.

Usage:
  BASE_DIR=... TUNED_DIR=... OUT_DIR=... python extract_lora.py analizza            # identity check of non-linear tensors + singular value decay on samples
  python extract_lora.py estrai --rango 16   # full extraction to PEFT format (adapter_model.safetensors + config)

CPU only, threads limited (shared machine with GPU benches). Reads both checkpoints lazily (safetensors).
"""
import argparse
import json
import os
import sys
import time
from pathlib import Path

import torch
from safetensors import safe_open
from safetensors.torch import save_file

BASE = Path(os.environ.get("BASE_DIR", "MiniCPM5-2B"))         # openbmb/MiniCPM5-2B safetensors
PJEV = Path(os.environ.get("TUNED_DIR", "pjev-2b"))            # Kailune-AI/pjev-2b safetensors
OUT = Path(os.environ.get("OUT_DIR", "pjev-lora"))
LINEARI = ("q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj")


def apri(cartella):
    idx = cartella / "model.safetensors.index.json"
    if idx.exists():
        mappa = json.loads(idx.read_text())["weight_map"]
    else:
        f = next(cartella.glob("*.safetensors"))
        with safe_open(f, "pt") as h:
            mappa = {k: f.name for k in h.keys()}
    handles = {fn: safe_open(cartella / fn, "pt") for fn in set(mappa.values())}
    return mappa, handles


def leggi(mappa, handles, nome):
    return handles[mappa[nome]].get_tensor(nome)


def analizza():
    mb, hb = apri(BASE)
    mp, hp = apri(PJEV)
    solo_p = sorted(set(mp) - set(mb))
    solo_b = sorted(set(mb) - set(mp))
    print(f"tensori base {len(mb)}, pjev {len(mp)}, solo pjev {solo_p[:5]}, solo base {solo_b[:5]}")
    identici, diversi_nonlin = 0, []
    for nome in sorted(set(mp) & set(mb)):
        if any(f".{l}." in nome for l in LINEARI):
            continue
        a, b = leggi(mb, hb, nome), leggi(mp, hp, nome)
        if a.shape == b.shape and torch.equal(a, b):
            identici += 1
        else:
            d = (a.float() - b.float()).abs().max().item() if a.shape == b.shape else float("nan")
            diversi_nonlin.append((nome, d))
    print(f"non lineari identici {identici}, diversi {len(diversi_nonlin)}: {diversi_nonlin[:8]}")
    for nome in ("model.layers.0.self_attn.q_proj.weight", "model.layers.20.mlp.down_proj.weight",
                 "model.layers.41.self_attn.v_proj.weight"):
        if nome not in mb or nome not in mp:
            print("manca", nome)
            continue
        d = leggi(mp, hp, nome).float() - leggi(mb, hb, nome).float()
        s = torch.linalg.svdvals(d)
        tot = (s ** 2).sum()
        cum = torch.cumsum(s ** 2, 0) / tot
        print(f"{nome} {tuple(d.shape)} |d|/|W| {d.norm() / leggi(mb, hb, nome).float().norm():.4f}")
        print("   s[0:40] / s[0]:", " ".join(f"{x:.3f}" for x in (s[:40] / s[0]).tolist()))
        print("   energia a rango 8/16/32/64/128:", " ".join(f"{cum[k - 1]:.4f}" for k in (8, 16, 32, 64, 128)))


def estrai(rango):
    mb, hb = apri(BASE)
    mp, hp = apri(PJEV)
    OUT.mkdir(parents=True, exist_ok=True)
    tensori, errori, moduli = {}, [], set()
    nomi = [n for n in sorted(set(mp) & set(mb)) if any(f".{l}." in n for l in LINEARI) and n.endswith(".weight")]
    t0 = time.time()
    for i, nome in enumerate(nomi):
        d = leggi(mp, hp, nome).float() - leggi(mb, hb, nome).float()
        U, S, V = torch.svd_lowrank(d, q=min(rango + 16, min(d.shape)), niter=6)
        U, S, V = U[:, :rango], S[:rango], V[:, :rango]
        rad = S.sqrt()
        B = U * rad                     # (out, r)
        A = (V * rad).T                 # (r, in)
        err = ((d - B @ A).norm() / d.norm()).item()
        errori.append((nome, err))
        base = "base_model.model." + nome[: -len(".weight")]
        tensori[base + ".lora_A.weight"] = A.contiguous().to(torch.float16)
        tensori[base + ".lora_B.weight"] = B.contiguous().to(torch.float16)
        moduli.add(nome.split(".")[-2])
        if i % 30 == 0:
            print(f"{i}/{len(nomi)} {nome} err {err:.4f} ({time.time() - t0:.0f} s)", flush=True)
    save_file(tensori, str(OUT / "adapter_model.safetensors"))
    cfg = {"peft_type": "LORA", "r": rango, "lora_alpha": rango, "target_modules": sorted(moduli),
           "base_model_name_or_path": "openbmb/MiniCPM5-2B", "bias": "none", "task_type": "CAUSAL_LM",
           "fan_in_fan_out": False, "lora_dropout": 0.0}
    (OUT / "adapter_config.json").write_text(json.dumps(cfg, indent=2))
    e = sorted(x for _, x in errori)
    print(f"fatto: {len(nomi)} matrici, rango {rango}, errore relativo mediana {e[len(e) // 2]:.4f} max {e[-1]:.4f}, "
          f"{time.time() - t0:.0f} s")
    (OUT / "errori.json").write_text(json.dumps(errori, indent=1))


if __name__ == "__main__":
    torch.set_num_threads(int(os.environ.get("THREADS", "2")))
    ap = argparse.ArgumentParser()
    ap.add_argument("cosa", choices=["analizza", "estrai"])
    ap.add_argument("--rango", type=int, default=16)
    a = ap.parse_args()
    analizza() if a.cosa == "analizza" else estrai(a.rango)
