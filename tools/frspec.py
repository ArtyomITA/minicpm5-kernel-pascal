"""FR-Spec for the MiniCPM5 DSpark draft: a draft GGUF with a reduced draft vocabulary (d2t), no C++ change.

The dflash graph of b10549 (src/models/dflash.cpp) already supports it: with tensor `d2t` (I64, target ids of the
kept rows) the draft uses its own `output.weight` [n_embd, N] and `markov_w2.weight` [R, N] instead of the target
head [n_embd, 130560], and scatters the logits back to the full vocabulary (-inf elsewhere). The target verifies
every drafted token, so the output distribution is unchanged; only acceptance can drop (a token outside the kept set
can never be drafted).

Steps (CPU only, no GPU):
  1. python3 frspec.py corpus <out.txt>
       MiniCPM5 outputs (reasoning + answer) of the suite runs Q-minicpm5-{it,en}-{n,r}; the T run is left out
       because the replay measures on it.
  2. CUDA_VISIBLE_DEVICES= llama-tokenize -m <target.gguf> -f <out.txt> --ids --log-disable > ids.txt
  3. python3 frspec.py build --ids ids.txt --n 32768 --out <dir>/MiniCPM5-DSpark-fr32k-Q8_0.gguf
       Kept set = every special/control token, the corpus ids by frequency, then the lowest ids (BPE merge order) up
       to N; rows copied byte for byte (Q8_0) from
       the target output.weight and the draft markov_w2.weight. Prints the corpus coverage of the kept set.
Run with a Python that has numpy, and PYTHONPATH=<llama.cpp>/gguf-py.
Paths: FRSPEC_RUN (folder with the turn jsonl files), FRSPEC_MODELS (folder with the target and draft GGUF).
"""
import argparse
import ast
import collections
import json
import os
import sys
from pathlib import Path

HOME = Path.home()
RUN = Path(os.environ.get("FRSPEC_RUN", "run"))
MODELS = Path(os.environ.get("FRSPEC_MODELS", "models"))
TARGET = MODELS / "MiniCPM5-2B-Q8_0.gguf"
DRAFT = MODELS / "MiniCPM5-2.6B-DSpark-Q8_0.gguf"
CORPUS_FILES = ["turni-Q-minicpm5-it-n.jsonl", "turni-Q-minicpm5-it-r.jsonl",
                "turni-Q-minicpm5-en-n.jsonl", "turni-Q-minicpm5-en-r.jsonl"]


def corpus(out):
    n = chars = 0
    with open(out, "w", encoding="utf-8") as f:
        for name in CORPUS_FILES:
            for line in open(RUN / name, encoding="utf-8"):
                r = json.loads(line)
                for k in ("pensiero", "testo"):
                    t = r.get(k) or ""
                    if t:
                        f.write(t + "\n\n")
                        n += 1
                        chars += len(t)
    print(f"corpus {out}: {n} texts, {chars} chars")


def read_ids(path):
    ids = []
    for line in open(path, encoding="utf-8"):
        line = line.strip()
        if line.startswith("["):
            ids.extend(ast.literal_eval(line))
    return ids


def build(a):
    import numpy as np
    import gguf

    ids = read_ids(a.ids)
    cnt = collections.Counter(ids)
    tr = gguf.GGUFReader(str(TARGET))
    dr = gguf.GGUFReader(str(DRAFT))
    n_vocab = len(tr.fields["tokenizer.ggml.tokens"].contents())
    ttype = tr.fields["tokenizer.ggml.token_type"].contents()
    special = {i for i, t in enumerate(ttype) if t in (gguf.TokenType.CONTROL, gguf.TokenType.USER_DEFINED)}
    keep = set(special)
    for tok, _ in cnt.most_common():
        if len(keep) >= a.n:
            break
        keep.add(tok)
    n_corpus = len(keep)
    # the corpus is small (a few thousand distinct ids): fill the rest with the lowest ids, a frequency prior
    # (BPE ids follow merge order, i.e. tokenizer-corpus frequency)
    for tok in range(n_vocab):
        if len(keep) >= a.n:
            break
        keep.add(tok)
    print(f"kept from corpus + special: {n_corpus}, filled by lowest ids: {len(keep) - n_corpus}")
    keep = sorted(keep)
    covered = sum(c for t, c in cnt.items() if t in set(keep))
    print(f"ids {len(ids)}, distinct {len(cnt)}, special {len(special)}, kept {len(keep)} of {n_vocab}, "
          f"corpus coverage {covered / max(1, len(ids)):.4%}")

    t_out = next(t for t in tr.tensors if t.name == "output.weight")
    d2t = np.array(keep, dtype=np.int64)
    out_rows = np.ascontiguousarray(t_out.data[d2t])  # Q8_0 rows, bytes, (N, 2176)

    w = gguf.GGUFWriter(a.out, arch=dr.fields["general.architecture"].contents())
    for field in dr.fields.values():
        if field.name == gguf.Keys.General.ARCHITECTURE or field.name.startswith("GGUF."):
            continue
        vt = field.types[0]
        st = field.types[-1] if vt == gguf.GGUFValueType.ARRAY else None
        w.add_key_value(field.name, field.contents(), vt, sub_type=st)
    w.add_key_value("dflash.frspec.n_kept", len(keep), gguf.GGUFValueType.UINT32)
    for t in dr.tensors:
        if t.name == "markov_w2.weight":
            assert t.tensor_type == gguf.GGMLQuantizationType.Q8_0, t.tensor_type
            w.add_tensor(t.name, np.ascontiguousarray(t.data[d2t]), raw_dtype=t.tensor_type)
        else:
            w.add_tensor(t.name, t.data, raw_dtype=t.tensor_type)
    w.add_tensor("output.weight", out_rows, raw_dtype=gguf.GGMLQuantizationType.Q8_0)
    w.add_tensor("d2t", d2t)
    w.write_header_to_file()
    w.write_kv_data_to_file()
    w.write_tensors_to_file()
    w.close()
    print(f"wrote {a.out}")


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    c = sub.add_parser("corpus")
    c.add_argument("out")
    b = sub.add_parser("build")
    b.add_argument("--ids", required=True)
    b.add_argument("--n", type=int, default=32768)
    b.add_argument("--out", required=True)
    a = ap.parse_args()
    if a.cmd == "corpus":
        corpus(a.out)
    else:
        build(a)


if __name__ == "__main__":
    sys.exit(main())
