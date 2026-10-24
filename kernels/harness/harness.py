"""Bit-exact comparison of two libggml-cuda builds (or valve on/off) on MiniCPM5-2B shapes, Linux port of
harness_compare.py (kmcG lab, 28 Sep).

usage: python3 harness.py <tag> <libdirA>[@ENV=V...] <libdirB>[@ENV=V...] [--quick | --fa] [--iters N] [--times]
  libdir = folder with libggml*.so (e.g. llama.cpp-ref/build/bin or llama.cpp-pascal/build/bin);
  kmc_check is linked without rpath, LD_LIBRARY_PATH selects the build; the loaded lib path is checked per run.
  Env vars after @ apply to that side only. Nothing else should run on the GPU.
Writes ./results/<tag>.harness.txt; exit code 1 if any output differs.
"""
import os
import subprocess
import sys
import time
from pathlib import Path


class BenchLock:
    """No-op context manager (the original serialized GPU jobs between several agents on one machine)."""

    def __init__(self, *a):
        pass

    def __enter__(self):
        return self

    def __exit__(self, *a):
        pass


HERE = Path(__file__).resolve().parent
BIN = HERE / "kmc_check"
OUT = Path(os.environ.get("HARNESS_OUT", "results"))

CASES = []
for n in range(1, 9):
    CASES += [f"mm 2048 {n} 2048", f"mm 256 {n} 2048", f"mm 6144 {n} 2048", f"mm 2048 {n} 6144",
              f"glu 6144 {n} 2048", f"mmadd 2048 {n} 2048", f"mmadd 2048 {n} 6144", f"qkv {n}"]
CASES += ["mm 130560 1 2048", "mm 130560 6 2048", "mm 130 3 2048", "glu 130 6 2048", "mmadd 130 6 6144",
          "glu 130 1 2048", "mmadd 130 1 6144", "mmadd 2048 1 2048", "mm 130560 5 2048",
          "mm 130560 1 256", "mm 130560 5 256", "mmadd 130560 1 256", "mmadd 130560 5 256"]
for kv in (512, 4096, 8192, 12288):
    for nb in (1, 2, 6):
        CASES.append(f"fa {kv} {nb}")
CASES += ["fa 8192 3", "fa 8192 4", "fa 8192 5", "fa 2048 7", "fa 4096 512", "fa 8192 1024"]
FA_CASES = [c for c in CASES if c.startswith("fa ")]
QUICK = ["mm 2048 1 2048", "mm 2048 6 2048", "glu 6144 1 2048", "glu 6144 6 2048", "mmadd 2048 6 2048", "qkv 1",
         "qkv 6", "mm 130560 5 256", "fa 8192 1", "fa 8192 6"]


def run(spec, case, iters):
    libdir, *envs = spec.split("@")
    libdir = str(Path(libdir).expanduser().resolve())
    env = dict(os.environ)
    env["LD_LIBRARY_PATH"] = libdir
    for kv in envs:
        k, v = kv.split("=", 1)
        env[k] = v
    r = subprocess.run([str(BIN), *case.split(), "--iters", str(iters)], capture_output=True, text=True,
                       env=env, timeout=600)
    loaded = [l[4:] for l in r.stderr.splitlines() if l.startswith("lib ")]
    if not loaded or not loaded[0].startswith(libdir + "/"):
        return None, f"WRONG LIB: wanted {libdir}, loaded {loaded}"
    line = [l for l in r.stdout.splitlines() if "fnv" in l]
    if not line:
        return None, (r.stdout + r.stderr)[-400:]
    return line[0].split("fnv ")[1].split()[0], line[0]


def main():
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    iters = 3
    if "--iters" in sys.argv:
        iters = int(sys.argv[sys.argv.index("--iters") + 1])
        args.remove(str(iters))
    tag, a, b = args[:3]
    cases = QUICK if "--quick" in sys.argv else FA_CASES if "--fa" in sys.argv else CASES
    OUT.mkdir(parents=True, exist_ok=True)
    out = OUT / f"{tag}.harness.txt"
    if out.exists():
        sys.exit(f"{out} exists: tags are never overwritten")
    bad = 0
    with BenchLock("SCRITTORE", f"harness12.py {tag} (kmc_check, GPU)", 15), open(out, "w", encoding="utf-8") as f:
        f.write(f"A={a}\nB={b}\n")
        for c in cases:
            ha, la = run(a, c, iters)
            hb, lb = run(b, c, iters)
            ok = ha is not None and ha == hb
            bad += not ok
            line = f"{'SAME' if ok else 'DIFF'} {c:22} A {ha} B {hb}"
            if not ok or "--times" in sys.argv:
                line += f"\n   A: {la}\n   B: {lb}"
            print(line, flush=True)
            f.write(line + "\n")
        f.write(f"cases {len(cases)} different {bad}\n")
    print(f"cases {len(cases)} different {bad}")
    sys.exit(1 if bad else 0)


if __name__ == "__main__":
    main()
