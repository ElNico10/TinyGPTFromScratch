"""Checks that a run stopped at step 3 and resumed gives exactly the same weights as one that ran straight to step 6 (with learned and with
rotary positions), and that the logged bits per byte equals val loss / ln 2 for a byte model. Tiny CPU model, takes seconds.
Usage: python test_resume.py"""
import re
import subprocess
import sys
import tempfile
from pathlib import Path
import numpy as np

TEXT = "data/TinyStoriesV2-GPT4-valid.txt"
tmp = Path(tempfile.mkdtemp())
BASE = [sys.executable, "-W", "ignore", "gpt.py", "--device", "cpu", "--text", TEXT, "--val-text", TEXT, "--d", "32", "--layers", "2", "--heads", "2",
        "--ctx", "32", "--batch", "4", "--steps", "6", "--eval-every", "3", "--warmup", "2"]


def run(*flags):
    return subprocess.run([*BASE, *flags], capture_output=True, text=True, check=True).stdout


for name, extra in (("learned positions", []), ("rotary positions", ["--rope"])):
    full = run(*extra, "--save", str(tmp / "full.npz"))
    run(*extra, "--stop-at", "3", "--save", str(tmp / "half.npz"))
    resumed = run(*extra, "--resume", str(tmp / "half.npz"), "--save", str(tmp / "resumed.npz"))
    a, b = np.load(tmp / "full.npz"), np.load(tmp / "resumed.npz")
    assert "resuming from step 3" in resumed
    bad = [k for k in a.files if k != "config" and not np.array_equal(a[k], b[k])]
    assert not bad, f"resumed weights differ in {bad}"
    assert ("P" in a.files) == (not extra)
    print(f"resume ok ({name}): {len(a.files) - 1} weight arrays identical after stop at 3 + resume to 6")
loss, bpb = (float(x) for x in re.findall(r"val loss ([\d.]+).*bpb ([\d.]+)", full)[-1])
assert abs(bpb - loss / np.log(2)) < 1e-3, (loss, bpb)  # a byte model: one token = one byte
print(f"bits per byte ok: val loss {loss} -> {bpb} bits per byte (loss / ln 2)")
