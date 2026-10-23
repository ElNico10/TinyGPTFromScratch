"""Plot training and validation loss from gpt.py logs, one colour per log.
Usage: python plot_log.py results/long/lr*.txt results/long/big.txt [-o results/curves.png]"""
import argparse
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path
from logs import parse_log

ap = argparse.ArgumentParser()
ap.add_argument("logs", nargs="+")
ap.add_argument("-o", default="results/curves.png")
args = ap.parse_args()
fig, ax = plt.subplots(figsize=(9, 5))
for i, path in enumerate(args.logs):
    rows = parse_log(path)["rows"]
    if not rows:
        print("no training lines in", path)
        continue
    c, name = f"C{i}", f"{Path(path).parent.name}/{Path(path).stem}"  # the folder tells byte runs from token runs
    ax.plot([r["step"] for r in rows], [r["bpb"] for r in rows], color=c, label=f"{name} validation")
    ax.plot([r["step"] for r in rows], [r["train_bpb"] for r in rows], color=c, ls=":", alpha=.7, label=f"{name} train")
ax.set(xlabel="step", ylabel="bits per byte (lower is better; comparable across tokenizers)", yscale="log", title="gpt.py training")
ax.grid(alpha=.3, which="both")
ax.legend()
fig.tight_layout()
fig.savefig(args.o, dpi=130)
print("wrote", args.o)
