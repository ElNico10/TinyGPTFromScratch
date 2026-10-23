"""Learning-rate sweep (3 short runs), then one long run at the best rate, all on the full TinyStories training file.
Usage: python run_long.py         logs: results/long/lr*.txt and results/long/big.txt, model: models/tinystories_big.npz"""
import re
import subprocess
import sys
from pathlib import Path

TRAIN, VAL = "data/TinyStoriesV2-GPT4-train.txt", "data/TinyStoriesV2-GPT4-valid.txt"
out = Path("results/long")
out.mkdir(parents=True, exist_ok=True)


def run(log, *flags):
    with open(out / log, "w") as f:
        subprocess.run([sys.executable, "-u", "-W", "ignore", "gpt.py", "--text", TRAIN, "--val-text", VAL, *flags], stdout=f, check=True)
    return (out / log).read_text()


final = {}
for lr in ("0.0005", "0.001", "0.002"):
    text = run(f"lr{lr}.txt", "--lr", lr, "--steps", "1000", "--eval-every", "250")
    final[lr] = float(re.findall(r"val loss ([\d.]+)", text)[-1])  # validation loss at the last step
    print("lr", lr, "val loss", final[lr], flush=True)
best = min(final, key=final.get)
print("best lr", best, flush=True)
run("big.txt", "--lr", best, "--steps", "15000", "--eval-every", "500", "--save-every", "2000", "--save", "models/tinystories_big.npz")
print("done", flush=True)
