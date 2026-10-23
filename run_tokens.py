"""Step 6: learning-rate sweep (1,000 steps each) on the token data, then one 15,000-step run at the best rate.
If the best rate is the lowest or highest tried, the sweep is extended outward (up to twice) before the long run starts.
Usage: python run_tokens.py     logs: results/tokens/lr*.txt and results/tokens/big.txt, model: models/tokens_big.npz"""
import re
import subprocess
import sys
from pathlib import Path

DATA = ["--text", "data/tinystories_train.npy", "--val-text", "data/tinystories_valid.npy", "--tokenizer", "tokenizer/mix4096.json"]
out = Path("results/tokens")
out.mkdir(parents=True, exist_ok=True)


def run(log, *flags):
    with open(out / log, "w") as f:
        subprocess.run([sys.executable, "-u", "-W", "ignore", "gpt.py", *DATA, *flags], stdout=f, check=True)
    return (out / log).read_text()


rates = [5e-4, 1e-3, 2e-3]  # the same three the byte model was swept over
final = {}
for attempt in range(3):
    for lr in rates:
        if lr not in final:
            text = run(f"lr{lr:g}.txt", "--lr", str(lr), "--steps", "1000", "--eval-every", "250")
            final[lr] = float(re.findall(r"val loss ([\d.]+)", text)[-1])  # validation loss at step 1000
            print(f"lr {lr:g}  val loss {final[lr]}", flush=True)
    best = min(final, key=final.get)
    if best == min(final) and attempt < 2:
        rates.append(best / 2)
    elif best == max(final) and attempt < 2:
        rates.append(best * 2)
    else:
        break
    print(f"best so far {best:g} is at the edge of the range; trying {rates[-1]:g} too", flush=True)
print(f"best lr {best:g}", flush=True)
run("big.txt", "--lr", str(best), "--steps", "15000", "--eval-every", "500", "--save-every", "2000", "--save", "models/tokens_big.npz")
print("done", flush=True)
