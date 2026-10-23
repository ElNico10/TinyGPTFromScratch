"""Stage A of step 7: learned vs rotary positions, seeds 0 and 1, 5,000 steps each at lr 2e-3; then a paired summary.
Same seed = same initial weights (apart from the position table) and the same training batches.
Usage: python run_rope.py     logs: results/rope/<learned|rotary>_s<seed>.txt, summary: results/rope/summary.txt"""
import statistics as st
import subprocess
import sys
from pathlib import Path
from logs import parse_log

DATA = ["--text", "data/tinystories_train.npy", "--val-text", "data/tinystories_valid.npy", "--tokenizer", "tokenizer/mix4096.json"]
out = Path("results/rope")
out.mkdir(parents=True, exist_ok=True)
SEEDS, STEPS = (0, 1), "5000"

for seed in SEEDS:
    for name, extra in (("learned", []), ("rotary", ["--rope"])):
        log = out / f"{name}_s{seed}.txt"
        if log.exists() and "sample:" in log.read_text():
            continue  # finished in an earlier attempt
        with open(log, "w") as f:
            subprocess.run([sys.executable, "-u", "-W", "ignore", "gpt.py", *DATA, "--lr", "0.002", "--steps", STEPS, "--eval-every", "500", "--seed", str(seed), *extra],
                           stdout=f, check=True)
        print("done", name, seed, flush=True)

rows = {(n, s): parse_log(out / f"{n}_s{s}.txt")["rows"] for n in ("learned", "rotary") for s in SEEDS}
last = lambda r: r[-1]["bpb"]  # final validation bits per byte
tail = lambda r: st.mean(x["bpb"] for x in r[-3:])  # average of the last three evaluations (steps 4000, 4500, 5000), less jittery than one point
lines = [f"{'run':14}{'final bpb':>11}{'last-3 avg':>12}{'train bpb':>11}"]
for (n, s), r in rows.items():
    lines.append(f"{n} s{s:<8}{last(r):>11.4f}{tail(r):>12.4f}{r[-1]['train_bpb']:>11.4f}")
d_final = [last(rows["rotary", s]) - last(rows["learned", s]) for s in SEEDS]
d_tail = [tail(rows["rotary", s]) - tail(rows["learned", s]) for s in SEEDS]
lines.append("")
lines.append("rotary minus learned (negative = rotary better), per seed, then mean:")
lines.append("  final:      " + ", ".join(f"{d:+.4f}" for d in d_final) + f"   mean {st.mean(d_final):+.4f}")
lines.append("  last-3 avg: " + ", ".join(f"{d:+.4f}" for d in d_tail) + f"   mean {st.mean(d_tail):+.4f}")
lines.append(f"  the two learned seeds differ from each other by {abs(last(rows['learned', 0]) - last(rows['learned', 1])):.4f} (final), the two rotary seeds by {abs(last(rows['rotary', 0]) - last(rows['rotary', 1])):.4f}")
Path("results/rope/summary.txt").write_text("\n".join(lines) + "\n")
print("\n".join(lines))
