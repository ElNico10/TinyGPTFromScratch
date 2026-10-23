"""Reads gpt.py training logs. Used by plot_log.py and viz/server.py."""
import math
import re
import time
from pathlib import Path

STEP = re.compile(r"step\s+(\d+)\s+train loss ([\d.]+)\s+val loss ([\d.]+)\s+\((\d+)s\)(?:\s+bpb ([\d.]+))?")


def parse_log(path):
    path = Path(path)
    text = path.read_text(encoding="utf-8", errors="replace")
    rows = []
    for a, b, c, d, e in STEP.findall(text):
        # bpb = validation bits per byte, comparable across tokenizers; older byte-model logs have no bpb column, and for them it is loss / ln 2
        bpb = float(e) if e else float(c) / math.log(2)
        rows.append(dict(step=int(a), train=float(b), val=float(c), secs=int(d), bpb=bpb, train_bpb=float(b) * bpb / float(c)))
    ns = re.search(r"Namespace\((.*?)\)\s*\n", text, re.S)
    args = {k: v.strip("'") for k, v in re.findall(r"(\w+)=('[^']*'|[^,)]+)", ns.group(1))} if ns else {}
    params = re.search(r"([\d,]+) parameters", text)
    total = int(args.get("steps", 0) or 0)
    return dict(rows=rows, args=args, params=int(params[1].replace(",", "")) if params else None, total=total,
                sample=text.split("sample:\n", 1)[1].strip() if "sample:\n" in text else None,
                running=bool(rows) and rows[-1]["step"] < total and time.time() - path.stat().st_mtime < 300)
