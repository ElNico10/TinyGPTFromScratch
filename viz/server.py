"""Local viewer for gpt.py training runs, plus a prompt box. Run:  python viz/server.py   then open http://127.0.0.1:8766
Read-only: it lists the logs in results/ and the saved models in models/, and generates text on the CPU by default (so it never
competes with a training run for the GPU) or on the GPU when the page asks. Bound to this machine only."""
import argparse
import json
import os
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
os.chdir(ROOT)
import gpt
from logs import STEP, parse_log

gpt.xp = np
MODELS, LOCK = {}, threading.Lock()


def safe(rel, folder):
    p = (ROOT / rel).resolve()
    if not p.is_relative_to((ROOT / folder).resolve()):
        raise ValueError("path outside " + folder)
    return p


def list_runs():
    out = []
    for f in (ROOT / "results").rglob("*.txt"):
        if not STEP.search(f.read_text(encoding="utf-8", errors="replace")):
            continue
        d = parse_log(f)
        out.append(dict(path=f.relative_to(ROOT).as_posix(), mtime=f.stat().st_mtime, done=d["rows"][-1]["step"], total=d["total"], running=d["running"]))
    return sorted(out, key=lambda r: -r["mtime"])


def list_models():
    out = []
    for f in (ROOT / "models").glob("*.npz"):
        try:
            cfg = json.loads(str(np.load(f)["config"]))
        except Exception:
            continue
        out.append(dict(path=f.relative_to(ROOT).as_posix(), mtime=f.stat().st_mtime, d=cfg["d"], layers=cfg["layers"], steps=cfg["steps"]))
    return sorted(out, key=lambda r: -r["mtime"])


def load(path, xp):
    key = (path, xp.__name__)
    if key not in MODELS:
        z = np.load(safe(path, "models"))
        MODELS[key] = ({k: xp.asarray(z[k]) for k in z.files if k != "config"}, json.loads(str(z["config"])))
    return MODELS[key]


def generate(path, prompt, length, temperature, device="cpu"):
    if device == "gpu":
        import cupy as xp
    else:
        xp = np
    with LOCK:  # gpt.py reads its array library from a module global, so only one request may switch it at a time
        gpt.xp = xp
        return generate_text(path, prompt, length, temperature, xp)


def generate_text(path, prompt, length, temperature, xp):
    p, cfg = load(path, xp)
    encode, decode = gpt.codec(cfg)  # byte models, the first character model, and token models
    try:
        ids = encode(prompt) or encode("\n")
    except ValueError:
        raise ValueError("the prompt has a character the model never saw in training; try plainer text")
    length, temperature = min(max(int(length), 1), 600), min(max(float(temperature), 0.1), 2.0)
    t0 = time.time()
    out = gpt.generate(p, cfg["heads"], cfg["layers"], cfg["ctx"], ids, length, temperature, np.random.default_rng())
    text = decode(out)
    shown = text.split("<|endoftext|>")[0]
    return dict(text=shown, ended="<|endoftext|>" in text, secs=round(time.time() - t0, 2), n=length, chars=len(shown) - len(prompt))


class Handler(BaseHTTPRequestHandler):
    def log_message(self, *a):
        pass

    def reply(self, body, kind="application/json", code=200):
        data = body if isinstance(body, bytes) else json.dumps(body).encode()
        self.send_response(code)
        self.send_header("Content-Type", kind)
        self.send_header("Content-Length", str(len(data)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(data)

    def do_GET(self):
        url = urlparse(self.path)
        q = {k: v[0] for k, v in parse_qs(url.query).items()}
        try:
            if url.path == "/":
                return self.reply((ROOT / "viz" / "index.html").read_bytes(), "text/html; charset=utf-8")
            if url.path == "/api/runs":
                return self.reply(list_runs())
            if url.path == "/api/models":
                return self.reply(list_models())
            if url.path == "/api/log":
                return self.reply(parse_log(safe(q["path"], "results")))
            if url.path == "/api/schedule":  # the learning rate at every step of a run, from its settings
                a = parse_log(safe(q["path"], "results"))["args"]
                n, lr, warm = int(a["steps"]), float(a["lr"]), int(a["warmup"])
                step = max(1, n // 200)
                return self.reply(dict(steps=list(range(0, n, step)), lr=[gpt.lr_at(s, n, lr, warm) for s in range(0, n, step)]))
            return self.reply({"error": "not found"}, code=404)
        except Exception as e:  # report problems to the page instead of dropping the connection
            return self.reply({"error": f"{type(e).__name__}: {e}"}, code=400)

    def do_POST(self):
        try:
            n = int(self.headers.get("Content-Length", 0))
            if n > 100_000:
                raise ValueError("request too large")
            body = json.loads(self.rfile.read(n))
            if urlparse(self.path).path == "/api/generate":
                return self.reply(generate(body["path"], body["prompt"], body["length"], body["temperature"], body.get("device", "cpu")))
            return self.reply({"error": "not found"}, code=404)
        except Exception as e:
            return self.reply({"error": f"{type(e).__name__}: {e}"}, code=400)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8766)
    args = ap.parse_args()
    print(f"viewer on http://127.0.0.1:{args.port}  (Ctrl+C to stop)", flush=True)
    ThreadingHTTPServer(("127.0.0.1", args.port), Handler).serve_forever()
