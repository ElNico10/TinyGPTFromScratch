"""A small GPT (decoder-only transformer) written from scratch in NumPy / CuPy, backprop by hand.
Usage: python gpt.py --check                         gradient check + causal-mask test, then exit
       python gpt.py --text data/x.txt --steps 3000  train a character-level model on a text file, print samples
"""
import argparse
import functools
import json
import time
from pathlib import Path
import numpy as np

xp = np  # becomes cupy with --device gpu


# ---- building blocks: each forward returns (output, cache); each backward takes the cache and the gradient of the output ----
def ln_forward(x, g, b, eps=1e-5):
    mu = x.mean(-1, keepdims=True)
    rstd = 1 / xp.sqrt(x.var(-1, keepdims=True) + eps)
    xhat = (x - mu) * rstd
    return xhat * g + b, (xhat, rstd, g)


def ln_backward(dy, cache):
    xhat, rstd, g = cache
    dxhat = dy * g
    dx = rstd * (dxhat - dxhat.mean(-1, keepdims=True) - xhat * (dxhat * xhat).mean(-1, keepdims=True))
    return dx, (dy * xhat).reshape(-1, xhat.shape[-1]).sum(0), dy.reshape(-1, xhat.shape[-1]).sum(0)


@functools.lru_cache(None)
def gelu_kernels():
    # One pass over the array each, where the formula written with array operations takes about ten passes (GELU was a third of a step).
    # shortcut: they compute in float32 whatever the input dtype, which is all this GPU training uses; upgrade if float64 GPU training is needed.
    import cupy
    fwd = cupy.ElementwiseKernel("T u", "T y", "float x = u; float t = tanhf(0.7978845608f * (x + 0.044715f * x * x * x)); y = 0.5f * x * (1.0f + t);", "gelu_fwd")
    back = cupy.ElementwiseKernel(
        "T dy, T u", "T du",
        "float x = u; float t = tanhf(0.7978845608f * (x + 0.044715f * x * x * x));"
        "du = dy * (0.5f * (1.0f + t) + 0.5f * x * (1.0f - t * t) * 0.7978845608f * (1.0f + 0.134145f * x * x));", "gelu_back")
    return fwd, back


def gelu(u):  # tanh approximation
    if xp is not np:
        return gelu_kernels()[0](u)
    return 0.5 * u * (1 + np.tanh(0.7978845608028654 * (u + 0.044715 * u ** 3)))


def gelu_back(dy, u):  # dy times the derivative of gelu at u
    if xp is not np:
        return gelu_kernels()[1](dy, u)
    t = np.tanh(0.7978845608028654 * (u + 0.044715 * u ** 3))
    return dy * (0.5 * (1 + t) + 0.5 * u * (1 - t * t) * 0.7978845608028654 * (1 + 3 * 0.044715 * u * u))


@functools.lru_cache(8)
def rope_tables(T, hd, dtype, gpu):
    # the angle for position t and dimension pair i is t * 10000^(-2i/hd); dtype and gpu are part of the cache key so the arrays match the library in use
    angles = np.outer(np.arange(T), 10000.0 ** (-np.arange(0, hd, 2) / hd))
    return xp.asarray(np.cos(angles), dtype=dtype), xp.asarray(np.sin(angles), dtype=dtype)


def rotate(x, cos, sin, back=False):
    # x: (B, heads, T, hd). The first half of each head's dimensions is rotated against the second half by an angle that grows with the position,
    # so a query and a key meet with a phase that depends only on how far apart they are. back=True rotates the other way (the gradient of the rotation).
    h = x.shape[-1] // 2
    x1, x2 = x[..., :h], x[..., h:]
    s = -sin if back else sin
    return xp.concatenate([x1 * cos - x2 * s, x2 * cos + x1 * s], axis=-1)


def init_params(vocab, d, layers, ctx, seed, dtype="float32", rope=False):
    # rope=True: positions come from rotating queries and keys in attention, so there is no learned position table (and "P" is absent from the weights)
    rng = np.random.default_rng(seed)
    n = lambda *s, std=0.02: xp.asarray(rng.normal(0, std, s), dtype=dtype)
    z = lambda *s: xp.zeros(s, dtype=dtype)
    o = lambda *s: xp.ones(s, dtype=dtype)
    p = {"E": n(vocab, d)}  # E also serves as the output layer (tied)
    P = n(ctx, d)  # always drawn, so every other weight starts identically with or without rope (a paired comparison)
    if not rope:
        p["P"] = P
    p.update({"lnfg": o(d), "lnfb": z(d)})
    for i in range(layers):
        p.update({f"{i}.ln1g": o(d), f"{i}.ln1b": z(d), f"{i}.Wqkv": n(d, 3 * d), f"{i}.bqkv": z(3 * d),
                  f"{i}.Wo": n(d, d, std=0.02 / np.sqrt(2 * layers)), f"{i}.bo": z(d),  # residual outputs are scaled down so the stack starts near identity
                  f"{i}.ln2g": o(d), f"{i}.ln2b": z(d), f"{i}.W1": n(d, 4 * d), f"{i}.b1": z(4 * d),
                  f"{i}.W2": n(4 * d, d, std=0.02 / np.sqrt(2 * layers)), f"{i}.b2": z(d)})
    return p


def forward(p, x, heads, layers, want_cache=True):
    # x: (B, T) integer tokens -> logits (B, T, vocab)
    B, T = x.shape
    d = p["E"].shape[1]
    hd = d // heads
    rope = "P" not in p  # a model without a position table uses rotary positions
    h = p["E"][x] if rope else p["E"][x] + p["P"][:T]
    if rope:
        cos, sin = rope_tables(T, hd, str(p["E"].dtype), xp is not np)
    mask = xp.triu(xp.ones((T, T), dtype=bool), 1)
    caches = []
    for i in range(layers):
        a, c1 = ln_forward(h, p[f"{i}.ln1g"], p[f"{i}.ln1b"])
        qkv = a @ p[f"{i}.Wqkv"] + p[f"{i}.bqkv"]
        q, k, v = (t.reshape(B, T, heads, hd).transpose(0, 2, 1, 3) for t in xp.split(qkv, 3, axis=-1))
        if rope:
            q, k = rotate(q, cos, sin), rotate(k, cos, sin)  # the rotated q and k are what attention uses, so those are what backprop needs
        s = q @ k.transpose(0, 1, 3, 2) / hd ** 0.5  # (a NumPy scalar here would silently turn everything into float64 on the GPU)
        s = xp.where(mask, -1e9, s)  # a token may not look at later tokens
        att = xp.exp(s - s.max(-1, keepdims=True))
        att /= att.sum(-1, keepdims=True)
        o = (att @ v).transpose(0, 2, 1, 3).reshape(B, T, d)
        h = h + o @ p[f"{i}.Wo"] + p[f"{i}.bo"]
        a2, c2 = ln_forward(h, p[f"{i}.ln2g"], p[f"{i}.ln2b"])
        u = a2 @ p[f"{i}.W1"] + p[f"{i}.b1"]
        h = h + gelu(u) @ p[f"{i}.W2"] + p[f"{i}.b2"]
        if want_cache:
            caches.append((c1, a, q, k, v, att, o, c2, a2, u))
    xf, cf = ln_forward(h, p["lnfg"], p["lnfb"])
    return xf @ p["E"].T, (x, caches, xf, cf)


def loss_and_grads(p, x, y, heads, layers):
    # y: (B, T) the token that follows each position
    logits, (x, caches, xf, cf) = forward(p, x, heads, layers)
    B, T = x.shape
    d = p["E"].shape[1]
    hd = d // heads
    rope = "P" not in p
    if rope:
        cos, sin = rope_tables(T, hd, str(p["E"].dtype), xp is not np)
    e = xp.exp(logits - logits.max(-1, keepdims=True))
    probs = e / e.sum(-1, keepdims=True)
    N = B * T
    loss = -xp.log(probs.reshape(N, -1)[xp.arange(N), y.reshape(-1)]).mean()
    dl = probs.reshape(N, -1)
    dl[xp.arange(N), y.reshape(-1)] -= 1
    dl /= N
    g = {"E": dl.T @ xf.reshape(N, d)}
    dxf = (dl @ p["E"]).reshape(B, T, d)
    dh, g["lnfg"], g["lnfb"] = ln_backward(dxf, cf)
    for i in reversed(range(layers)):
        c1, a, q, k, v, att, o, c2, a2, u = caches[i]
        flat = lambda t: t.reshape(N, -1)
        g[f"{i}.b2"] = flat(dh).sum(0)
        gu = gelu(u)
        g[f"{i}.W2"] = flat(gu).T @ flat(dh)
        du = gelu_back(dh @ p[f"{i}.W2"].T, u)
        g[f"{i}.b1"] = flat(du).sum(0)
        g[f"{i}.W1"] = flat(a2).T @ flat(du)
        dx2, g[f"{i}.ln2g"], g[f"{i}.ln2b"] = ln_backward(du @ p[f"{i}.W1"].T, c2)
        dh = dh + dx2
        g[f"{i}.bo"] = flat(dh).sum(0)
        g[f"{i}.Wo"] = flat(o).T @ flat(dh)
        do = (dh @ p[f"{i}.Wo"].T).reshape(B, T, heads, hd).transpose(0, 2, 1, 3)
        datt = do @ v.transpose(0, 1, 3, 2)
        dv = att.transpose(0, 1, 3, 2) @ do
        ds = att * (datt - (datt * att).sum(-1, keepdims=True)) / hd ** 0.5  # masked entries have att = 0, so they get no gradient
        dq = ds @ k
        dk = ds.transpose(0, 1, 3, 2) @ q
        if rope:
            dq, dk = rotate(dq, cos, sin, back=True), rotate(dk, cos, sin, back=True)  # undo the rotation: back to the gradient with respect to the unrotated q and k
        dqkv = xp.concatenate([t.transpose(0, 2, 1, 3).reshape(B, T, d) for t in (dq, dk, dv)], axis=-1)
        g[f"{i}.bqkv"] = flat(dqkv).sum(0)
        g[f"{i}.Wqkv"] = flat(a).T @ flat(dqkv)
        dx1, g[f"{i}.ln1g"], g[f"{i}.ln1b"] = ln_backward(dqkv @ p[f"{i}.Wqkv"].T, c1)
        dh = dh + dx1
    if not rope:
        g["P"] = xp.zeros_like(p["P"])
        g["P"][:T] = dh.sum(0)
    dE = g["E"]
    if xp is np:
        np.add.at(dE, x, dh)
    else:
        import cupyx
        cupyx.scatter_add(dE, x, dh)
    return float(loss), g


def eval_loss(p, x, y, heads, layers):  # the same loss without backprop: keeps no intermediates, so it uses far less memory
    logits = forward(p, x, heads, layers, want_cache=False)[0]
    logp = logits - logits.max(-1, keepdims=True)
    logp -= xp.log(xp.exp(logp).sum(-1, keepdims=True))
    return float(-xp.take_along_axis(logp, y[..., None], axis=-1).mean())


def adam_update(w, g, m, v, t, lr, b1=0.9, b2=0.95, eps=1e-8):  # in-place; b2 = 0.95 is the usual choice for transformers
    m *= b1
    m += (1 - b1) * g
    v *= b2
    v += (1 - b2) * g * g
    w -= lr * (m / (1 - b1 ** t)) / (xp.sqrt(v / (1 - b2 ** t)) + eps)


def lr_at(step, total, base, warmup):  # linear warmup, then cosine decay to 10% of the peak
    if step < warmup:
        return base * (step + 1) / warmup
    return float(base * (0.1 + 0.9 * 0.5 * (1 + np.cos(np.pi * (step - warmup) / max(1, total - warmup)))))  # plain float, same reason


def generate(p, heads, layers, ctx, prompt, n, temperature, rng):
    toks = list(prompt)
    for _ in range(n):  # no key/value cache: every new token reruns the whole window (fine at this size)
        logits = forward(p, xp.asarray([toks[-ctx:]]), heads, layers, want_cache=False)[0][0, -1]
        pr = xp.exp((logits - logits.max()) / temperature)
        pr = pr / pr.sum()
        pr = np.asarray(pr.get() if hasattr(pr, "get") else pr, dtype=np.float64)
        toks.append(int(rng.choice(len(pr), p=pr / pr.sum())))
    return toks


def encode(path, vocab=None):
    # file -> array of token ids, one per byte; vocab = the distinct bytes seen (taken from the file unless given)
    raw = np.frombuffer(Path(path).read_bytes(), dtype=np.uint8)
    if vocab is None:
        vocab = np.flatnonzero(np.bincount(raw, minlength=256))  # the distinct byte values, without sorting billions of bytes
    assert len(vocab) < 255, "a byte-level model needs fewer than 255 distinct bytes"
    lut = np.full(256, 255, dtype=np.uint8)  # uint8 ids keep a 2GB file at 2GB; 255 marks a byte the vocabulary lacks
    lut[vocab] = np.arange(len(vocab), dtype=np.uint8)
    ids = lut[raw]
    assert not (ids == 255).any(), f"{path} contains bytes that are not in the training vocabulary"
    return ids, vocab


def load_data(path, vocab=None):
    # a .npy file holds token ids made by tok.py (memory-mapped, never fully loaded); anything else is raw text read as bytes
    if str(path).endswith(".npy"):
        return np.load(path, mmap_mode="r"), None
    return encode(path, vocab)


def codec(cfg):
    # (encode: str -> ids, decode: ids -> str) for a saved model's config
    if cfg.get("tokenizer"):
        import tok
        t = tok.Tokenizer.load(cfg["tokenizer"])
        return t.encode, t.decode
    vocab = cfg["chars"]  # byte values (byte models) or single characters (the very first model)
    if isinstance(vocab[0], int):
        return (lambda s: [vocab.index(b) for b in s.encode("utf-8")]), (lambda t: bytes(vocab[i] for i in t).decode("utf-8", "replace"))
    return (lambda s: [vocab.index(c) for c in s]), (lambda t: "".join(vocab[i] for i in t))


def host(w):
    return np.asarray(w.get() if hasattr(w, "get") else w)


def save(path, p, args, chars, step=0, opt=None):
    # the model file holds weights + settings; the optimizer state (needed only to resume) goes in a second file next to it
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    np.savez(path, config=json.dumps({**vars(args), "chars": chars, "trained_steps": step}), **{k: host(w) for k, w in p.items()})
    if opt:
        m, v = opt
        np.savez(str(path) + ".opt.npz", step=step, **{f"m.{k}": host(w) for k, w in m.items()}, **{f"v.{k}": host(w) for k, w in v.items()})


def check():
    global xp
    xp = np
    for rope in (False, True):
        check_model(rope)
    check_rotation()


def check_rotation():
    rng = np.random.default_rng(1)
    B, H, T, hd = 2, 3, 9, 8
    x = rng.normal(size=(B, H, T, hd))
    cos, sin = rope_tables(T, hd, "float64", False)
    assert np.abs(rotate(rotate(x, cos, sin), cos, sin, back=True) - x).max() < 1e-12
    assert np.allclose(np.linalg.norm(rotate(x, cos, sin), axis=-1), np.linalg.norm(x, axis=-1))  # a rotation never changes a vector's length
    # the defining property: put the same query and key at every position; the score between positions i and j must then depend only on i - j
    q, k = (np.broadcast_to(rng.normal(size=(1, H, 1, hd)), (1, H, T, hd)) for _ in range(2))
    s = rotate(q, cos, sin) @ rotate(k, cos, sin).transpose(0, 1, 3, 2)
    assert np.abs(s[:, :, :-1, :-1] - s[:, :, 1:, 1:]).max() < 1e-12, "scores must depend only on the distance between positions"
    assert np.abs(s[:, :, 0, 0] - s[:, :, 0, 1]).max() > 1e-3, "the rotation must actually change the scores with distance"
    # starting the sequence 7 positions later changes nothing
    cos2, sin2 = rope_tables(T + 7, hd, "float64", False)
    q, k = rng.normal(size=(2, 1, H, T, hd))
    a = rotate(q, cos, sin) @ rotate(k, cos, sin).transpose(0, 1, 3, 2)
    b = rotate(q, cos2[7:], sin2[7:]) @ rotate(k, cos2[7:], sin2[7:]).transpose(0, 1, 3, 2)
    assert np.abs(a - b).max() < 1e-10
    print("rotation ok: inverts exactly, keeps lengths, scores depend only on distance, shifting every position changes nothing")


def check_model(rope):
    rng = np.random.default_rng(0)
    V, d, L, H, T = 11, 16, 2, 4, 6
    p = init_params(V, d, L, T, 0, "float64", rope)
    label = "rotary positions" if rope else "learned positions"
    for k in p:  # random values everywhere (also biases and layer-norm gains) so no gradient is trivially zero
        p[k] = p[k] + rng.normal(0, 0.3, p[k].shape)
    x, y = rng.integers(0, V, (3, T)), rng.integers(0, V, (3, T))
    _, g = loss_and_grads(p, x, y, H, L)
    worst = 0
    for k in p:
        j = np.unravel_index(np.abs(g[k]).argmax(), g[k].shape)  # the largest-gradient entry, so a zero gradient cannot pass
        old, eps = p[k][j], 1e-6
        p[k][j] = old + eps
        lp = loss_and_grads(p, x, y, H, L)[0]
        p[k][j] = old - eps
        lm = loss_and_grads(p, x, y, H, L)[0]
        p[k][j] = old
        num = (lp - lm) / (2 * eps)
        rel = abs(num - g[k][j]) / max(abs(num), abs(g[k][j]), 1e-8)
        assert abs(g[k][j]) > 1e-9 and rel < 1e-5, f"gradient mismatch in {k}: analytic {g[k][j]}, numeric {num}"
        worst = max(worst, rel)
    print(f"gradient check ok ({label}): {len(p)} parameter arrays, worst relative error {worst:.1e}")
    assert abs(eval_loss(p, x, y, H, L) - loss_and_grads(p, x, y, H, L)[0]) < 1e-12
    x2 = x.copy()
    x2[:, -1] = (x2[:, -1] + 1) % V  # change only the last token
    a, b = forward(p, x, H, L)[0], forward(p, x2, H, L)[0]
    assert np.abs(a[:, :-1] - b[:, :-1]).max() == 0 and np.abs(a[:, -1] - b[:, -1]).max() > 0
    print(f"causal mask ok ({label}): changing the last token leaves every earlier prediction unchanged")


def main():
    global xp
    a = argparse.ArgumentParser()
    a.add_argument("--check", action="store_true")
    a.add_argument("--text", default="")
    a.add_argument("--val-text", default="", help="validate on this separate file instead of holding out the last 10% of --text")
    a.add_argument("--save-every", type=int, default=0, help="also write the --save file every this many steps")
    a.add_argument("--device", default="gpu", choices=("cpu", "gpu"))
    a.add_argument("--d", type=int, default=384)
    a.add_argument("--layers", type=int, default=6)
    a.add_argument("--heads", type=int, default=6)
    a.add_argument("--ctx", type=int, default=256)
    a.add_argument("--batch", type=int, default=64)
    a.add_argument("--steps", type=int, default=3000)
    a.add_argument("--lr", type=float, default=1e-3)
    a.add_argument("--warmup", type=int, default=100)
    a.add_argument("--weight-decay", type=float, default=0.1)
    a.add_argument("--seed", type=int, default=0)
    a.add_argument("--eval-every", type=int, default=250)
    a.add_argument("--save", default="")
    a.add_argument("--resume", default="", help="continue a run from this saved model (its .opt.npz file must sit next to it); --steps is the total, not the extra")
    a.add_argument("--rope", action="store_true", help="rotary positions in attention instead of a learned position table")
    a.add_argument("--stop-at", type=int, default=0, help="stop (and save) after this step instead of at --steps")
    a.add_argument("--tokenizer", default="", help="tokenizer file from tok.py; required when --text is a .npy token file")
    a = a.parse_args()
    if a.check:
        return check()
    if a.device == "gpu":
        import cupy
        xp = cupy
    train, vocab = load_data(a.text)  # bytes: one token per byte; a .npy file: token ids from tok.py
    if a.val_text:
        val = load_data(a.val_text, vocab)[0]  # a separate file the model never trains on
    else:
        cut = int(0.9 * len(train))
        train, val = train[:cut], train[cut:]  # the last 10% of the text is held out
    if vocab is None:  # token data: the tokenizer fixes the vocabulary, and the model must carry it to decode text later
        assert a.tokenizer, "--tokenizer is required when --text is a token file"
        import tok
        tokenizer = tok.Tokenizer.load(a.tokenizer)
        size, chars = tokenizer.size, None
        bytes_per_token = float(tokenizer.lengths[np.asarray(val[:5_000_000])].mean())  # the average bytes a token stands for, measured on the validation text
    else:
        size, chars, bytes_per_token = len(vocab), vocab.tolist(), 1.0
    if a.resume:
        z = np.load(a.resume)
        assert all(json.loads(str(z["config"]))[k] == getattr(a, k.replace("-", "_")) for k in ("d", "layers", "heads", "ctx")), "--resume needs the same d, layers, heads and ctx"
        p = {k: xp.asarray(z[k]) for k in z.files if k != "config"}
        assert ("P" not in p) == a.rope, "--resume needs the same positions: pass --rope exactly when the saved model used it"
        o = np.load(a.resume + ".opt.npz")
        start = int(o["step"])
        m, v = ({k: xp.asarray(o[f"{s}.{k}"]) for k in p} for s in "mv")
    else:
        p, start = init_params(size, a.d, a.layers, a.ctx, a.seed, rope=a.rope), 0
        m, v = ({k: xp.zeros_like(w) for k, w in p.items()} for _ in range(2))
    nparams = sum(w.size for w in p.values())
    print(f"{a}\n{len(train):,} training tokens, {len(val):,} validation tokens, vocabulary {size}, {bytes_per_token:.2f} bytes per token, {nparams:,} parameters"
          + (f"\nresuming from step {start}" if start else "") + "\n")

    def batch(data, n, rng):
        s = rng.integers(0, len(data) - a.ctx - 1, n)
        idx = s[:, None] + np.arange(a.ctx + 1)
        w = xp.asarray(np.asarray(data[idx]).astype(np.int64))
        return w[:, :-1], w[:, 1:]

    def val_loss(count=8):  # the same 8 validation batches at every evaluation, so the curve is not jittered by which batches came up
        rng = np.random.default_rng(999)
        return float(np.mean([eval_loss(p, *batch(val, a.batch, rng), a.heads, a.layers) for _ in range(count)]))

    t0 = time.time()
    end = a.stop_at or a.steps  # --stop-at ends a run early (the learning-rate schedule still spans --steps), to be continued with --resume
    for step in range(start + 1, end + 1):
        rng = np.random.default_rng((a.seed, step))  # one random stream per step: a resumed run draws exactly the batches an uninterrupted run would
        loss, g = loss_and_grads(p, *batch(train, a.batch, rng), a.heads, a.layers)
        lr = lr_at(step - 1, a.steps, a.lr, a.warmup)
        for k in p:
            if k.endswith(("Wqkv", "Wo", "W1", "W2")):
                p[k] *= 1 - lr * a.weight_decay  # decoupled weight decay on the weight matrices only
            adam_update(p[k], g[k], m[k], v[k], step, lr)
        if step % a.eval_every == 0 or step == end:
            vl = val_loss()
            print(f"step {step:5d}  train loss {loss:.4f}  val loss {vl:.4f}  ({time.time() - t0:.0f}s)  bpb {vl / np.log(2) / bytes_per_token:.4f}", flush=True)  # bpb = validation bits per byte
        if a.save and a.save_every and step % a.save_every == 0:
            save(a.save, p, a, chars, step, (m, v))  # a long run that crashes keeps its latest checkpoint
    encode_text, decode_text = codec({"tokenizer": a.tokenizer if vocab is None else None, "chars": chars})
    start_ids = encode_text("<|endoftext|>")
    sample = generate(p, a.heads, a.layers, a.ctx, start_ids, 400, 0.8, np.random.default_rng(1))
    print("\nsample:\n" + decode_text(sample[len(start_ids):]).split("<|endoftext|>")[0])
    if a.save:
        save(a.save, p, a, chars, end, (m, v))


if __name__ == "__main__":
    main()
