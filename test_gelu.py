"""Checks the fused GPU GELU kernels against the plain array-operation formula, on single operations and on a whole training step,
and times a step with each. Needs the GPU and models/tokens_big.npz. Usage: python test_gelu.py"""
import time
import numpy as np
import cupy as cp
from cupyx.profiler import benchmark
import gpt

gpt.xp = cp
fused = (gpt.gelu, gpt.gelu_back)


def plain_gelu(u):  # the formula before fusing, one array operation at a time
    return 0.5 * u * (1 + cp.tanh(0.7978845608028654 * (u + 0.044715 * u ** 3)))


def plain_back(dy, u):
    t = cp.tanh(0.7978845608028654 * (u + 0.044715 * u ** 3))
    return dy * (0.5 * (1 + t) + 0.5 * u * (1 - t * t) * 0.7978845608028654 * (1 + 3 * 0.044715 * u * u))


# 1. single operations, against a float64 reference computed on the CPU with the numpy formula
rng = np.random.default_rng(0)
u64, dy64 = rng.normal(0, 3, 2_000_000), rng.normal(0, 1, 2_000_000)
gpt.xp = np
ref_f, ref_b = gpt.gelu(u64), gpt.gelu_back(dy64, u64)
gpt.xp = cp
u, dy = cp.asarray(u64, dtype=cp.float32), cp.asarray(dy64, dtype=cp.float32)
for name, got, want in (("gelu", fused[0](u).get(), ref_f), ("gelu_back", fused[1](dy, u).get(), ref_b)):
    err = float(np.abs(got - want).max())
    scale = max(1.0, float(np.abs(want).max()))  # float32 rounding grows with the size of the values (plain float32 code has the same error)
    assert err < 1e-6 * scale, (name, err)
    print(f"{name}: largest difference from the float64 formula {err:.1e} on values up to {scale:.1f}")
empty = fused[0](cp.zeros((0, 4), dtype=cp.float32))
assert empty.shape == (0, 4)

# 2. a whole training step on real weights and a real batch: fused against the plain formula
z = np.load("models/tokens_big.npz")
p = {k: cp.asarray(z[k]) for k in z.files if k != "config"}
val = np.load("data/tinystories_valid.npy", mmap_mode="r")
s = np.random.default_rng(0).integers(0, len(val) - 257, 64)
w = cp.asarray(np.asarray(val[s[:, None] + np.arange(257)]).astype(np.int64))
x, y = w[:, :-1], w[:, 1:]


def step():
    return gpt.loss_and_grads(p, x, y, 6, 6)


loss_fused, g_fused = step()
gpt.gelu, gpt.gelu_back = plain_gelu, plain_back
loss_plain, g_plain = step()
worst = 0.0
for k in g_plain:
    a, b = g_plain[k].astype(cp.float64).ravel(), g_fused[k].astype(cp.float64).ravel()
    worst = max(worst, float(cp.linalg.norm(a - b) / cp.linalg.norm(a)))
assert abs(loss_fused - loss_plain) < 1e-5 and worst < 1e-4, (loss_fused, loss_plain, worst)
print(f"whole step: loss {loss_fused:.5f} (fused) vs {loss_plain:.5f} (plain); worst relative gradient difference over {len(g_plain)} arrays {worst:.1e}")

# 3. timing
ms = lambda: float(np.mean(benchmark(step, n_repeat=10, n_warmup=3).gpu_times)) * 1000
plain_ms = ms()
gpt.gelu, gpt.gelu_back = fused
fused_ms = ms()
print(f"step time: {plain_ms:.0f} ms plain, {fused_ms:.0f} ms fused -> {plain_ms / fused_ms:.2f}x")
