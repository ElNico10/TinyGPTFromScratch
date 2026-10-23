"""Type a beginning, get the model's continuation.
Usage: python chat.py models/tinystories_big.npz [--temperature 0.8] [--length 300] [--device cpu|gpu]   (length counts tokens: one character for the byte models)
Type 'quit' to leave. Lower temperature = safer and more repetitive, higher = more surprising and more mistakes."""
import argparse
import json
import numpy as np
import gpt

a = argparse.ArgumentParser()
a.add_argument("model")
a.add_argument("--temperature", type=float, default=0.8)
a.add_argument("--length", type=int, default=300)
a.add_argument("--device", default="cpu", choices=("cpu", "gpu"))  # cpu by default so it never competes with a training run for the GPU
a = a.parse_args()
if a.device == "gpu":
    import cupy
    gpt.xp = cupy
xp = gpt.xp
z = np.load(a.model)
cfg = json.loads(str(z["config"]))
p = {k: xp.asarray(z[k]) for k in z.files if k != "config"}
encode, decode = gpt.codec(cfg)  # byte models, the first character model, and token models
rng = np.random.default_rng(0)
print(f"{a.model}: {sum(w.size for w in p.values()):,} parameters, temperature {a.temperature}. Type a beginning (or just press Enter), 'quit' to leave.")
while True:
    text = input("\n> ")
    if text.strip() == "quit":
        break
    try:
        prompt = encode(text) or encode("\n")
    except ValueError:
        print("That contains a character the model never saw in training; try plainer text.")
        continue
    out = gpt.generate(p, cfg["heads"], cfg["layers"], cfg["ctx"], prompt, a.length, a.temperature, rng)
    print(decode(out).split("<|endoftext|>")[0])  # the stories in the training data are separated by this marker
