# TinyGPT from scratch

A small GPT (decoder-only transformer) written in NumPy / CuPy with hand-written backpropagation, no PyTorch. It includes a byte-pair tokenizer, rotary positions, a fused GPU GELU kernel, checkpoint resume, and a local web viewer. Built step by step as a learning project, with help from Claude Code.

Trained on [TinyStories](https://huggingface.co/datasets/roneneldan/TinyStories) (`TinyStoriesV2-GPT4-train.txt` and `-valid.txt`) on one RTX 5070 (12 GB).

## Results

Validation bits per byte on TinyStories (lower is better; one run each, 15,000 steps at batch 64 x 256 tokens, so differences under about 0.002 are noise):

| Model | Parameters | Bits per byte | Training time |
|---|---|---|---|
| Byte level, learned positions | 10.8M | 0.713 | 62 min |
| 4,096-token BPE, learned positions | 12.3M | 0.484 | 64 min |
| 4,096-token BPE, rotary positions | 12.2M | **0.478** | 47 min (with the fused GELU kernel) |

Rotary positions beat learned positions by 0.012 bits per byte in a 2-seed comparison at 5,000 steps, and by 0.007 in a single 15,000-step run. The fused GELU kernel makes a training step 1.43x faster (256 to 179 ms). Mixed precision was measured and rejected (TF32 1.03x; float16 1.21x with gradient errors of 7-40%).

## Files

| File | Purpose |
|---|---|
| `gpt.py` | The model, backprop, training loop, gradient checks (`--check`), resume, rotary option (`--rope`) |
| `tok.py` | Byte-level BPE tokenizer: `train`, `encode`, `test` |
| `prep_mix.py` | Turns two chat/code JSON-lines files into plain text with `<|user|>` / `<|assistant|>` markers |
| `chat.py` | Terminal prompt: type a beginning, get a continuation |
| `viz/` | Local web viewer: training curves (live), prompt box, CPU/GPU toggle (`python viz/server.py`, then http://127.0.0.1:8766) |
| `logs.py`, `plot_log.py` | Read training logs; plot loss in bits per byte |
| `run_*.py` | The experiments behind the table above |
| `test_*.py` | Resume, fused-GELU and other checks |
| `tokenizer/` | The trained 4,096-token tokenizers |
| `results/` | Training logs and plots |

## Train on your own text

You need Python with NumPy; for speed an NVIDIA GPU with CuPy (`--device cpu` works but is slow). The default model uses about 5 GB of GPU memory.

1. Put your text in UTF-8 files: one for training, one for validation. For chat data, a format that matches the reserved markers is `<|user|>\nquestion\n<|assistant|>\nanswer<|endoftext|>\n` (see `prep_mix.py`).
2. Train a tokenizer (the number is the vocabulary size; use fewer, such as 1024-2048, for small data):
   ```
   python tok.py train tokenizer/mine.json 4096 data/mine_train.txt
   python tok.py encode tokenizer/mine.json data/mine_train.txt data/mine_train.npy
   python tok.py encode tokenizer/mine.json data/mine_valid.txt data/mine_valid.npy
   ```
3. Train:
   ```
   python gpt.py --text data/mine_train.npy --val-text data/mine_valid.npy --tokenizer tokenizer/mine.json --rope --lr 0.002 --steps 5000 --save models/mine.npz --save-every 1000
   ```
   `--resume models/mine.npz` continues a stopped run; `python gpt.py --check` runs the gradient checks.
4. Talk to it: `python chat.py models/mine.npz`

Notes for small datasets: each step reads 16,384 tokens, so 15,000 steps is about 246 million tokens. A chat log much smaller than that will be repeated many times and the model will overfit; watch the gap between train and validation loss in the viewer and lower `--steps`. The learning rate (2e-3) was tuned only for TinyStories, so try a few values. Do not commit chat logs: `data/` is in `.gitignore`.

## Known limits

- Single runs, one seed for most comparisons; the model is small (12M parameters) and writes simple children's stories, not general text.
- The generation cache (`step 8`) is not built yet, so generation recomputes the whole window for each token.
- Trained weights are not in the repository (too large).
