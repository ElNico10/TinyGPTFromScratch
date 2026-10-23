"""Byte-level BPE tokenizer (byte-pair encoding), written from scratch.
Start from the 256 byte values; repeatedly merge the most frequent adjacent pair into a new token; reserve a few special tokens.
Usage: python tok.py train OUT.json VOCAB FILE[:MB] ...     learn merges from the first MB megabytes of each file (default: the whole file)
       python tok.py encode TOKENIZER.json IN.txt OUT.npy   turn a text file into 16-bit token ids
       python tok.py test TOKENIZER.json                    round-trip and compression tests"""
import json
import re
import sys
import time
from collections import Counter, defaultdict
from itertools import chain
from pathlib import Path
import numpy as np

SPECIALS = ["<|endoftext|>", "<|user|>", "<|assistant|>"]
SPECIAL_BYTES = {s.encode() for s in SPECIALS}
SPECIAL_SPLIT = re.compile(b"(" + b"|".join(re.escape(s.encode()) for s in SPECIALS) + b")")
# a "word": a run of letters, of digits or of punctuation (each with an optional leading space), or whitespace; bytes >= 128 (non-English letters) count as letters
WORD = re.compile(rb" ?[A-Za-z\x80-\xff]+| ?[0-9]+| ?[^\sA-Za-z0-9\x80-\xff]+|\s+(?!\S)|\s+")


class Tokenizer:
    def __init__(self, merges):
        self.merges = [tuple(m) for m in merges]  # merge i creates token 256 + i from the pair of earlier tokens
        self.rank = {m: i for i, m in enumerate(self.merges)}
        self.pieces = [bytes([i]) for i in range(256)]
        for a, b in self.merges:
            self.pieces.append(self.pieces[a] + self.pieces[b])
        self.special = {s.encode(): len(self.pieces) + i for i, s in enumerate(SPECIALS)}
        self.pieces += [s.encode() for s in SPECIALS]
        self.size = len(self.pieces)
        self.lengths = np.array([len(p) for p in self.pieces])  # bytes each token stands for
        self.cache = {}

    @classmethod
    def load(cls, path):
        return cls(json.loads(Path(path).read_text())["merges"])

    def save(self, path):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        Path(path).write_text(json.dumps({"vocab": self.size, "specials": SPECIALS, "merges": self.merges}))

    def word_ids(self, w):
        if w not in self.cache:
            ids = list(w)
            while len(ids) > 1:
                best = min(zip(ids, ids[1:]), key=lambda p: self.rank.get(p, 1 << 30))  # the pair merged earliest in training
                if best not in self.rank:
                    break
                new, out, i = 256 + self.rank[best], [], 0
                while i < len(ids):
                    if i + 1 < len(ids) and (ids[i], ids[i + 1]) == best:
                        out.append(new)
                        i += 2
                    else:
                        out.append(ids[i])
                        i += 1
                ids = out
            self.cache[w] = ids
        return self.cache[w]

    def encode(self, text):
        data = text.encode("utf-8") if isinstance(text, str) else text
        out = []
        for part in SPECIAL_SPLIT.split(data):
            if part in self.special:
                out.append(self.special[part])
            else:
                out.extend(chain.from_iterable(self.word_ids(w) for w in WORD.findall(part)))
        return out

    def decode_bytes(self, ids):
        return b"".join(self.pieces[int(i)] for i in ids)

    def decode(self, ids):
        return self.decode_bytes(ids).decode("utf-8", "replace")


def train(files, vocab_size):
    counts = Counter()
    for path, mb in files:
        with open(path, "rb") as f:
            data = f.read(int(mb * 1e6)) if mb else f.read()
        data = data[:data.rfind(b"\n") + 1] if mb else data  # do not learn from a word cut in half at the end
        for start in range(0, len(data), 8_000_000):  # in pieces, to keep the list of words small
            for part in SPECIAL_SPLIT.split(data[start:start + 8_000_000]):
                if part not in SPECIAL_BYTES:  # the markers are not words to learn merges from
                    counts.update(WORD.findall(part))
        print(f"{path}: {len(data):,} bytes read", flush=True)
    words = [list(w) for w in counts]
    freq = [counts[w] for w in counts]
    print(f"{len(words):,} distinct words", flush=True)
    pair_count, where = defaultdict(int), defaultdict(set)
    for i, w in enumerate(words):
        for pair in zip(w, w[1:]):
            pair_count[pair] += freq[i]
            where[pair].add(i)
    merges, t0 = [], time.time()
    while len(merges) < vocab_size - 256 - len(SPECIALS) and pair_count:
        best = max(pair_count.items(), key=lambda kv: (kv[1], kv[0]))[0]  # ties go to the larger pair, so training is deterministic
        new = 256 + len(merges)
        merges.append(best)
        for i in list(where[best]):  # only the words containing the pair change
            w, f = words[i], freq[i]
            for pair in zip(w, w[1:]):
                pair_count[pair] -= f
                where[pair].discard(i)
                if pair_count[pair] <= 0:
                    del pair_count[pair]
                    where.pop(pair, None)
            out, j = [], 0
            while j < len(w):
                if j + 1 < len(w) and (w[j], w[j + 1]) == best:
                    out.append(new)
                    j += 2
                else:
                    out.append(w[j])
                    j += 1
            words[i] = out
            for pair in zip(out, out[1:]):
                pair_count[pair] += f
                where[pair].add(i)
        if len(merges) % 500 == 0:
            print(f"  {len(merges)} merges ({time.time() - t0:.0f}s)", flush=True)
    return Tokenizer(merges)


def encode_file(t, src, dst, chunk=32_000_000):
    # whole-file encoding in pieces; cutting only right after an <|endoftext|> marker makes the result identical to encoding in one go
    eot, out, buf, total = b"<|endoftext|>", [], b"", 0
    with open(src, "rb") as f:
        while True:
            more = f.read(chunk)
            buf += more
            cut = len(buf) if not more else buf.rfind(eot) + len(eot) if eot in buf else 0
            if cut:
                out.append(np.array(t.encode(buf[:cut]), dtype=np.uint16))
                total += cut
                buf = buf[cut:]
                print(f"  {total / 1e6:,.0f} MB encoded", end="\r", flush=True)
            if not more:
                break
    ids = np.concatenate(out)
    np.save(dst, ids)
    print(f"\n{src}: {total:,} bytes -> {len(ids):,} tokens ({total / len(ids):.2f} bytes per token), saved {dst}")


def self_test(t):
    rng = np.random.default_rng(0)
    cases = [b"", b" ", b"\n", b"hello world", b"  two  spaces ", b"\n\n    indented\n\tcode:\n        return x\n", "café über 中文 \U0001F415".encode(),
             b"a<|endoftext|>b<|user|>hi<|assistant|>yo<|endoftext|>", bytes(range(256)), rng.integers(0, 256, 5000, dtype=np.uint8).tobytes()]
    for c in cases:
        assert t.decode_bytes(t.encode(c)) == c, f"round trip failed for {c[:40]!r}"
    print(f"round-trip ok on {len(cases)} cases (empty, whitespace, indentation, unicode, special markers, all 256 bytes, 5000 random bytes)")
    assert t.encode("<|endoftext|>") == [t.special[b"<|endoftext|>"]] and t.size == 256 + len(t.merges) + len(SPECIALS)
    print(f"vocabulary {t.size}: 256 bytes + {len(t.merges)} merges + {len(SPECIALS)} special tokens")


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else ""
    if cmd == "train":
        spec = [((m[1], float(m[2])) if (m := re.fullmatch(r"(.+):(\d+(?:\.\d+)?)", a)) else (a, None)) for a in sys.argv[4:]]
        tk = train(spec, int(sys.argv[3]))
        tk.save(sys.argv[2])
        self_test(tk)
    elif cmd == "encode":
        encode_file(Tokenizer.load(sys.argv[2]), sys.argv[3], sys.argv[4])
    elif cmd == "test":
        self_test(Tokenizer.load(sys.argv[2]))
    else:
        print(__doc__)
