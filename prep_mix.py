"""Turns the two downloaded JSON-lines files into plain text with real newlines and the chat markers the tokenizer reserves:
  <|user|> question <|assistant|> answer <|endoftext|>
Usage: python prep_mix.py   (reads data/tokenizer_mix/*.jsonl, writes dolly.txt and code.txt next to them)"""
import json
from pathlib import Path

d = Path("data/tokenizer_mix")


def convert(src, dst, ask, answer):
    n = 0
    with open(d / src, encoding="utf-8") as f, open(d / dst, "w", encoding="utf-8", newline="\n") as out:
        for line in f:
            r = json.loads(line)
            out.write(f"<|user|>\n{ask(r)}\n<|assistant|>\n{answer(r)}<|endoftext|>\n")
            n += 1
    print(f"{dst}: {n:,} examples, {(d / dst).stat().st_size:,} bytes")


convert("databricks-dolly-15k.jsonl", "dolly.txt", lambda r: r["instruction"] + (("\n" + r["context"]) if r["context"] else ""), lambda r: r["response"])
convert("python-codes-25k.jsonl", "code.txt", lambda r: r["instruction"] + (("\n" + r["input"]) if r["input"] else ""), lambda r: r["output"])
