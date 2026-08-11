"""
Eğitilmiş modelden metin üret.

    python sample.py --prompt "Türkiye" --tokens 500
    python sample.py --temperature 0.6 --top-k 40   # daha temkinli
    python sample.py --temperature 1.2              # daha savruk
"""

import argparse
import os
import pickle
import sys

import torch

from model import GPT, GPTConfig

sys.stdout.reconfigure(encoding="utf-8")

ROOT = os.path.dirname(os.path.abspath(__file__))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--prompt", default="\n")
    ap.add_argument("--tokens", type=int, default=500)
    ap.add_argument("--temperature", type=float, default=0.8)
    ap.add_argument("--top-k", type=int, default=None)
    ap.add_argument("--samples", type=int, default=1)
    ap.add_argument("--ckpt", default=os.path.join(ROOT, "ckpt.pt"))
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"

    with open(os.path.join(ROOT, "data", "meta.pkl"), "rb") as f:
        meta = pickle.load(f)
    stoi, itos = meta["stoi"], meta["itos"]

    ckpt = torch.load(args.ckpt, map_location=device, weights_only=False)
    model = GPT(GPTConfig(**ckpt["config"])).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()
    print(f"ckpt: iter {ckpt['iter']}, val loss {ckpt['val_loss']:.4f}\n")

    # Sözlükte olmayan karakterleri sessizce atıyoruz
    ids = [stoi[ch] for ch in args.prompt if ch in stoi]
    if not ids:
        ids = [stoi.get("\n", 0)]
    idx = torch.tensor([ids], dtype=torch.long, device=device)

    for i in range(args.samples):
        out = model.generate(idx, args.tokens, args.temperature, args.top_k)
        text = "".join(itos[int(t)] for t in out[0])
        print(f"--- örnek {i + 1} " + "-" * 50)
        print(text)
        print()


if __name__ == "__main__":
    main()
