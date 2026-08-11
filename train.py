"""
Eğitim döngüsü.

    python train.py                      # varsayılan ayarlar
    python train.py --batch-size 64      # VRAM boşsa büyüt
    python train.py --use-sdpa           # hızlı attention (aynı matematik)

Checkpoint her doğrulama iyileştiğinde ckpt.pt olarak kaydedilir.
"""

import argparse
import math
import os
import pickle
import sys
import time

import numpy as np
import torch

from model import GPT, GPTConfig

sys.stdout.reconfigure(encoding="utf-8")

ROOT = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(ROOT, "data")


def get_batch(split, block_size, batch_size, device):
    # memmap'i her seferinde yeniden açıyoruz; uzun eğitimlerde bellek sızıntısını önlüyor
    path = os.path.join(DATA_DIR, "train.bin" if split == "train" else "val.bin")
    data = np.memmap(path, dtype=np.uint16, mode="r")

    ix = torch.randint(len(data) - block_size - 1, (batch_size,))
    x = torch.stack([torch.from_numpy(data[i : i + block_size].astype(np.int64)) for i in ix])
    # y, x'in bir karakter kaydırılmış hali: her pozisyonda hedef "bir sonraki karakter"
    y = torch.stack([torch.from_numpy(data[i + 1 : i + 1 + block_size].astype(np.int64)) for i in ix])

    if device.startswith("cuda"):
        return x.pin_memory().to(device, non_blocking=True), y.pin_memory().to(device, non_blocking=True)
    return x.to(device), y.to(device)


@torch.no_grad()
def estimate_loss(model, eval_iters, block_size, batch_size, device, ctx):
    """Tek batch'in loss'u çok gürültülü; birkaç batch'in ortalamasını alıyoruz."""
    model.eval()
    out = {}
    for split in ("train", "val"):
        losses = torch.zeros(eval_iters)
        for i in range(eval_iters):
            x, y = get_batch(split, block_size, batch_size, device)
            with ctx:
                _, loss = model(x, y)
            losses[i] = loss.item()
        out[split] = losses.mean().item()
    model.train()
    return out


def get_lr(it, warmup, max_iters, lr, min_lr):
    """Isınma + kosinüs sönümü. Isınma olmadan ilk adımlar modeli bozabiliyor."""
    if it < warmup:
        return lr * (it + 1) / (warmup + 1)
    if it >= max_iters:
        return min_lr
    ratio = (it - warmup) / max(1, max_iters - warmup)
    return min_lr + 0.5 * (1.0 + math.cos(math.pi * ratio)) * (lr - min_lr)


def make_optimizer(model, weight_decay, lr, betas):
    # 2 boyutlu tensörler (matrisler) weight decay alır; bias ve LayerNorm almaz.
    decay, no_decay = [], []
    for p in model.parameters():
        if p.requires_grad:
            (decay if p.dim() >= 2 else no_decay).append(p)
    print(f"weight decay uygulanan: {sum(p.numel() for p in decay):,} parametre")
    print(f"uygulanmayan          : {sum(p.numel() for p in no_decay):,} parametre")
    return torch.optim.AdamW(
        [
            {"params": decay, "weight_decay": weight_decay},
            {"params": no_decay, "weight_decay": 0.0},
        ],
        lr=lr,
        betas=betas,
        fused=torch.cuda.is_available(),
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--block-size", type=int, default=256)
    ap.add_argument("--n-layer", type=int, default=6)
    ap.add_argument("--n-head", type=int, default=6)
    ap.add_argument("--n-embd", type=int, default=384)
    ap.add_argument("--dropout", type=float, default=0.2)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--min-lr", type=float, default=1e-4)
    ap.add_argument("--max-iters", type=int, default=5000)
    ap.add_argument("--warmup", type=int, default=200)
    ap.add_argument("--weight-decay", type=float, default=0.1)
    ap.add_argument("--grad-clip", type=float, default=1.0)
    ap.add_argument("--eval-interval", type=int, default=250)
    ap.add_argument("--eval-iters", type=int, default=50)
    ap.add_argument("--use-sdpa", action="store_true", help="elle yazılmış attention yerine füzyonlu çekirdek")
    ap.add_argument("--seed", type=int, default=1337)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    if device == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        use_bf16 = torch.cuda.is_bf16_supported()
        ctx = torch.autocast("cuda", dtype=torch.bfloat16 if use_bf16 else torch.float16)
        print(f"cihaz: {torch.cuda.get_device_name(0)} | {'bf16' if use_bf16 else 'fp16'}")
    else:
        ctx = torch.autocast("cpu", enabled=False)
        print("cihaz: CPU (yavaş olacak)")

    meta_path = os.path.join(DATA_DIR, "meta.pkl")
    if not os.path.exists(meta_path):
        sys.exit("data/meta.pkl yok. Önce prepare.py çalıştır.")
    with open(meta_path, "rb") as f:
        meta = pickle.load(f)

    cfg = GPTConfig(
        vocab_size=meta["vocab_size"],
        block_size=args.block_size,
        n_layer=args.n_layer,
        n_head=args.n_head,
        n_embd=args.n_embd,
        dropout=args.dropout,
        use_sdpa=args.use_sdpa,
    )
    model = GPT(cfg).to(device)
    print(f"parametre: {model.num_params() / 1e6:.2f}M | sözlük: {cfg.vocab_size}")

    optimizer = make_optimizer(model, args.weight_decay, args.lr, (0.9, 0.95))

    best_val = float("inf")
    t0 = time.time()
    t_win = t0

    for it in range(args.max_iters + 1):
        lr = get_lr(it, args.warmup, args.max_iters, args.lr, args.min_lr)
        for g in optimizer.param_groups:
            g["lr"] = lr

        if it % args.eval_interval == 0:
            losses = estimate_loss(model, args.eval_iters, args.block_size, args.batch_size, device, ctx)
            elapsed = time.time() - t0
            print(
                f"iter {it:5d} | train {losses['train']:.4f} | val {losses['val']:.4f} "
                f"| lr {lr:.2e} | {elapsed / 60:.1f} dk"
            )
            if losses["val"] < best_val:
                best_val = losses["val"]
                torch.save(
                    {
                        "model": model.state_dict(),
                        "config": cfg.__dict__,
                        "iter": it,
                        "val_loss": best_val,
                    },
                    os.path.join(ROOT, "ckpt.pt"),
                )

        if it == args.max_iters:
            break

        x, y = get_batch("train", args.block_size, args.batch_size, device)
        with ctx:
            _, loss = model(x, y)

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        if args.grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
        optimizer.step()

        if it % 100 == 0 and it > 0:
            dt = time.time() - t_win
            tok_s = 100 * args.batch_size * args.block_size / dt
            print(f"  iter {it:5d} | loss {loss.item():.4f} | {tok_s / 1000:.0f}k token/sn")
            t_win = time.time()

    print(f"\nBitti. En iyi val loss: {best_val:.4f} -> ckpt.pt")
    print(f"Toplam süre: {(time.time() - t0) / 60:.1f} dakika")
    print("\nÖrnek üretmek için:  python sample.py --prompt \"Türkiye\"")


if __name__ == "__main__":
    main()
