"""
Veriyi hazırla: ham metin -> karakter sözlüğü -> train.bin / val.bin

Kullanım:
    python prepare.py --hf --limit-mb 40          # Türkçe Wikipedia'dan indir
    python prepare.py --input C:\\yol\\metin.txt  # kendi metin dosyandan

Çıktı (data/ içine):
    train.bin, val.bin  -> uint16 karakter indeksleri
    meta.pkl            -> stoi / itos sözlükleri
"""

import argparse
import os
import pickle
import sys
from collections import Counter

import numpy as np

sys.stdout.reconfigure(encoding="utf-8")

DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")


def load_from_file(path):
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        return f.read()


def load_from_hf(limit_mb, config):
    """Türkçe Wikipedia'yı akış halinde indir, limit_mb dolunca dur."""
    try:
        from datasets import load_dataset
    except ImportError:
        sys.exit(
            "`datasets` kurulu değil. Ya kur:\n"
            "    pip install datasets\n"
            "ya da --input ile kendi metin dosyanı ver."
        )

    print(f"Türkçe Wikipedia indiriliyor (config={config}, hedef ~{limit_mb} MB)...")
    ds = load_dataset("wikimedia/wikipedia", config, split="train", streaming=True)

    budget = limit_mb * 1024 * 1024
    parts, total, n = [], 0, 0
    for row in ds:
        text = row["text"]
        parts.append(text)
        total += len(text.encode("utf-8"))
        n += 1
        if n % 5000 == 0:
            print(f"  {n} makale, {total / 1024 / 1024:.1f} MB")
        if total >= budget:
            break

    print(f"Bitti: {n} makale, {total / 1024 / 1024:.1f} MB")
    return "\n\n".join(parts)


def build_vocab(text, min_count):
    """Sözlüğü kur. Nadir karakterleri (tek tük Çince, matematik sembolü vs.)
    tamamen atıyoruz — sözlüğü şişirmelerinin bir faydası yok."""
    counts = Counter(text)
    keep = sorted(ch for ch, c in counts.items() if c >= min_count)

    dropped = len(counts) - len(keep)
    dropped_chars = sum(c for ch, c in counts.items() if c < min_count)
    print(
        f"Sözlük: {len(keep)} karakter tutuldu, {dropped} tür atıldı "
        f"({dropped_chars} karakter, toplamın %{100 * dropped_chars / len(text):.3f}'ü)"
    )

    stoi = {ch: i for i, ch in enumerate(keep)}
    itos = {i: ch for ch, i in stoi.items()}
    return stoi, itos


def main():
    ap = argparse.ArgumentParser()
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--input", help="UTF-8 metin dosyası")
    src.add_argument("--hf", action="store_true", help="Türkçe Wikipedia'dan indir")
    ap.add_argument("--hf-config", default="20231101.tr", help="wikimedia/wikipedia config adı")
    ap.add_argument("--limit-mb", type=int, default=40, help="--hf için indirme sınırı")
    ap.add_argument("--min-count", type=int, default=200, help="bir karakteri tutmak için asgari sıklık")
    ap.add_argument("--val-frac", type=float, default=0.005, help="doğrulama için ayrılan oran")
    args = ap.parse_args()

    text = load_from_file(args.input) if args.input else load_from_hf(args.limit_mb, args.hf_config)
    print(f"Ham metin: {len(text):,} karakter")

    stoi, itos = build_vocab(text, args.min_count)

    print("Kodlanıyor... (birkaç on saniye sürebilir)")
    ids = np.fromiter(
        (stoi[ch] for ch in text if ch in stoi), dtype=np.uint16, count=-1
    )
    print(f"Kodlandı: {len(ids):,} token")

    # Doğrulama kümesini sondan alıyoruz. Rastgele karıştırmıyoruz:
    # bitişik metinde eğitim ve doğrulama örtüşürse loss yalan söyler.
    n_val = max(1, int(len(ids) * args.val_frac))
    train_ids, val_ids = ids[:-n_val], ids[-n_val:]

    os.makedirs(DATA_DIR, exist_ok=True)
    train_ids.tofile(os.path.join(DATA_DIR, "train.bin"))
    val_ids.tofile(os.path.join(DATA_DIR, "val.bin"))
    with open(os.path.join(DATA_DIR, "meta.pkl"), "wb") as f:
        pickle.dump({"vocab_size": len(stoi), "stoi": stoi, "itos": itos}, f)

    print(f"\ntrain.bin : {len(train_ids):,} token")
    print(f"val.bin   : {len(val_ids):,} token")
    print(f"vocab     : {len(stoi)}")
    print(f"\nÖrnek sözlük: {''.join(sorted(stoi)[:80])}")


if __name__ == "__main__":
    main()
