# turkish-gpt

Karakter seviyesinde bir GPT — sıfırdan. `transformers` yok, hazır attention yok.
Amaç bir ürün çıkarmak değil, transformer'ın içindeki her parçayı bir kez elle yazmak.

## Kurulum

```
.venv\Scripts\activate
```

PyTorch zaten kurulu (CUDA 12.8 derlemesi — RTX 50 serisi için gerekli olan bu).

## Kullanım

**1. Veriyi hazırla**

```
python prepare.py --hf --limit-mb 40
```

Türkçe Wikipedia'dan ~40 MB metin indirir, karakter sözlüğünü çıkarır,
`data/train.bin` ve `data/val.bin` üretir. Kendi metnin varsa:

```
python prepare.py --input C:\yol\metin.txt
```

**2. Eğit**

```
python train.py
```

~10M parametre, 5000 adım. Checkpoint doğrulama loss'u her iyileştiğinde
`ckpt.pt`'ye yazılır.

VRAM boşsa `--batch-size 64` ile hızlandır. Attention'ın hızlı halini denemek
için `--use-sdpa` (matematik aynı, sadece füzyonlu çekirdek).

**3. Üret**

```
python sample.py --prompt "Türkiye" --tokens 500
python sample.py --temperature 0.6 --top-k 40
```

## Dosyalar

| Dosya | Ne var içinde |
|---|---|
| `model.py` | Attention, MLP, blok, GPT. Projenin özü burası. |
| `prepare.py` | Metin → karakter sözlüğü → uint16 binary |
| `train.py` | Eğitim döngüsü, LR planı, doğrulama, checkpoint |
| `sample.py` | Üretim (temperature, top-k) |

## Loss ne anlama geliyor

Karakter seviyesinde cross-entropy. Rastgele tahmin, `ln(vocab_size)` civarı
verir — bizim sözlüğümüz 149 karakter, yani başlangıç **~5.0**.

- **~2.5** → harf dağılımını öğrendi, kelime yok
- **~1.8** → Türkçe'ye benzeyen uydurma kelimeler, hece yapısı doğru
- **~1.4** → gerçek kelimeler, bozuk dilbilgisi
- **~1.2 ve altı** → yer yer düzgün cümleler

Anlamlı metin beklemiyoruz; 10M parametreli bir model 40 MB veriyle dünyayı
öğrenemez. Beklediğimiz şey, **Türkçe'nin biçimini** öğrenmesi: ünlü uyumu,
ekler, kelime uzunlukları, noktalama.

## Sonraki adım

Bu modelin kodu Adım 2'de (diakritik geri koyma) yeniden kullanılacak:
aynı bloklar, causal maske kaldırılmış ve üstüne karakter başına sınıflandırma
kafası eklenmiş hali.
