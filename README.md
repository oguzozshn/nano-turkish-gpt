# nano-turkish-gpt

Karakter seviyesinde bir GPT — sıfırdan. `transformers` yok, hazır attention katmanı yok.
Amaç bir ürün çıkarmak değil, transformer'ın içindeki her parçayı bir kez elle yazmak.

Attention `model.py` içinde açıkça hesaplanıyor: `q @ kᵀ / √d`, nedensel maske, softmax,
`att @ v`. `--use-sdpa` bayrağı PyTorch'un füzyonlu çekirdeğine geçiriyor — matematik aynı,
sadece daha hızlı.

## Sonuçlar

10.78M parametre, 38.4M token Türkçe Wikipedia, 5000 adım.

| | |
|---|---|
| Donanım | RTX 5070 Laptop (8 GB), bf16 |
| Süre | **11.3 dakika** |
| Hız | ~129k token/sn |
| En iyi val loss | **1.1804** (iter 4750) |
| VRAM | 1.94 GB (batch 32) / ~3.5 GB (batch 64) |

### Loss eğrisi

| iter | train | val | | iter | train | val |
|---:|---:|---:|---|---:|---:|---:|
| 0 | 4.9637 | 4.9666 | | 3000 | 1.2364 | 1.2442 |
| 500 | 1.8453 | 1.8440 | | 3500 | 1.2005 | 1.2289 |
| 1000 | 1.5192 | 1.5349 | | 4000 | 1.1824 | 1.2075 |
| 1500 | 1.3841 | 1.3884 | | 4500 | 1.1665 | 1.1842 |
| 2000 | 1.3089 | 1.3137 | | **4750** | 1.1636 | **1.1804** |
| 2500 | 1.2595 | 1.2745 | | 5000 | 1.1607 | 1.1834 |

Val loss 4750'de dibi görüp 5000'de hafifçe yukarı dönüyor — bu boyut ve bu veriyle model
doymuş durumda. Daha fazla adım değil, daha fazla parametre veya daha fazla veri gerekiyor.

Seed sabit (`--seed 1337`); aynı komut aynı loss eğrisini üretiyor.

## Eğitim boyunca ne değişti

Aynı prompt (`"Türkiye"`), üç farklı checkpoint. İlginç olan kalitenin artması değil,
**neyin ne zaman öğrenildiği.**

**iter 500 — val 1.84**

> Türkiye etkilerinde dildi. Ayrıcıç sahip takımında yakın anı alan ve bu çük roumlardan
> tarafında yer arasında kavaşmasında, "Vabi ve piyatistan beykerasından birliğinin ne
> kurummaktadır.

Kökler uydurma, ama **ekler doğru**. "beykerasından", "kavaşmasında", "kurummaktadır" —
kalın ünlülü köke kalın ek, ince ünlülüye ince ek. Ünlü uyumunu 500 adımda çıkarmış.

**iter 2500 — val 1.27**

> Türkiye'nin bölgelerine göre başka kalmıştır. Bu karar devrinde, en büyük kararlı bir
> hayatın 1330 km²'yi bulmuştur. Gerekçe, 12 bölgede, arttığı kararlı yakınlarda bulunan
> iki bölgede bölgedeki Fransız, 16'sını kazanan bu durum olarak kullanılmıştır.

~45 kelimede uydurma olan tek kelime: "Artkın". Ayrıca **kesme işareti kuralını** öğrenmiş:
"Türkiye'nin", "16'sını" — özel isme ve sayıya gelen ek ayrılıyor.

**iter 4750 — val 1.18** (aşağıdaki örnekler)

Kelimeler ve yazım oturmuş, cümle içi öbekler tutarlı. Eksik olan **anlam** — ve o bu
ölçekte gelmiyor.

## Örnek çıktılar

`--prompt "Türkiye"`, final checkpoint.

**temperature 0.6** — temkinli, ve kendini tekrar ediyor

> Türkiye Cumhuriyeti ve ABD'de yapılan günlerde ayrıca en büyük konuşma sistemi olan
> sistem sisteminin yayınlanması için kullanılmaktadır. Türkiye'de de en büyük değişik
> değişikliklerin de önemli bir kısmı olan Kanada, Kanada'daki çalışmalar arasında en
> küçük bir değişiklik değişikliğine dayanmaktadır.

Düşük sıcaklıkta model hep en olası karakteri seçtiği için döngüye giriyor:
"sistem sisteminin", "değişiklik değişikliğine".

**temperature 0.8** — dengeli

> Türkiye'de birçok kişi yayınlaması üzerine, kadın Kültür ve Tekniklerini işgal etmesine
> rağmen daha sonra bu kişiler ile teknik olduğu tahmin edilmektedir. İlk teknik olarak,
> bunların restorasyonu ve erkek olarak, kişilerin en az olduğu için büyümelerin
> ayrılmasından oluşan yeni ilkeler çalışmalar yapmaktadır.

**temperature 1.0** — çeşitli, ama savruk

> Türkiye Cumhuriyeti tarafından geri alınmaktadır. Engepli Amerika ve Yukarı Aydınlar,
> Talibanlı orduları ile birlikte, Harp; Avusturya Birliği'nin düşüncesine kadar sürmüştür.
> Bununla birlikte, genellikle transfer bunu tercüme etmek için hayatını ele almış olan'dan
> dolayı bir İsveç liderliği tarafından gittiğini bildiren bun

Özel isimler çıkmaya başlıyor (Avusturya Birliği, İsveç, Taliban) ama hatalar da artıyor —
"olan'dan" gibi yanlış kesme işaretleri.

## Kurulum

```
python -m venv .venv
.venv\Scripts\activate
pip install torch --index-url https://download.pytorch.org/whl/cu128
pip install numpy datasets
```

`cu128` derlemesi RTX 50 serisi (Blackwell, sm_120) için gerekli — daha eski CUDA
derlemeleri "no kernel image available" hatası verir.

## Kullanım

**1. Veriyi hazırla**

```
python prepare.py --hf --limit-mb 40
```

Türkçe Wikipedia'dan ~40 MB metin indirir, karakter sözlüğünü çıkarır (149 karakter),
`data/train.bin` ve `data/val.bin` üretir. Kendi metnin varsa:

```
python prepare.py --input C:\yol\metin.txt
```

**2. Eğit**

```
python train.py --batch-size 64
```

Checkpoint, doğrulama loss'u her iyileştiğinde `ckpt.pt`'ye yazılır.
Attention'ın hızlı halini denemek için `--use-sdpa`.

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
| `train.py` | Eğitim döngüsü, ısınmalı kosinüs LR planı, doğrulama, checkpoint |
| `sample.py` | Üretim (temperature, top-k) |

## Loss ne anlama geliyor

Karakter seviyesinde cross-entropy. Rastgele tahmin `ln(vocab_size)` civarı verir —
149 karakterlik sözlükte bu **~5.0**, ve eğitim tam oradan başlıyor.

- **~2.5** → harf dağılımını öğrendi, kelime yok
- **~1.8** → Türkçe'ye benzeyen uydurma kelimeler, ünlü uyumu doğru
- **~1.4** → gerçek kelimeler, bozuk dilbilgisi
- **~1.2** → doğru yazım, tutarlı öbekler, anlam yok

## Doğrulama

`model.py` üç şeye karşı test edildi:

- **Başlangıç loss'u = `ln(vocab)`** — eğitilmemiş model her karakteri eşit olası
  görmeli. 4.9637 ölçüldü, `ln(149) = 5.0039`. Bozuk init ve yanlış loss hesabını
  aynı anda eler.
- **Elle yazılan attention = PyTorch çekirdeği** — aynı girdide maksimum fark
  `3.58e-07`, yani float32 gürültüsü.
- **Nedensellik sızdırmıyor** — girdinin ikinci yarısı tamamen değiştirildiğinde ilk
  yarının çıktısındaki fark tam `0.00e+00`.

## Sonraki adım

Bu modelin blokları bir sonraki projede yeniden kullanılacak: **Türkçe diakritik geri
koyma** (`"cok guzel bir gun"` → `"çok güzel bir gün"`). Aynı transformer, nedensel maske
kaldırılmış ve üstüne karakter başına sınıflandırma kafası eklenmiş hali. Etiketli veri
bedava: herhangi bir Türkçe metnin şapkalarını sök, hedef zaten elinde.
