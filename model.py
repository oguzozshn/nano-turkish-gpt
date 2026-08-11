"""
Karakter seviyesinde GPT — sıfırdan.

Tek bağımlılık: torch. `transformers` yok, hazır bir attention katmanı yok.
Amaç, transformer'ın içindeki her parçayı görünür tutmak.
"""

import math
from dataclasses import dataclass

import torch
import torch.nn as nn
from torch.nn import functional as F


@dataclass
class GPTConfig:
    vocab_size: int = 128
    block_size: int = 256  # modelin geriye bakabildiği karakter sayısı (bağlam)
    n_layer: int = 6
    n_head: int = 6
    n_embd: int = 384  # her kafaya 384 / 6 = 64 boyut düşer
    dropout: float = 0.2
    bias: bool = False
    # False: attention elle hesaplanır (okunur, öğretici)
    # True : PyTorch'un füzyonlu çekirdeği kullanılır (~2x hızlı, aynı matematik)
    use_sdpa: bool = False


class CausalSelfAttention(nn.Module):
    """Çok kafalı, nedensel (causal) self-attention.

    Nedensel: t. pozisyondaki karakter yalnızca <= t'yi görebilir. Geleceği
    görmesini engellemezsek model bir sonraki karakteri "kopya çekerek" bilir
    ve hiçbir şey öğrenmez.
    """

    def __init__(self, cfg: GPTConfig):
        super().__init__()
        assert cfg.n_embd % cfg.n_head == 0, "n_embd, n_head'e tam bölünmeli"

        self.n_head = cfg.n_head
        self.n_embd = cfg.n_embd
        self.head_dim = cfg.n_embd // cfg.n_head
        self.dropout_p = cfg.dropout
        self.use_sdpa = cfg.use_sdpa

        # Q, K, V'yi tek bir matris çarpımında üretip sonra üçe bölüyoruz.
        # Üç ayrı Linear ile aynı şey, ama bu hali GPU'da daha verimli.
        self.c_attn = nn.Linear(cfg.n_embd, 3 * cfg.n_embd, bias=cfg.bias)
        self.c_proj = nn.Linear(cfg.n_embd, cfg.n_embd, bias=cfg.bias)

        self.attn_dropout = nn.Dropout(cfg.dropout)
        self.resid_dropout = nn.Dropout(cfg.dropout)

        # Alt üçgen maske. Öğrenilen bir parametre değil, o yüzden buffer.
        self.register_buffer(
            "mask",
            torch.tril(torch.ones(cfg.block_size, cfg.block_size)).view(
                1, 1, cfg.block_size, cfg.block_size
            ),
            persistent=False,
        )

    def forward(self, x):
        B, T, C = x.shape  # batch, zaman (karakter sayısı), kanal (n_embd)

        q, k, v = self.c_attn(x).split(self.n_embd, dim=2)

        # (B, T, C) -> (B, n_head, T, head_dim)
        # Kafaları ayrı bir eksene alıyoruz ki her kafa bağımsız çalışsın.
        q = q.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        k = k.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.n_head, self.head_dim).transpose(1, 2)

        if self.use_sdpa:
            y = F.scaled_dot_product_attention(
                q,
                k,
                v,
                dropout_p=self.dropout_p if self.training else 0.0,
                is_causal=True,
            )
        else:
            # --- attention, elle ---
            # 1) Her sorgu (query) her anahtarla (key) ne kadar ilgili? (B, nh, T, T)
            #    sqrt(head_dim)'e bölmek: boyut büyüdükçe skorlar da büyür,
            #    bölmezsek softmax doyar ve gradyan ölür.
            att = (q @ k.transpose(-2, -1)) / math.sqrt(self.head_dim)

            # 2) Geleceği kapat. -inf, softmax'tan sonra tam 0 olur.
            att = att.masked_fill(self.mask[:, :, :T, :T] == 0, float("-inf"))

            # 3) Skorları olasılığa çevir: her satır toplamı 1
            att = F.softmax(att, dim=-1)
            att = self.attn_dropout(att)

            # 4) Değerlerin (value) ağırlıklı ortalaması -> (B, nh, T, head_dim)
            y = att @ v

        # Kafaları geri birleştir: (B, nh, T, hd) -> (B, T, C)
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        return self.resid_dropout(self.c_proj(y))


class MLP(nn.Module):
    """Pozisyon başına ileri besleme. Attention bilgiyi taşır, burası işler."""

    def __init__(self, cfg: GPTConfig):
        super().__init__()
        self.c_fc = nn.Linear(cfg.n_embd, 4 * cfg.n_embd, bias=cfg.bias)
        self.c_proj = nn.Linear(4 * cfg.n_embd, cfg.n_embd, bias=cfg.bias)
        self.dropout = nn.Dropout(cfg.dropout)

    def forward(self, x):
        return self.dropout(self.c_proj(F.gelu(self.c_fc(x))))


class Block(nn.Module):
    """Bir transformer bloğu: attention + MLP, ikisi de artık (residual) bağlantılı.

    LayerNorm alt katmandan *önce* (pre-LN). Derin modellerde post-LN'e göre
    çok daha stabil eğitiliyor.
    """

    def __init__(self, cfg: GPTConfig):
        super().__init__()
        self.ln_1 = nn.LayerNorm(cfg.n_embd, bias=cfg.bias)
        self.attn = CausalSelfAttention(cfg)
        self.ln_2 = nn.LayerNorm(cfg.n_embd, bias=cfg.bias)
        self.mlp = MLP(cfg)

    def forward(self, x):
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x


class GPT(nn.Module):
    def __init__(self, cfg: GPTConfig):
        super().__init__()
        self.cfg = cfg

        self.tok_emb = nn.Embedding(cfg.vocab_size, cfg.n_embd)  # hangi karakter
        self.pos_emb = nn.Embedding(cfg.block_size, cfg.n_embd)  # kaçıncı sırada
        self.drop = nn.Dropout(cfg.dropout)
        self.blocks = nn.ModuleList([Block(cfg) for _ in range(cfg.n_layer)])
        self.ln_f = nn.LayerNorm(cfg.n_embd, bias=cfg.bias)
        self.lm_head = nn.Linear(cfg.n_embd, cfg.vocab_size, bias=False)

        # Ağırlık paylaşımı: giriş embedding'i ile çıkış projeksiyonu aynı matris.
        # Parametre sayısını düşürür ve genelde sonucu iyileştirir.
        self.tok_emb.weight = self.lm_head.weight

        self.apply(self._init_weights)

        # Artık yoluna yazan projeksiyonları ölçekli başlat. Katman sayısı
        # arttıkça residual toplamı büyür; bu, varyansı sabit tutar.
        for name, p in self.named_parameters():
            if name.endswith("c_proj.weight"):
                nn.init.normal_(p, mean=0.0, std=0.02 / math.sqrt(2 * cfg.n_layer))

    @staticmethod
    def _init_weights(module):
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def num_params(self):
        # lm_head, tok_emb ile paylaşıldığı için iki kez sayılmıyor
        return sum(p.numel() for p in self.parameters())

    def forward(self, idx, targets=None):
        """idx: (B, T) karakter indeksleri. targets verilirse loss da döner."""
        B, T = idx.shape
        assert T <= self.cfg.block_size, f"{T} karakter, block_size {self.cfg.block_size}"

        pos = torch.arange(T, device=idx.device)
        x = self.drop(self.tok_emb(idx) + self.pos_emb(pos))

        for block in self.blocks:
            x = block(x)
        x = self.ln_f(x)

        logits = self.lm_head(x)  # (B, T, vocab_size)

        loss = None
        if targets is not None:
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)), targets.reshape(-1)
            )
        return logits, loss

    @torch.no_grad()
    def generate(self, idx, max_new_tokens, temperature=1.0, top_k=None):
        """Her adımda bir karakter üretip girdiye ekler.

        temperature: <1 daha temkinli, >1 daha savruk
        top_k: sadece en olası k karakter arasından seç
        """
        was_training = self.training
        self.eval()

        for _ in range(max_new_tokens):
            # Bağlam block_size'ı aşarsa baştan kırp
            idx_cond = idx[:, -self.cfg.block_size :]
            logits, _ = self(idx_cond)
            logits = logits[:, -1, :] / max(temperature, 1e-8)

            if top_k is not None:
                k = min(top_k, logits.size(-1))
                threshold = torch.topk(logits, k).values[:, [-1]]
                logits = logits.masked_fill(logits < threshold, float("-inf"))

            probs = F.softmax(logits, dim=-1)
            idx = torch.cat((idx, torch.multinomial(probs, num_samples=1)), dim=1)

        if was_training:
            self.train()
        return idx
