%%writefile model.py
import math, torch, torch.nn as nn, torch.nn.functional as F

# ------------------------------
# Positional Encoding (sinusoidal)
# ------------------------------
class SinusoidalPositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=10000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term[:pe[:,1::2].shape[1]])
        self.register_buffer('pe', pe.unsqueeze(0))
    def forward(self, x):
        T = x.size(1)
        return self.pe[:, :T, :]

# ------------------------------
# Norm (LayerNorm or RMSNorm)
# ------------------------------
class RMSNorm(nn.Module):
    def __init__(self, d, eps=1e-8):
        super().__init__()
        self.scale = nn.Parameter(torch.ones(d))
        self.eps = eps
    def forward(self, x):
        norm = x.pow(2).mean(dim=-1, keepdim=True).add(self.eps).rsqrt()
        return self.scale * x * norm

def Norm(d, use_rmsnorm=False):
    return RMSNorm(d) if use_rmsnorm else nn.LayerNorm(d)

# ------------------------------
# Multi-Head Self-Attention with causal mask
# ------------------------------
class MultiHeadSelfAttention(nn.Module):
    def __init__(self, d_model, n_heads, dropout=0.0):
        super().__init__()
        assert d_model % n_heads == 0
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.qkv = nn.Linear(d_model, 3 * d_model, bias=False)
        self.proj = nn.Linear(d_model, d_model, bias=False)
        self.attn_drop = nn.Dropout(dropout)
        self.resid_drop = nn.Dropout(dropout)

    def forward(self, x):
        B, T, C = x.shape
        qkv = self.qkv(x)
        q, k, v = qkv.split(C, dim=-1)
        q = q.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        k = k.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        att = (q @ k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        mask = torch.triu(torch.ones(T, T, device=att.device), diagonal=1).bool()
        att = att.masked_fill(mask, float('-inf'))
        att = F.softmax(att, dim=-1)
        att = self.attn_drop(att)
        y = att @ v
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        return self.resid_drop(self.proj(y))

# ------------------------------
# Feed-Forward MLP
# ------------------------------
class FeedForward(nn.Module):
    def __init__(self, d_model, mult=4, dropout=0.0):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_model, mult * d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mult * d_model, d_model),
            nn.Dropout(dropout),
        )
    def forward(self, x):
        return self.net(x)

# ------------------------------
# Transformer Block
# ------------------------------
class Block(nn.Module):
    def __init__(self, d_model, n_heads, dropout=0.0, use_rmsnorm=False):
        super().__init__()
        self.norm1 = Norm(d_model, use_rmsnorm)
        self.attn  = MultiHeadSelfAttention(d_model, n_heads, dropout)
        self.norm2 = Norm(d_model, use_rmsnorm)
        self.ff    = FeedForward(d_model, dropout=dropout)
    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.ff(self.norm2(x))
        return x

# ------------------------------
# Transformer Language Model
# ------------------------------
class TinyTransformer(nn.Module):
    def __init__(self, vocab_size, d_model, n_layers, n_heads, block_size, dropout=0.0, use_rmsnorm=False):
        super().__init__()
        self.block_size = block_size
        self.tok_emb = nn.Embedding(vocab_size, d_model)
        self.pos_enc = SinusoidalPositionalEncoding(d_model, max_len=block_size+1024)
        self.drop    = nn.Dropout(dropout)
        self.blocks  = nn.ModuleList([Block(d_model, n_heads, dropout, use_rmsnorm) for _ in range(n_layers)])
        self.norm_f  = Norm(d_model, use_rmsnorm)
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)
        self.lm_head.weight = self.tok_emb.weight  # weight tying

    def forward(self, idx, targets=None):
        B, T = idx.shape
        assert T <= self.block_size
        x = self.tok_emb(idx)
        x = x + self.pos_enc(x)[:, :T, :]
        x = self.drop(x)
        for blk in self.blocks:
            x = blk(x)
        x = self.norm_f(x)
        logits = self.lm_head(x)
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
        return logits, loss

    @torch.no_grad()
    def generate(self, idx, max_new_tokens=200, temperature=1.0, top_k=None):
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -self.block_size:]
            logits, _ = self.forward(idx_cond)
            logits = logits[:, -1, :] / temperature
            if top_k is not None:
                v, _ = torch.topk(logits, top_k)
                logits[logits < v[:, [-1]]] = -float('Inf')
            probs = F.softmax(logits, dim=-1)
            next_tok = torch.multinomial(probs, num_samples=1)
            idx = torch.cat([idx, next_tok], dim=1)
        return idx
