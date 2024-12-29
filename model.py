from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def _norm(self, x):
        # reciprocal
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x):
        output = self._norm(x.float()).type_as(x)
        return output * self.weight

class FFNSwishGLU(nn.Module):
    def __init__(self, embed_dim, hidden_dim, bias=False, dropout=0.0):
        super().__init__()
        self.ln1 = nn.Linear(embed_dim, hidden_dim, bias=bias)
        self.ln2 = nn.Linear(hidden_dim, embed_dim, bias=bias)
        self.ln3 = nn.Linear(embed_dim, hidden_dim, bias=bias)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return self.dropout(
            self.ln2(
                F.silu(self.ln1(x)) * self.ln3(x)
            )
        )
    
class MultiHeadAttention(nn.Module):
    def __init__(self, embed_dim, num_heads=8, bias=False, dropout=0.0):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads

        self.dropout = dropout
        self.qkv = nn.Linear(embed_dim, 3 * embed_dim, bias=bias)
        self.proj = nn.Linear(embed_dim, embed_dim, bias=bias)

    def forward(self, x):
        # x is [batch_size, seq_len, embed_dim]
        batch_size, seq_len, embed_dim = x.size()
        q, k, v = self.qkv(x).split(self.embed_dim, dim=-1)
        # [batch_size, seq_len, embed_dim] -> [batch_size, num_heads, seq_len, ,head_dim]
        q = q.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)

        y = F.scaled_dot_product_attention(q, k, v, dropout_p=self.dropout)
        y = y.transpose(1, 2).reshape(batch_size, seq_len, embed_dim)

        return self.proj(y)

class Block(nn.Module):
    def __init__(self, embed_dim, num_heads, bias=False, dropout=0.0):
        super().__init__()
        self.norm1 = RMSNorm(embed_dim)
        self.attn = MultiHeadAttention(embed_dim, num_heads, bias, dropout)
        self.norm2 = RMSNorm(embed_dim)
        self.ffn = FFNSwishGLU(embed_dim, embed_dim*4, bias, dropout)

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.ffn(self.norm2(x))
        return x

# ---------------------------
@dataclass
class PPClassifierConfig:
    image_size: int = 224
    patch_size: int = 16
    num_classes: int = 6

    embed_dim: int = 768
    num_heads: int = 12
    num_layers: int = 6
    bias: bool = False
    dropout: float = 0.0
# ----------------------------

class PPClassifier(nn.Module):
    def __init__(self, config: PPClassifierConfig):
        super().__init__()
        self.config = config

        self.num_patches = (config.image_size // config.patch_size) ** 2 # default=196

        # more efficient to use a conv layer than flatten & linear
        self.patch_embedding = nn.Conv2d(3, config.embed_dim,
                                         kernel_size=config.patch_size,
                                         stride=config.patch_size)
        #learnable class token
        self.cls_token = nn.Parameter(torch.randn(1, 1, config.embed_dim))
        
        # learnable positional embedding
        self.pos_embedding = nn.Parameter(torch.randn(1, self.num_patches+1, config.embed_dim) * 0.02)

        self.blocks = nn.ModuleList([
            Block(config.embed_dim, config.num_heads, config.bias, config.dropout)
            for _ in range(config.num_layers)
        ])

        self.norm = RMSNorm(config.embed_dim)
        self.head = nn.Linear(config.embed_dim, config.num_classes, bias=config.bias)

    def forward(self, x):
        batch_size = x.size(0)
        # x is [batch_size, 3, image_size, image_size]
        # patch embedding -> [batch_size, embed_dim, num_patches, num_patches]
        x = self.patch_embedding(x)
        x = x.flatten(2, -1).transpose(1, 2)
        # cls token [1, 1, embed_dim] -> [batch_size, 1, embed_dim]
        cls_tokens = self.cls_token.expand(batch_size, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        x = x + self.pos_embedding

        for block in self.blocks:
            x = block(x)

        # x is [batch_size, num_patches+1, embed_dim]
        x = self.norm(x[:, 0])
        x = self.head(x)
        return x


