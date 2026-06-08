import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class TransformerBlock(nn.Module):
    def __init__(self, d_model=128, heads=8, ff_hidden=256):
        super().__init__()
        self.heads = heads
        self.head_dim = d_model // heads
        self.scale = 1.0 / math.sqrt(float(self.head_dim))
        
        self.norm1 = nn.LayerNorm(d_model, eps=1e-5)
        self.attn_in = nn.Linear(d_model, 3 * d_model)
        self.attn_out = nn.Linear(d_model, d_model)
        
        self.norm2 = nn.LayerNorm(d_model, eps=1e-5)
        self.ff1 = nn.Linear(d_model, ff_hidden)
        self.ff2 = nn.Linear(ff_hidden, d_model)
        
    def forward(self, x, mask=None):
        z = self.norm1(x)
        B, C, d_model = z.shape
        qkv = self.attn_in(z)
        q, k, v = torch.chunk(qkv, 3, dim=-1)
        
        q = q.view(B, C, self.heads, self.head_dim).transpose(1, 2)
        k = k.view(B, C, self.heads, self.head_dim).transpose(1, 2)
        v = v.view(B, C, self.heads, self.head_dim).transpose(1, 2)
        
        scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        if mask is not None:
            expanded_mask = mask.unsqueeze(1).unsqueeze(2)
            scores = scores.masked_fill(expanded_mask == 0, -1e9)
            
        probs = F.softmax(scores, dim=-1)
        context = torch.matmul(probs, v)
        context = context.transpose(1, 2).contiguous().view(B, C, d_model)
        attn = self.attn_out(context)
        
        x = x + attn
        z2 = self.norm2(x)
        hid = F.silu(self.ff1(z2))
        ff = self.ff2(hid)
        
        return x + ff

class AttentionRanker(nn.Module):
    def __init__(self, d_model=128, heads=8, layers=3, ff_hidden=256, score_hidden=64, max_candidates=24):
        super().__init__()
        self.d_model = d_model
        
        self.register_buffer("mean", torch.zeros(46))
        self.register_buffer("std", torch.ones(46))
        
        self.input_layer = nn.Linear(46, d_model)
        self.rank_embedding = nn.Embedding(max_candidates, d_model)
        
        self.blocks = nn.ModuleList([
            TransformerBlock(d_model, heads, ff_hidden)
            for _ in range(layers)
        ])
        
        self.norm = nn.LayerNorm(d_model, eps=1e-5)
        self.score1 = nn.Linear(d_model * 3, score_hidden)
        self.score2 = nn.Linear(score_hidden, 1)
        
    def forward(self, x, mask=None):
        B, C, _ = x.shape
        x = (x - self.mean.view(1, 1, -1)) / (self.std.view(1, 1, -1) + 1e-6)
        h = F.silu(self.input_layer(x))
        
        idx = torch.arange(C, device=x.device).unsqueeze(0).expand(B, -1)
        idx = torch.clamp(idx, max=self.rank_embedding.num_embeddings - 1)
        h = h + self.rank_embedding(idx)
        
        for block in self.blocks:
            h = block(h, mask)
            
        h = self.norm(h)
        
        if mask is not None:
            mask_float = mask.unsqueeze(-1).float()
            sum_h = torch.sum(h * mask_float, dim=1)
            num_valid = torch.sum(mask_float, dim=1).clamp(min=1.0)
            ctx = sum_h / num_valid
        else:
            ctx = torch.mean(h, dim=1)
            
        ctx_expanded = ctx.unsqueeze(1).expand(-1, C, -1)
        z = torch.cat([h, ctx_expanded, h * ctx_expanded], dim=-1)
        
        hidden = F.silu(self.score1(z))
        scores = self.score2(hidden).squeeze(-1)
        
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
            
        return scores
