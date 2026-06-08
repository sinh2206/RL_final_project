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

class PPOActorCritic(nn.Module):
    def __init__(self, d_model=128, heads=8, layers=3, ff_hidden=256, score_hidden=64, max_candidates=25):
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
        
        # Actor head for ranking candidates
        self.actor_score1 = nn.Linear(d_model * 3, score_hidden)
        self.actor_score2 = nn.Linear(score_hidden, 1)
        
        # Critic head for estimating state/action-value
        self.critic_proj = nn.Linear(d_model, score_hidden)
        self.critic_value = nn.Linear(score_hidden, 1)
        
    def forward(self, x, mask=None):
        """
        Input:
            x: tensor of shape [B, C, 46] (batch of candidate features)
            mask: tensor of shape [B, C] (boolean mask)
        Returns:
            logits: [B, C] (candidate target choice logits)
            value: [B, 1] (value estimate for each source planet state)
        """
        B, C, _ = x.shape
        x_norm = (x - self.mean.view(1, 1, -1)) / (self.std.view(1, 1, -1) + 1e-6)
        h = F.silu(self.input_layer(x_norm))
        
        idx = torch.arange(C, device=x.device).unsqueeze(0).expand(B, -1)
        h = h + self.rank_embedding(idx)
        
        for block in self.blocks:
            h = block(h, mask)
            
        h = self.norm(h)
        
        # Global context pooling over candidates
        if mask is not None:
            mask_float = mask.unsqueeze(-1).float()
            sum_h = torch.sum(h * mask_float, dim=1)
            num_valid = torch.sum(mask_float, dim=1).clamp(min=1.0)
            ctx = sum_h / num_valid
        else:
            ctx = torch.mean(h, dim=1)
            
        # 1. Actor Logits
        ctx_expanded = ctx.unsqueeze(1).expand(-1, C, -1)
        z = torch.cat([h, ctx_expanded, h * ctx_expanded], dim=-1)
        
        actor_hidden = F.silu(self.actor_score1(z))
        logits = self.actor_score2(actor_hidden).squeeze(-1)
        
        if mask is not None:
            logits = logits.masked_fill(mask == 0, -1e9)
            
        # 2. Critic Values (for each decision point/source planet)
        critic_hidden = F.silu(self.critic_proj(ctx))
        values = self.critic_value(critic_hidden) # [B, 1]
        
        return logits, values
