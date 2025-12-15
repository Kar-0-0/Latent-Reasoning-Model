import torch 
import torch.nn as nn
import torch.nn.functional as F

class LatentEncoder(nn.Module):
    def __init__(self, vocab_size, latent_dim):
        super().__init__()
        self.tok_embs = nn.Embedding(vocab_size, latent_dim)
        self.l1 = nn.Linear(latent_dim, latent_dim)
    
    def forward(self, x):
        embs = self.tok_embs(x)
        z = self.l1(embs)

        return z
    

class LatentDynamics(nn.Module):
    def __init__(self, latent_dim):
        super().__init__()
        self.gate = nn.Linear(latent_dim*2, latent_dim)
        self.sigmoid = nn.Sigmoid()
        self.alpha = nn.Parameter(torch.full((latent_dim,), fill_value=0.1))
        self.tanh = nn.Tanh()
    
    def forward(self, h_t, z):
        B, T, latent_dim = z.shape
        h = torch.zeros((B, T, latent_dim), device=z.device)
        for t in range(T):
            z_t = z[:, t, :] # (B, latent_dim)
            gate_in = torch.cat([h_t, z_t], dim=-1) # (B, latent_dim*2)
            gate = self.gate(gate_in) # (B, latent_dim)
            gate = self.sigmoid(gate)
            leak = 1 - self.alpha
            h_t = self.tanh((h_t * leak.unsqueeze(0)) + (gate * z_t))
            h[:, t, :] = h_t

        return h


class SelfAttention(nn.Module):
    def __init__(self, attn_dim, latent_dim, n_heads):
        super().__init__()
        self.attn_proj = nn.Linear(latent_dim, attn_dim)
        self.qkv = nn.Linear(attn_dim, attn_dim*3)
        self.proj = nn.Linear(attn_dim, attn_dim)
        self.n_heads = n_heads
        self.head_size = attn_dim // n_heads
    
    def forward(self, x):
        x = self.attn_proj(x) # (B, T, C)
        B, T, C = x.shape

        qkv = self.qkv(x) # (B, T, C*3)
        q, k, v = qkv.split(C, dim=2) # (B, T, C)

        q = q.view(B, T, self.n_heads, self.head_size).transpose(1, 2) # (B, nh, T, hs)
        k = k.view(B, T, self.n_heads, self.head_size).transpose(1, 2) # (B, nh, T, hs)
        v = v.view(B, T, self.n_heads, self.head_size).transpose(1, 2) # (B, nh, T, hs)

        attn_scores = (q @ k.transpose(-2, -1)) * 1.0/self.head_size**0.5 # (B, nh, T, T)
        attn_scores = F.softmax(attn_scores, dim=-1)

        out = attn_scores @ v # (B, nh, T, hs)
        out = out.transpose(1, 2).contiguous().view(B, T, C)
        out = self.proj(out)

        return out
    

class FeedForward(nn.Module):
    def __init__(self, attn_dim, dropout):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(attn_dim, attn_dim*4),
            nn.GELU(),
            nn.Linear(4*attn_dim, attn_dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        return self.net(x)


class Block(nn.Module):
    def __init__(
            self,
            latent_dim,
            attn_dim,
            n_heads,
            dropout
    ):
        super().__init__()
        self.ln1 = nn.LayerNorm(latent_dim)
        self.sa = SelfAttention(
            attn_dim,
            latent_dim,
            n_heads
        )

        self.ln2 = nn.LayerNorm(attn_dim)
        self.g2 = nn.GELU()
        self.ffwd = FeedForward(attn_dim, dropout)
    
    def forward(self, x):
        h = x + self.sa(self.ln1(x))
        h = h + self.ffwd(self.g2(self.ln2(h)))

        return h


class LatentDecoder(nn.Module):
    def __init__(
            self,
            n_layers,
            latent_dim,
            attn_dim,
            input_length,
            n_heads,
            vocab_size,
            dropout
    ):
        super().__init__()
        self.blocks = nn.Sequential(*[Block(latent_dim, attn_dim, n_heads, dropout) for _ in range(n_layers)])
        self.lm_head = nn.Linear(attn_dim, vocab_size)
    
    def forward(self, x):
        x = self.blocks(x)
        logits = self.lm_head(x)

        return logits


class LatentReasoningModel(nn.Module):
    def __init__(
      self,
      n_layers, 
      latent_dim,
      attn_dim,
      input_length,
      n_heads,
      vocab_size,
      dropout=0.2      
    ):
        super().__init__()
        self.encoder = LatentEncoder(vocab_size, latent_dim)
        self.latent_update = LatentDynamics(latent_dim)
        self.decoder = LatentDecoder(
            n_layers,
            latent_dim, 
            attn_dim,
            input_length,
            n_heads,
            vocab_size,
            dropout
        )

    def forward(self, x, return_latents=True):
        B, _ = x.shape
        z = self.encoder(x) # (B, T, latent_dim)
        _, _, latent_dim = z.shape
        h_0 = torch.zeros(B, latent_dim, device=z.device)
        h = self.latent_update(h_0, z) # (B, T, latent_dim)
        logits = self.decoder(h)
        if return_latents:
            return logits, h

        return logits
