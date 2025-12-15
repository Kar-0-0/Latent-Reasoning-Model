import torch 
import torch.nn as nn
import torch.nn.functional as F

class LatentEncoder(nn.Module):
    def __init__(self, vocab_size, n_emb):
        super().__init__()
        self.tok_embs = nn.Embedding(vocab_size, n_emb)
        self.l1 = nn.Linear(n_emb, n_emb)
    
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


