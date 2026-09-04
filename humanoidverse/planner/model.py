"""Motion autoencoder for P1: Full window (50x43 normalized) -> c_r (R^32) -> Full window.

Encoder: input proj + learned positional embedding, 4-layer TransformerEncoder,
mean-pool over time -> latent head.
Decoder: 50 learned queries cross-attending to the latent token (memory len 1),
4-layer TransformerDecoder, per-frame heads:
  head_motion -> 43D window (same representation as input)
  head_contact -> 2D per-frame contact logits (auxiliary, not part of the
  reconstructed state — contact is a prediction target, never a command).

~5.5M params @ d_model=256, ff=512.
"""

import torch
import torch.nn as nn


class MotionAE(nn.Module):
    def __init__(self, in_dim=43, d_model=256, nhead=4, num_layers=4,
                 latent_dim=32, window=50, ff_dim=512, dropout=0.1):
        super().__init__()
        self.latent_dim = latent_dim
        self.in_proj = nn.Linear(in_dim, d_model)
        self.pos = nn.Parameter(torch.randn(window, d_model) * 0.02)
        enc = nn.TransformerEncoderLayer(
            d_model, nhead, ff_dim, dropout, batch_first=True, norm_first=True)
        self.encoder = nn.TransformerEncoder(enc, num_layers)
        self.to_latent = nn.Linear(d_model, latent_dim)

        self.query = nn.Parameter(torch.randn(window, d_model) * 0.02)
        self.latent_proj = nn.Linear(latent_dim, d_model)
        dec = nn.TransformerDecoderLayer(
            d_model, nhead, ff_dim, dropout, batch_first=True, norm_first=True)
        self.decoder = nn.TransformerDecoder(dec, num_layers)
        self.head_motion = nn.Linear(d_model, in_dim)
        self.head_contact = nn.Linear(d_model, 2)

    def encode(self, x):                    # x (B, W, in_dim), normalized
        h = self.encoder(self.in_proj(x) + self.pos)
        return self.to_latent(h.mean(dim=1))  # (B, latent)

    def decode(self, c):                    # c (B, latent)
        memory = self.latent_proj(c).unsqueeze(1)          # (B, 1, d)
        q = self.query.unsqueeze(0).expand(c.shape[0], -1, -1)
        h = self.decoder(q, memory)
        return self.head_motion(h), self.head_contact(h)   # (B,W,in), (B,W,2)

    def forward(self, x):
        c = self.encode(x)
        motion, contact = self.decode(c)
        return motion, contact, c

    def num_params(self):
        return sum(p.numel() for p in self.parameters())

class SparseEncoder(nn.Module):
    """P2 student: sparse history + sparse future (2 x W tokens) -> latent
    (superseded by CommandedEncoder for controllability; kept for loading
    the P2/P2.5/P2.6/P2.7 checkpoints)."""

    def __init__(self, in_dim=32, d_model=256, nhead=4, num_layers=4,
                 latent_dim=64, window=50, ff_dim=512, dropout=0.1):
        super().__init__()
        self.in_proj = nn.Linear(in_dim, d_model)
        self.pos = nn.Parameter(torch.randn(window, d_model) * 0.02)
        self.seg = nn.Parameter(torch.zeros(2, 1, d_model))  # 0=hist, 1=fut
        enc = nn.TransformerEncoderLayer(
            d_model, nhead, ff_dim, dropout, batch_first=True, norm_first=True)
        self.encoder = nn.TransformerEncoder(enc, num_layers)
        self.to_latent = nn.Linear(d_model, latent_dim)

    def forward(self, sp_hist, sp_fut, cmd=None):
        h = self.in_proj(torch.cat([sp_hist, sp_fut], dim=1))
        h = h + torch.cat([self.pos, self.pos], dim=0).unsqueeze(0)
        h = h + torch.cat([self.seg[0], self.seg[1]], dim=1)
        h = self.encoder(h)
        return self.to_latent(h.mean(dim=1))

    def num_params(self):
        return sum(p.numel() for p in self.parameters())


class CommandedEncoder(nn.Module):
    """P2.8: explicit command channel — state tokens + command tokens.

    Observation-derived sparse fields stay for base/feet/contact (proven
    informative by the P2.5 ablation), but the neck command enters as an
    EXPLICIT joint-space target (4 numbers/frame) with its own projection and
    segment embedding — bypassing the observation-representation problem
    (neck_yaw is statistically invisible in head pose: corr(rv_z, yaw)=-0.005).
    """

    def __init__(self, in_dim=32, cmd_dim=4, d_model=256, nhead=4,
                 num_layers=4, latent_dim=64, window=50, ff_dim=512,
                 dropout=0.1):
        super().__init__()
        self.in_proj = nn.Linear(in_dim, d_model)
        self.cmd_proj = nn.Linear(cmd_dim, d_model)
        self.pos = nn.Parameter(torch.randn(window, d_model) * 0.02)
        self.seg = nn.Parameter(torch.zeros(3, 1, d_model))  # hist/fut/cmd
        enc = nn.TransformerEncoderLayer(
            d_model, nhead, ff_dim, dropout, batch_first=True, norm_first=True)
        self.encoder = nn.TransformerEncoder(enc, num_layers)
        self.to_latent = nn.Linear(d_model, latent_dim)

    def forward(self, sp_hist, sp_fut, cmd=None):  # (B,W,in) (B,W,in) (B,W,4)
        parts = [self.in_proj(sp_hist), self.in_proj(sp_fut)]
        if cmd is not None:
            parts.append(self.cmd_proj(cmd))
        h = torch.cat(parts, dim=1)
        w = sp_hist.shape[1]
        pos = torch.cat([self.pos] * len(parts), dim=0).unsqueeze(0)
        seg = torch.cat([self.seg[i].repeat(1, w, 1) for i in range(len(parts))],
                        dim=1)
        h = self.encoder(h + pos + seg)
        return self.to_latent(h.mean(dim=1))

    def num_params(self):
        return sum(p.numel() for p in self.parameters())


