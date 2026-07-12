"""Supervised forecasting architectures for the matched baseline (pure PyTorch, no extra deps).

Primary: Temporal Fusion Transformer (Lim et al., Int. J. Forecasting 2021) — the same family
Kapral used, with native known-future-covariate handling and quantile outputs, so its I/O
matches TiRex-2 exactly. Registry `MODELS` lets us add PatchTST / iTransformer later.

Inputs (per window):
  past   [B, Lc, n_past]    MAP + past covariates
  future [B, Lc+H, n_fut]   known future covariates over context+horizon
Output:
  q      [B, H, n_quantiles]   forecast quantiles over the horizon
"""
from __future__ import annotations
import torch
import torch.nn as nn


class GLU(nn.Module):
    def __init__(self, d):
        super().__init__(); self.fc = nn.Linear(d, 2 * d)
    def forward(self, x):
        a, b = self.fc(x).chunk(2, dim=-1)
        return a * torch.sigmoid(b)


class GRN(nn.Module):
    """Gated Residual Network (TFT building block)."""
    def __init__(self, input_size, hidden, output_size, dropout=0.1):
        super().__init__()
        self.skip = nn.Linear(input_size, output_size) if input_size != output_size else nn.Identity()
        self.fc1 = nn.Linear(input_size, hidden)
        self.fc2 = nn.Linear(hidden, hidden)
        self.elu = nn.ELU()
        self.drop = nn.Dropout(dropout)
        self.glu = GLU(hidden)
        self.proj = nn.Linear(hidden, output_size)
        self.norm = nn.LayerNorm(output_size)
    def forward(self, x):
        h = self.fc2(self.elu(self.fc1(x)))
        h = self.proj(self.drop(self.glu(h)))
        return self.norm(h + self.skip(x))


class VariableSelection(nn.Module):
    """Per-timestep soft selection over input variables (TFT VSN)."""
    def __init__(self, n_vars, d, dropout=0.1):
        super().__init__()
        self.n_vars = n_vars; self.d = d
        self.embed = nn.ModuleList([nn.Linear(1, d) for _ in range(n_vars)])
        self.weight_grn = GRN(n_vars * d, d, n_vars, dropout)
        self.var_grn = nn.ModuleList([GRN(d, d, d, dropout) for _ in range(n_vars)])
    def forward(self, x):                       # x: [B, T, n_vars]
        embs = [self.embed[i](x[..., i:i + 1]) for i in range(self.n_vars)]  # each [B,T,d]
        flat = torch.cat(embs, dim=-1)                                       # [B,T,n_vars*d]
        w = torch.softmax(self.weight_grn(flat), dim=-1).unsqueeze(-1)       # [B,T,n_vars,1]
        proc = torch.stack([self.var_grn[i](embs[i]) for i in range(self.n_vars)], dim=2)  # [B,T,n_vars,d]
        return (w * proc).sum(dim=2)                                         # [B,T,d]


class TFT(nn.Module):
    def __init__(self, n_past, n_fut, horizon, d=64, n_heads=4, dropout=0.1, n_quantiles=9):
        super().__init__()
        self.H = horizon
        self.vsn_past = VariableSelection(n_past, d, dropout)
        self.vsn_fut = VariableSelection(n_fut, d, dropout)
        self.enc_lstm = nn.LSTM(d, d, batch_first=True)
        self.dec_lstm = nn.LSTM(d, d, batch_first=True)
        self.gate = GLU(d); self.norm1 = nn.LayerNorm(d)
        self.attn = nn.MultiheadAttention(d, n_heads, dropout=dropout, batch_first=True)
        self.norm2 = nn.LayerNorm(d)
        self.pos_grn = GRN(d, d, d, dropout)
        self.head = nn.Linear(d, n_quantiles)
    def forward(self, past, future):            # past [B,Lc,n_past], future [B,Lc+H,n_fut]
        B, Lc, _ = past.shape
        fp = self.vsn_past(past)                                    # [B,Lc,d]
        ff = self.vsn_fut(future)                                   # [B,Lc+H,d]
        enc_in, dec_in = ff[:, :Lc], ff[:, Lc:]                     # split future emb at the origin
        enc_feat = fp + enc_in                                     # combine past+known over context
        enc_out, (h, c) = self.enc_lstm(enc_feat)
        dec_out, _ = self.dec_lstm(dec_in, (h, c))                 # decoder over the horizon
        temporal = torch.cat([enc_out, dec_out], dim=1)            # [B,Lc+H,d]
        vsn_all = torch.cat([enc_feat, dec_in], dim=1)             # skip features
        x = self.norm1(self.gate(temporal) + vsn_all)             # gated skip
        T = x.size(1)
        causal = torch.triu(torch.ones(T, T, device=x.device, dtype=torch.bool), diagonal=1)
        a, _ = self.attn(x, x, x, attn_mask=causal, need_weights=False)
        x = self.norm2(a + x)
        dec = self.pos_grn(x[:, Lc:])                              # horizon positions [B,H,d]
        return self.head(dec)                                     # [B,H,Q]


MODELS = {"tft": TFT}


def build_model(name, n_past, n_fut, horizon, **kw):
    if name not in MODELS:
        raise ValueError(f"unknown model '{name}'; have {list(MODELS)}")
    return MODELS[name](n_past=n_past, n_fut=n_fut, horizon=horizon, **kw)
