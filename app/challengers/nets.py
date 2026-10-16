"""Arsitektur jaringan saraf challenger (PyTorch, CPU).

Semua model mengeluarkan KUANTIL log return terstandardisasi (pinball loss), dengan kepala kuantil MONOTON
(median + kenaikan softplus kumulatif) sehingga kuantil tidak pernah bersilangan.

- RNNQuantile (LSTM / GRU): sekuens L hari fitur → kuantil log return H hari.
- TemporalFusionTransformer: implementasi ringkas namun setia pada Lim et al. (2021, "Temporal Fusion Transformers for
  interpretable multi-horizon time series forecasting"):
    * input statis (sektor) → static covariate encoders (konteks VSN, enrichment, state awal LSTM)
    * variable selection networks (VSN) untuk input masa lalu (fitur teramati + hari) dan masa depan yang diketahui (hari)
    * encoder–decoder LSTM + gated skip (GLU, add & norm)
    * static enrichment, interpretable multi-head attention (value dibagi antar head, mask kausal)
    * position-wise GRN, kuantil untuk setiap horizon 1..H (target: log return kumulatif)
  Bobot VSN & attention dapat dibaca untuk interpretasi (`return_weights=True`).
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


# ============================================================================== komponen umum
class MonotoneQuantileHead(nn.Module):
    def __init__(self, d_in: int, n_quantiles: int, median_index: int):
        super().__init__()
        self.lin = nn.Linear(d_in, n_quantiles)
        self.m = median_index

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        raw = self.lin(h)
        mid = raw[..., self.m:self.m + 1]
        up = F.softplus(raw[..., self.m + 1:]).cumsum(-1)
        dn = F.softplus(raw[..., :self.m]).flip(-1).cumsum(-1).flip(-1)
        return torch.cat([mid - dn, mid, mid + up], dim=-1)


def pinball_loss(pred: torch.Tensor, target: torch.Tensor, levels: torch.Tensor) -> torch.Tensor:
    """pred (..., Q), target (...) → rata-rata pinball loss atas semua kuantil & elemen."""
    d = target.unsqueeze(-1) - pred
    return torch.maximum(levels * d, (levels - 1) * d).mean()


# ============================================================================== LSTM / GRU
class RNNQuantile(nn.Module):
    def __init__(self, n_features: int, kind: str = "lstm", hidden: int = 64, layers: int = 2, dropout: float = 0.1,
                 n_quantiles: int = 7, median_index: int = 3):
        super().__init__()
        self.kind = kind
        self.inp = nn.Linear(n_features, hidden)
        rnn_cls = nn.LSTM if kind == "lstm" else nn.GRU
        self.rnn = rnn_cls(hidden, hidden, num_layers=layers, batch_first=True, dropout=dropout if layers > 1 else 0.0)
        self.post = nn.Sequential(nn.LayerNorm(hidden), nn.Linear(hidden, hidden), nn.GELU(), nn.Dropout(dropout))
        self.head = MonotoneQuantileHead(hidden, n_quantiles, median_index)

    def forward(self, x: torch.Tensor, static=None, past_dow=None, future_dow=None) -> torch.Tensor:
        out, _ = self.rnn(F.gelu(self.inp(x)))
        return self.head(self.post(out[:, -1]))                         # (B, Q)


# ============================================================================== TFT
class GLU(nn.Module):
    def __init__(self, d_in: int, d_out: int):
        super().__init__()
        self.lin = nn.Linear(d_in, 2 * d_out)

    def forward(self, x):
        a, b = self.lin(x).chunk(2, dim=-1)
        return a * torch.sigmoid(b)


class GRN(nn.Module):
    """Gated residual network: LayerNorm(skip(a) + GLU(W1 ELU(W2 a + W3 c)))."""

    def __init__(self, d_in: int, d_hidden: int, d_out: int, dropout: float, d_context: int | None = None):
        super().__init__()
        self.skip = nn.Linear(d_in, d_out) if d_in != d_out else nn.Identity()
        self.fc_a = nn.Linear(d_in, d_hidden)
        self.fc_c = nn.Linear(d_context, d_hidden, bias=False) if d_context else None
        self.fc2 = nn.Linear(d_hidden, d_hidden)
        self.drop = nn.Dropout(dropout)
        self.glu = GLU(d_hidden, d_out)
        self.norm = nn.LayerNorm(d_out)

    def forward(self, a, c=None):
        h = self.fc_a(a)
        if self.fc_c is not None and c is not None:
            h = h + self.fc_c(c)
        h = self.fc2(F.elu(h))
        return self.norm(self.skip(a) + self.glu(self.drop(h)))


class GroupedGRN(nn.Module):
    """V GRN independen (satu per variabel) dengan bobot terkelompok — tanpa loop Python per variabel."""

    def __init__(self, n_vars: int, d: int, dropout: float):
        super().__init__()
        self.w1 = nn.Parameter(torch.randn(n_vars, d, d) * d ** -0.5)
        self.b1 = nn.Parameter(torch.zeros(n_vars, d))
        self.w2 = nn.Parameter(torch.randn(n_vars, d, 2 * d) * d ** -0.5)
        self.b2 = nn.Parameter(torch.zeros(n_vars, 2 * d))
        self.drop = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(d)

    def forward(self, x):                                              # x (..., V, d)
        h = F.elu(torch.einsum("...vd,vde->...ve", x, self.w1) + self.b1)
        g = torch.einsum("...vd,vde->...ve", self.drop(h), self.w2) + self.b2
        a, b = g.chunk(2, dim=-1)
        return self.norm(x + a * torch.sigmoid(b))


class VariableSelection(nn.Module):
    def __init__(self, n_vars: int, d: int, dropout: float, d_context: int | None):
        super().__init__()
        self.weights_grn = GRN(n_vars * d, d, n_vars, dropout, d_context)
        self.var_grn = GroupedGRN(n_vars, d, dropout)

    def forward(self, emb, context=None):                              # emb (B, T, V, d); context (B, d)
        flat = emb.flatten(-2)
        if context is not None:
            context = context.unsqueeze(1).expand(-1, emb.shape[1], -1)
        w = torch.softmax(self.weights_grn(flat, context), dim=-1)     # (B, T, V)
        out = (self.var_grn(emb) * w.unsqueeze(-1)).sum(-2)            # (B, T, d)
        return out, w


class InterpretableMultiHeadAttention(nn.Module):
    """Head-specific Q/K, value DIBAGI antar head; keluaran = rata-rata head (Lim et al., §4.4)."""

    def __init__(self, d: int, heads: int, dropout: float):
        super().__init__()
        self.h, self.dk = heads, max(1, d // heads)
        self.q = nn.Linear(d, heads * self.dk)
        self.k = nn.Linear(d, heads * self.dk)
        self.v = nn.Linear(d, self.dk)
        self.out = nn.Linear(self.dk, d)
        self.drop = nn.Dropout(dropout)

    def forward(self, q, k, v, mask):
        B, Tq, _ = q.shape
        Tk = k.shape[1]
        Q = self.q(q).view(B, Tq, self.h, self.dk).transpose(1, 2)    # (B, h, Tq, dk)
        K = self.k(k).view(B, Tk, self.h, self.dk).transpose(1, 2)
        V = self.v(v).unsqueeze(1)                                     # (B, 1, Tk, dk) — dibagi
        att = (Q @ K.transpose(-1, -2)) / self.dk ** 0.5
        att = att.masked_fill(~mask, float("-inf"))
        att = self.drop(torch.softmax(att, dim=-1))
        heads = att @ V                                                # (B, h, Tq, dk)
        return self.out(heads.mean(1)), att.mean(1)                    # (B, Tq, d), (B, Tq, Tk)


class TemporalFusionTransformer(nn.Module):
    def __init__(self, n_features: int, n_sectors: int, horizon: int, hidden: int = 32, heads: int = 4,
                 dropout: float = 0.1, n_quantiles: int = 7, median_index: int = 3):
        super().__init__()
        d = hidden
        self.horizon = horizon
        # embedding: setiap fitur kontinu → vektor d; hari dalam minggu → embedding (variabel 'known')
        self.cont_w = nn.Parameter(torch.randn(n_features, d) * 0.1)
        self.cont_b = nn.Parameter(torch.zeros(n_features, d))
        self.dow_emb = nn.Embedding(7, d)
        self.sector_emb = nn.Embedding(n_sectors + 1, d)
        # static covariate encoders
        self.cs, self.ce, self.ch, self.cc = (GRN(d, d, d, dropout) for _ in range(4))
        # variable selection
        self.vsn_past = VariableSelection(n_features + 1, d, dropout, d)
        self.vsn_future = VariableSelection(1, d, dropout, d)
        # sequence-to-sequence
        self.encoder = nn.LSTM(d, d, batch_first=True)
        self.decoder = nn.LSTM(d, d, batch_first=True)
        self.gate_lstm = GLU(d, d)
        self.norm_lstm = nn.LayerNorm(d)
        self.enrich = GRN(d, d, d, dropout, d)
        self.attn = InterpretableMultiHeadAttention(d, heads, dropout)
        self.gate_attn = GLU(d, d)
        self.norm_attn = nn.LayerNorm(d)
        self.pos_grn = GRN(d, d, d, dropout)
        self.gate_out = GLU(d, d)
        self.norm_out = nn.LayerNorm(d)
        self.head = MonotoneQuantileHead(d, n_quantiles, median_index)

    def forward(self, x, static, past_dow, future_dow, return_weights: bool = False):
        """x (B, L, F) fitur teramati; static (B,) kode sektor; past_dow (B, L); future_dow (B, H).
        Keluaran: kuantil (B, H, Q) log return kumulatif 1..H hari."""
        B, L, _ = x.shape
        s = self.sector_emb(static)                                     # (B, d)
        c_s, c_e, c_h, c_c = self.cs(s), self.ce(s), self.ch(s), self.cc(s)
        past = torch.cat([x.unsqueeze(-1) * self.cont_w + self.cont_b, self.dow_emb(past_dow).unsqueeze(2)], dim=2)
        fut = self.dow_emb(future_dow).unsqueeze(2)                     # (B, H, 1, d)
        xp, w_past = self.vsn_past(past, c_s)
        xf, w_fut = self.vsn_future(fut, c_s)
        h0, c0 = c_h.unsqueeze(0).contiguous(), c_c.unsqueeze(0).contiguous()
        enc, (hn, cn) = self.encoder(xp, (h0, c0))
        dec, _ = self.decoder(xf, (hn, cn))
        lstm_out = torch.cat([enc, dec], dim=1)                         # (B, L+H, d)
        vsn_out = torch.cat([xp, xf], dim=1)
        phi = self.norm_lstm(vsn_out + self.gate_lstm(lstm_out))
        theta = self.enrich(phi, c_e.unsqueeze(1).expand(-1, phi.shape[1], -1))
        T = L + self.horizon
        mask = torch.ones(T, T, dtype=torch.bool, device=x.device).tril()[L:]   # query = posisi decoder; kausal
        q = theta[:, L:]
        att_out, att_w = self.attn(q, theta, theta, mask.unsqueeze(0).unsqueeze(0))
        delta = self.norm_attn(q + self.gate_attn(att_out))
        psi = self.pos_grn(delta)
        out = self.norm_out(phi[:, L:] + self.gate_out(psi))
        quantiles = self.head(out)                                      # (B, H, Q)
        if return_weights:
            return quantiles, {"past_variables": w_past.mean(dim=(0, 1)), "attention": att_w}
        return quantiles


def build_model(kind: str, n_features: int, n_sectors: int, horizon: int, arch: dict, n_quantiles: int,
                median_index: int) -> nn.Module:
    if kind in ("lstm", "gru"):
        return RNNQuantile(n_features, kind, int(arch.get("hidden", 64)), int(arch.get("layers", 2)),
                           float(arch.get("dropout", 0.1)), n_quantiles, median_index)
    if kind == "tft":
        return TemporalFusionTransformer(n_features, n_sectors, horizon, int(arch.get("hidden", 32)),
                                         int(arch.get("heads", 4)), float(arch.get("dropout", 0.1)), n_quantiles,
                                         median_index)
    raise ValueError(f"model sekuens tidak dikenal: {kind}")
