import math
import torch
import torch.nn as nn
import torch.nn.functional as F

def _trunc_normal_(tensor, mean=0.0, std=0.02):
    with torch.no_grad():
        size = tensor.shape
        tmp = tensor.new_empty(size + (4,)).normal_()
        valid = (tmp < 2) & (tmp > -2)
        ind = valid.max(-1, keepdim=True)[1]
        tensor.data.copy_(tmp.gather(-1, ind).squeeze(-1))
        tensor.data.mul_(std).add_(mean)
        return tensor

class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 500):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe.unsqueeze(0))

    def forward(self, x):
        L = x.size(1)
        return x + self.pe[:, :L, :]

class AdaptiveAdjacency(nn.Module):
    def __init__(self, num_nodes: int, rank: int = 8, symmetric: bool = True):
        super().__init__()
        self.E1 = nn.Parameter(torch.randn(num_nodes, rank) * 0.1)
        self.E2 = nn.Parameter(torch.randn(num_nodes, rank) * 0.1)
        self.symmetric = symmetric

    def forward(self):
        A = F.relu(self.E1 @ self.E2.t())
        if self.symmetric:
            A = 0.5 * (A + A.t())
        A = F.softmax(A, dim=-1)
        return A

class NodeGraphBlock(nn.Module):
    def __init__(self, d_model: int, dropout: float):
        super().__init__()
        self.lin = nn.Linear(d_model, d_model, bias=False)
        self.ln = nn.LayerNorm(d_model)
        self.mlp = nn.Sequential(
            nn.Linear(d_model, 4*d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(4*d_model, d_model),
            nn.Dropout(dropout),
        )

    def forward(self, x, A):
        y = torch.matmul(A, x)
        y = self.lin(y)
        z = self.ln(y + x)
        z = z + self.mlp(z)
        return z

class TemporalEncoder(nn.Module):
    def __init__(self, in_ch: int, d_model: int, n_heads: int, n_layers: int,
                 patch_len: int, stride: int, dropout: float):
        super().__init__()
        self.p = patch_len
        self.s = stride
        self.proj = nn.Linear(in_ch * patch_len, d_model)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_heads,
            dim_feedforward=4*d_model, dropout=dropout,
            batch_first=True, activation="gelu", norm_first=True
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        self.pos = PositionalEncoding(d_model, max_len=512)
        self.cls = nn.Parameter(torch.zeros(1, 1, d_model))
        _trunc_normal_(self.cls, std=0.02)

    def forward(self, x):  # x: (B,T,N,C)
        B, T, N, C = x.shape
        x = x.permute(0, 2, 3, 1).contiguous().view(B*N, C, T)
        patches = x.unfold(dimension=2, size=self.p, step=self.s)
        L = patches.size(2)
        patches = patches.permute(0, 2, 3, 1).reshape(B*N, L, self.p*C)
        tok = self.proj(patches)
        tok = self.pos(tok)
        enc = self.encoder(tok)
        h = enc.mean(dim=1)
        h = h.view(B, N, -1)
        return h

class PTSTG(nn.Module):
    def __init__(self,
                 node_num: int,
                 input_dim: int,
                 output_dim: int,
                 horizon: int,
                 d_model: int = 128,
                 n_heads: int = 4,
                 n_layers: int = 2,
                 patch_len: int = 4,
                 stride: int = 2,
                 graph_rank: int = 8,
                 graph_layers: int = 1,
                 dropout: float = 0.1):
        super().__init__()
        self.node_num = node_num
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.horizon = horizon

        self.temporal = TemporalEncoder(
            in_ch=input_dim, d_model=d_model, n_heads=n_heads, n_layers=n_layers,
            patch_len=patch_len, stride=stride, dropout=dropout
        )

        self.adpA = AdaptiveAdjacency(node_num, rank=graph_rank, symmetric=True)
        self.graph_blocks = nn.ModuleList([NodeGraphBlock(d_model, dropout) for _ in range(graph_layers)])
        self.out_norm = nn.LayerNorm(d_model)
        self.head = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, horizon * output_dim)
        )

        self._reset()

    def _reset(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                _trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, X, label=None):
        B, _, N, _ = X.shape
        h = self.temporal(X)
        A = self.adpA()
        for blk in self.graph_blocks:
            h = blk(h, A)
        h = self.out_norm(h)
        y = self.head(h)
        y = y.view(B, N, self.horizon, self.output_dim).permute(0, 2, 1, 3).contiguous()
        return y

    def param_num(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
