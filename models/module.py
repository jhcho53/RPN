import torch
import torch.nn as nn

from typing import Dict, Any, Tuple, List, Optional
# -------------------- Model blocks --------------------

class TinyFeat(nn.Module):
    """[RGB_norm, P/dmax, E_norm, ML] -> 64ch"""
    def __init__(self, in_ch=3+1+1+1, ch=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, ch, 3,1,1), nn.ReLU(inplace=True),
            nn.Conv2d(ch,   ch, 3,1,1), nn.ReLU(inplace=True),
            nn.Conv2d(ch,   ch, 3,1,1), nn.ReLU(inplace=True),
        )
    def forward(self, x): return self.net(x)

class ResidualHead(nn.Module):
    def __init__(self, in_ch=64, hid=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, hid, 3,1,1), nn.ReLU(inplace=True),
            nn.Conv2d(hid,   hid, 3,1,1), nn.ReLU(inplace=True),
            nn.Conv2d(hid,     1, 3,1,1),
        )
    def forward(self, feat): return self.net(feat)

class CurvatureGen(nn.Module):
    """
    Curvature & FiLM generator (shared)
      inputs: features feat (64ch) + E_norm (1ch)
      outputs per-kernel:
        - kappa_k(x) ∈ [kappa_min,kappa_max]
        - FiLM scale γ_k(x) ∈ [0.5, 1.5], bias β_k(x) ∈ [-0.5, 0.5]
    """
    def __init__(self, in_ch=64+1, K: Tuple[int,...]=(3,5,7),
                 kappa_min=1e-3, kappa_max=1.0):
        super().__init__()
        self.K = K
        self.kappa_min = kappa_min
        self.kappa_max = kappa_max
        out_ch = len(K)*3  # kappa, scale, bias
        self.body = nn.Sequential(
            nn.Conv2d(in_ch, 64, 3,1,1), nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, 3,1,1), nn.ReLU(inplace=True),
            nn.Conv2d(64, out_ch, 1,1,0),
        )

    def forward(self, feat, E_norm):
        x = torch.cat([feat, E_norm], dim=1)
        y = self.body(x)
        B, C, H, W = y.shape
        m = len(self.K)
        y = y.view(B, m, 3, H, W)  # [B, K, {kappa,scale,bias}, H, W]
        kappa = torch.sigmoid(y[:, :, 0]) * (self.kappa_max - self.kappa_min) + self.kappa_min
        scale = torch.sigmoid(y[:, :, 1]) * 1.0 + 0.5   # [0.5, 1.5]
        bias  = torch.tanh(  y[:, :, 2]) * 0.5          # [-0.5, 0.5]
        return kappa, scale, bias

# -------- shared helpers --------

def normalize_affinity_list(A_list: List[torch.Tensor]) -> List[torch.Tensor]:
    """각 커널별 채널(k*k)을 tanh→L1-normalize (signed)."""
    out = []
    for Ak in A_list:
        Ak = torch.tanh(Ak)                      # [-1,1]
        sum_abs = Ak.abs().sum(1, keepdim=True) + 1e-6
        Ak = Ak / (1.1 * sum_abs)               # 안정화
        out.append(Ak)
    return out

class KernelGate(nn.Module):
    """σ_k(x): 커널 혼합 게이트 (softmax)"""
    def __init__(self, in_ch=64, K: Tuple[int,...]=(3,5,7)):
        super().__init__()
        self.K = K
        self.head = nn.Sequential(
            nn.Conv2d(in_ch, 32, 3,1,1), nn.ReLU(inplace=True),
            nn.Conv2d(32, len(K), 1,1,0)
        )
    def forward(self, feat):
        g = self.head(feat)                  # (B,K,H,W)
        return torch.softmax(g, dim=1)       # σ over K
    
# learnable anchor (α map) head
class AnchorHead(nn.Module):
    """feat -> α(x) in [0,1]"""
    def __init__(self, in_ch=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, 32, 3,1,1), nn.ReLU(inplace=True),
            nn.Conv2d(32,  1,  1,1,0)
        )
    def forward(self, feat):
        return torch.sigmoid(self.net(feat))