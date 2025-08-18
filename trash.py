# oneshot_mcprop_geo_fixed.py
# 1-shot Multi-geometry Propagation (Hyperbolic approx. & Elliptic/Spherical) with learnable affinity
# - Estimation(8bit) -> [0,1] normalization
# - Learnable propagation: curvature-conditioned affinity + kernel gate
# - Residual-on-pseudo, soft-Dirichlet LiDAR anchor
# - Saves sparse LiDAR jetmap along with predictions

import os, json, math, argparse
from dataclasses import dataclass
from typing import Dict, Any, Tuple, List, Optional

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


# -------------------- I/O utils --------------------

def load_rgb(path: str) -> torch.Tensor:
    im = cv2.imread(path, cv2.IMREAD_COLOR)
    if im is None: raise FileNotFoundError(path)
    im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB).astype(np.float32)
    return torch.from_numpy(im).permute(2,0,1).unsqueeze(0)  # (1,3,H,W)

def load_depth16_mm_as_m(path: str) -> torch.Tensor:
    d = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    if d is None: raise FileNotFoundError(path)
    if d.ndim == 3: d = cv2.cvtColor(d, cv2.COLOR_BGR2GRAY)
    d = d.astype(np.float32) / 256.0
    return torch.from_numpy(d).unsqueeze(0).unsqueeze(0)     # (1,1,H,W)

def load_pseudo_auto(path: str, zmin=0.5, zmax=80.0) -> torch.Tensor:
    """pseudo는 16-bit(m*256) 또는 8-bit일 수 있음 → m 단위로 반환"""
    x = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    if x is None: raise FileNotFoundError(path)
    if x.ndim == 3: x = cv2.cvtColor(x, cv2.COLOR_BGR2GRAY)
    if x.dtype == np.uint16 and x.max() > 255:
        z = x.astype(np.float32) / 256.0
    else:
        z = (x.astype(np.float32)/255.0)*(zmax-zmin)+zmin
    return torch.from_numpy(z).unsqueeze(0).unsqueeze(0)

def load_estimation_8bit_norm(path: str) -> torch.Tensor:
    """Estimation 0-255 → [0,1]"""
    x = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    if x is None: raise FileNotFoundError(path)
    if x.ndim == 3: x = cv2.cvtColor(x, cv2.COLOR_BGR2GRAY)
    x = x.astype(np.float32)/255.0
    return torch.from_numpy(x).unsqueeze(0).unsqueeze(0)

def save_jet(path: str, z_m: torch.Tensor, dmin=0.5, dmax=80.0):
    z = z_m.detach().cpu().clamp(dmin, dmax)
    z = ((z - dmin) / (dmax - dmin + 1e-6)).squeeze().numpy()
    z8 = np.uint8(np.clip(z*255.0, 0, 255))
    jet = cv2.applyColorMap(z8, cv2.COLORMAP_JET)
    cv2.imwrite(path, jet)

def save_sparse_jet(path: str, DL: torch.Tensor, ML: torch.Tensor, dmin=0.5, dmax=80.0):
    """유효 LiDAR만 컬러, 나머지는 검정."""
    z = DL.detach().cpu().clamp(dmin, dmax).squeeze().numpy()
    m = (ML.detach().cpu().squeeze().numpy() > 0).astype(np.uint8)
    z_norm = np.uint8(np.clip((z - dmin) / (dmax - dmin + 1e-6) * 255.0, 0, 255))
    jet = cv2.applyColorMap(z_norm, cv2.COLORMAP_JET)
    jet[m == 0] = (0, 0, 0)  # invalid은 검정
    cv2.imwrite(path, jet)

def unfold_neighbors(x: torch.Tensor, k: int) -> torch.Tensor:
    pad = k//2
    patches = F.unfold(x, kernel_size=k, padding=pad)  # (B, k*k, H*W) for 1ch
    return patches.view(x.size(0), k*k, x.size(2), x.size(3))  # (B, kk, H, W)


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


# -------- Hyperbolic-inspired (approx.) affinity --------

class HCLApproxAffinity(nn.Module):
    """
    (Approx.) Hyperbolic Convolution for affinity maps
      A_k = Conv_k( scale_k * feat + bias_k )  -> (B, k*k, H, W)
    이후 normalize_affinity_list()로 안정화.
    """
    def __init__(self, in_ch=64, K: Tuple[int,...]=(3,5,7)):
        super().__init__()
        self.K = K
        self.convs = nn.ModuleDict()
        for k in K:
            kk = k*k
            self.convs[str(k)] = nn.Conv2d(in_ch, kk, kernel_size=3, stride=1, padding=1, bias=True)

    def forward(self, feat, scale, bias) -> List[torch.Tensor]:
        out = []
        for idx, k in enumerate(self.K):
            s = scale[:, idx:idx+1]  # (B,1,H,W)
            b = bias[:,  idx:idx+1]
            Fm = s * feat + b        # FiLM
            Ak = self.convs[str(k)](Fm)  # (B, k*k, H, W)
            out.append(Ak)
        return out


# -------- Elliptic/Spherical affinity (ours) --------

class EllipticAffinity(nn.Module):
    """
    Local spherical (elliptic) affinity via cosine similarity with temperature from curvature.
      - For each kernel k:
          q_k = Conv1x1(feat), k_k = Conv1x1(feat)  (Ca channels)
          Ak_logits(i, offsets) = cos( q_k(i), k_k(j) ) / tau_k(i)
      - logits → tanh → L1-normalize (signed CSPN style).
    """
    def __init__(self, in_ch=64, K: Tuple[int,...]=(3,5,7), c_aff: int = 32,
                 tau_min: float = 0.03, tau_max: float = 0.5):
        super().__init__()
        self.K = K
        self.tau_min = tau_min
        self.tau_max = tau_max
        self.qconvs = nn.ModuleDict()
        self.kconvs = nn.ModuleDict()
        for k in K:
            self.qconvs[str(k)] = nn.Conv2d(in_ch, c_aff, kernel_size=1, bias=False)
            self.kconvs[str(k)] = nn.Conv2d(in_ch, c_aff, kernel_size=1, bias=False)

    def _tau_from_kappa(self, kappa: torch.Tensor) -> torch.Tensor:
        """kappa ∈ [kmin,kmax] -> tau ∈ [tau_min, tau_max]"""
        kappa_norm = (kappa - kappa.min()) / (kappa.max() - kappa.min() + 1e-6)
        tau = self.tau_max - kappa_norm * (self.tau_max - self.tau_min)
        return tau.clamp(self.tau_min, self.tau_max)  # (B,1,H,W)

    def forward(self, feat, kappa) -> List[torch.Tensor]:
        out = []
        for idx, k in enumerate(self.K):
            q = self.qconvs[str(k)](feat)                    # (B,Ca,H,W)
            key = self.kconvs[str(k)](feat)                  # (B,Ca,H,W)
            qn = F.normalize(q, dim=1)
            kn = F.normalize(key, dim=1)

            kk = k*k
            patches = F.unfold(kn, kernel_size=k, padding=k//2)  # (B, Ca*kk, H*W)
            B, Ca, H, W = kn.shape
            patches = patches.view(B, Ca, kk, H, W)              # (B,Ca,kk,H,W)

            sim = (qn.unsqueeze(2) * patches).sum(1)             # (B,kk,H,W), cosine
            tau = self._tau_from_kappa(kappa[:, idx:idx+1])      # (B,1,H,W)
            logits = sim / (tau + 1e-6)                          # (B,kk,H,W)
            out.append(logits)
        return out


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


# -------------------- Full model --------------------

@dataclass
class MCPropCfg:
    dmax: float = 80.0
    steps: int = 6
    kernels: Tuple[int,...] = (3,5,7)
    use_residual: bool = True
    use_sparse: bool = True
    anchor_alpha: float = 0.7         # LiDAR soft-Dirichlet
    kappa_min: float = 1e-3
    kappa_max: float = 1.0
    geometry: str = "hyperbolic"      # "hyperbolic" | "elliptic"

class MCPropNet(nn.Module):
    """
    Multi-geometry (hyperbolic approx. / elliptic spherical) learnable propagation
    """
    def __init__(self, cfg: MCPropCfg):
        super().__init__()
        self.cfg = cfg
        in_ch = 3+1+1+1  # RGB, P/dmax, E_norm, ML

        self.enc = TinyFeat(in_ch, 64)
        self.res = ResidualHead(64, 64) if cfg.use_residual else None

        self.curv = CurvatureGen(64+1, cfg.kernels, cfg.kappa_min, cfg.kappa_max)
        self.gate = KernelGate(64, cfg.kernels)

        if cfg.geometry.lower().startswith("ellip"):
            self.aff_head = EllipticAffinity(64, cfg.kernels, c_aff=32, tau_min=0.03, tau_max=0.5)
        else:
            self.aff_head = HCLApproxAffinity(64, cfg.kernels)

    def forward(self, I, DL, ML, P, E_norm):
        """
        I: (B,3,H,W)  RGB [0..255]
        DL: (B,1,H,W) m
        ML: (B,1,H,W) {0,1}
        P:  (B,1,H,W) m
        E_norm: (B,1,H,W) [0,1] (estimation)
        """
        cfg = self.cfg
        I01 = I / 255.0
        x_in = torch.cat([I01, P/cfg.dmax, E_norm, ML], dim=1)
        feat = self.enc(x_in)                     # (B,64,H,W)

        # Initial depth
        if self.res is not None:
            D0 = (P + self.res(feat)).clamp(0, cfg.dmax)
        else:
            D0 = P

        # Curvature (and FiLM)
        kappa, scale, bias = self.curv(feat, E_norm)     # each: (B,K,H,W)

        # Affinity per geometry
        if cfg.geometry.lower().startswith("ellip"):
            A_list_raw = self.aff_head(feat, kappa)            # logits list
        else:
            A_list_raw = self.aff_head(feat, scale, bias)      # conv list

        A_list = normalize_affinity_list(A_list_raw)           # normalized
        sigma  = self.gate(feat)                               # (B,K,H,W)

        # Propagation
        Dt = D0.clone()
        for _ in range(cfg.steps):
            mix_k = []
            for idx, k in enumerate(cfg.kernels):
                Ak = A_list[idx]                      # (B,kk,H,W)
                kk = k*k
                patches = unfold_neighbors(Dt, k)     # (B,kk,H,W)
                # center index in unfold ordering (row-major)
                center = kk // 2
                patches_center = patches.clone()
                patches_center[:, center:center+1, :, :] = D0  # center ← D0
                D_next_k = (Ak * patches_center).sum(1, keepdim=True)  # (B,1,H,W)
                mix_k.append(D_next_k)

            # Σ_k σ_k * D_{t+1,k}
            Dmix = torch.zeros_like(Dt)
            for idx, Dk in enumerate(mix_k):
                Dmix = Dmix + (sigma[:, idx:idx+1] * Dk)

            # LiDAR soft-Dirichlet
            if self.cfg.use_sparse:
                Dt = (1.0 - self.cfg.anchor_alpha*ML) * Dmix + (self.cfg.anchor_alpha*ML) * DL
            else:
                Dt = Dmix

            Dt = Dt.clamp(0, cfg.dmax)

        aux = {"D0": D0, "sigma": sigma, "A_list": A_list, "kappa": kappa}
        return Dt, aux


# -------------------- Losses --------------------

def l1l2_composite(pred: torch.Tensor, gt: Optional[torch.Tensor]) -> torch.Tensor:
    if gt is None: return pred.new_tensor(0.0)
    M = (gt>0).float(); n = M.sum().clamp_min(1.0)
    e = (pred-gt)*M
    return (e.abs() + e.pow(2)).sum()/n

def scale_invariant_log_loss(Drel_pos: torch.Tensor, Dgt: torch.Tensor, eps=1e-6) -> torch.Tensor:
    """Drel_pos는 양수면 스케일 불변; 여기서는 E_norm(0..1)+eps 사용 권장."""
    if Dgt is None: return Drel_pos.new_tensor(0.0)
    M = (Dgt>0).float(); n = M.sum().clamp_min(1.0)
    x = torch.log(Drel_pos.clamp_min(eps))
    y = torch.log(Dgt.clamp_min(eps))
    d = (x-y)*M
    return (d.pow(2).sum()/n) - (d.sum()/n).pow(2)

def lidar_consistency(pred: torch.Tensor, DL: torch.Tensor, ML: torch.Tensor) -> torch.Tensor:
    n = ML.sum().clamp_min(1.0)
    return ((pred-DL).abs()*ML).sum()/n


# -------------------- 1-shot train loop --------------------

@dataclass
class TrainCfg:
    epochs: int = 200
    lr: float = 1e-3
    save_dir: str = "runs_oneshot"
    tag: str = "mcprop_1shot"

@dataclass
class LossW:
    mu_scaleinv: float = 0.1
    w_lidar: float    = 0.3

@dataclass
class OneShotPaths:
    rgb: str
    sparse: str
    pseudo: str
    estim: str
    gt: Optional[str] = None

def rmse_mm(pred: torch.Tensor, gt: Optional[torch.Tensor]) -> float:
    if gt is None: return float('nan')
    M = (gt>0)
    if M.sum()==0: return float('nan')
    e2 = (pred[M]-gt[M])**2
    return float(torch.sqrt(e2.mean()).item()*1000.0)

def load_oneshot(paths: OneShotPaths, device="cuda"):
    I  = load_rgb(paths.rgb).to(device)
    DL = load_depth16_mm_as_m(paths.sparse).to(device)
    ML = (DL>0).float()
    P  = load_pseudo_auto(paths.pseudo).to(device)
    E  = load_estimation_8bit_norm(paths.estim).to(device)     # <-- [0,1]
    GT = load_depth16_mm_as_m(paths.gt).to(device) if (paths.gt and os.path.isfile(paths.gt)) else None
    return I, DL, ML, P, E, GT

def train_oneshot(model: MCPropNet,
                  I, DL, ML, P, E_norm, GT,
                  tcfg: TrainCfg, w: LossW) -> float:
    os.makedirs(tcfg.save_dir, exist_ok=True)
    opt = torch.optim.AdamW(model.parameters(), lr=tcfg.lr, weight_decay=1e-2)
    best = float('inf')
    best_path = os.path.join(tcfg.save_dir, f"{tcfg.tag}_best.pt")

    # sparse jet 저장(한 번)
    save_sparse_jet(os.path.join(tcfg.save_dir, f"{tcfg.tag}_sparse_jet.png"), DL, ML, dmin=0.5, dmax=model.cfg.dmax)

    for ep in range(1, tcfg.epochs+1):
        model.train()
        pred, aux = model(I, DL, ML, P, E_norm)

        # Losses
        L_l1l2 = l1l2_composite(pred, GT)
        L_si   = scale_invariant_log_loss(E_norm + 1e-3, GT)
        L_lidar = lidar_consistency(pred, DL, ML) if model.cfg.use_sparse else pred.new_tensor(0.0)
        L = L_l1l2 + w.mu_scaleinv * L_si + w.w_lidar * L_lidar

        opt.zero_grad(set_to_none=True); L.backward(); opt.step()

        with torch.no_grad():
            rm = rmse_mm(pred, GT)
        print(f"[{tcfg.tag}] ep {ep:03d} | L={float(L):.4f}  L1L2={float(L_l1l2):.4f}  "
              f"SI={float(L_si):.4f}  LiDAR={float(L_lidar):.4f}  RMSE(mm)={rm:.1f}")

        if ep % max(1, tcfg.epochs//10) == 0 or ep==1:
            save_jet(os.path.join(tcfg.save_dir, f"{tcfg.tag}_ep{ep:03d}_pred_jet.png"), pred, dmax=model.cfg.dmax)
            save_jet(os.path.join(tcfg.save_dir, f"{tcfg.tag}_ep{ep:03d}_d0_jet.png"),   aux["D0"], dmax=model.cfg.dmax)

        if np.isfinite(rm) and rm < best:
            best = rm
            torch.save({"state_dict": model.state_dict(),
                        "cfg": model.cfg.__dict__}, best_path)
            print(f"  -> best RMSE {best:.1f} mm  (saved: {best_path})")

    return best


# -------------------- Config / Entry --------------------

_DEFAULT_CFG: Dict[str, Any] = {
    "paths": {
        # ===== 사용자가 주신 경로 유지 =====
        "rgb":      "/home/vip/Desktop/DC/DenseLiDAR/datasets/kitti_raw/train/2011_09_26_drive_0001_sync/proj_depth/image_02/0000000005.png",
        "sparse":   "/home/vip/Desktop/DC/DenseLiDAR/datasets/data_depth_velodyne/train/2011_09_26_drive_0001_sync/proj_depth/velodyne_raw/image_02/0000000005.png",
        "pseudo":   "/home/vip/Desktop/DC/DenseLiDAR/datasets/pseudo_depth_map/train/2011_09_26_drive_0001_sync/proj_depth/velodyne_raw/image_02/0000000005.png",
        "estim":    "/home/vip/Desktop/DC/DenseLiDAR/datasets/kitti_raw_da/train/2011_09_26_drive_0001_sync/proj_depth/image_02/0000000005.png",
        "gt":       "/home/vip/Desktop/DC/DenseLiDAR/datasets/data_depth_annotated/train/2011_09_26_drive_0001_sync/proj_depth/groundtruth/image_02/0000000005.png"
    },
    "train": {
        "epochs": 200,
        "lr": 1e-3,
        "save_dir": "runs_oneshot",
        "tag": "mcprop_1shot"
    },
    "model": {
        "dmax": 80.0,
        "steps": 6,
        "use_residual": False,
        "use_sparse": True,
        "anchor_alpha": 0.7,
        "kappa_min": 1e-3,
        "kappa_max": 1.0,
        "kernels": [3,5,7],
        "geometry": "hyperbolic"   # "hyperbolic" or "elliptic"
    },
    "loss": {
        "mu_scaleinv": 0.1,
        "w_lidar": 0.3
    },
    # 선택: 두 설정을 한 번에 돌리는 ablation
    "experiments": [
        {"geometry": "hyperbolic", "tag": "mcprop_1shot_hyp"},
        {"geometry": "elliptic",   "tag": "mcprop_1shot_ellip"}
    ]
}

def make_model_from_cfg(mcfg: Dict[str,Any]) -> MCPropNet:
    oc = MCPropCfg(
        dmax=mcfg.get("dmax",80.0),
        steps=mcfg.get("steps",6),
        kernels=tuple(mcfg.get("kernels",[3,5,7])),
        use_residual=mcfg.get("use_residual",True),
        use_sparse=mcfg.get("use_sparse",True),
        anchor_alpha=mcfg.get("anchor_alpha",0.7),
        kappa_min=mcfg.get("kappa_min",1e-3),
        kappa_max=mcfg.get("kappa_max",1.0),
        geometry=mcfg.get("geometry","hyperbolic")
    )
    return MCPropNet(oc)

def main():
    parser = argparse.ArgumentParser("1-shot multi-geometry propagation (hyperbolic/elliptic)")
    parser.add_argument("--config", type=str, default="", help="JSON config path (optional)")
    args = parser.parse_args()

    cfg = dict(_DEFAULT_CFG)
    if args.config and os.path.isfile(args.config):
        with open(args.config,"r") as f: user = json.load(f)
        for k,v in user.items(): cfg[k]=v

    device = "cuda" if torch.cuda.is_available() else "cpu"
    paths = OneShotPaths(
        rgb=cfg["paths"]["rgb"],
        sparse=cfg["paths"]["sparse"],
        pseudo=cfg["paths"]["pseudo"],
        estim=cfg["paths"]["estim"],
        gt=cfg["paths"].get("gt","") or None
    )
    I, DL, ML, P, E_norm, GT = load_oneshot(paths, device=device)

    os.makedirs(cfg["train"]["save_dir"], exist_ok=True)

    # (옵션) ablation: 여러 geometry를 순차 학습
    if "experiments" in cfg and isinstance(cfg["experiments"], list) and len(cfg["experiments"])>0:
        for exp in cfg["experiments"]:
            mcfg = dict(cfg["model"]); mcfg["geometry"] = exp.get("geometry", mcfg.get("geometry","hyperbolic"))
            model = make_model_from_cfg(mcfg).to(device)

            tcfg  = TrainCfg(**cfg["train"])
            tcfg.tag = exp.get("tag", f"mcprop_1shot_{mcfg['geometry']}")
            wloss = LossW(**cfg["loss"])

            print(f"\n=== Run: geometry={mcfg['geometry']}  tag={tcfg.tag} ===")
            best = train_oneshot(model, I, DL, ML, P, E_norm, GT, tcfg, wloss)
            print(f"[{tcfg.tag}] Best RMSE(mm)={best:.1f}")
    else:
        model = make_model_from_cfg(cfg["model"]).to(device)
        tcfg  = TrainCfg(**cfg["train"])
        wloss = LossW(**cfg["loss"])

        best = train_oneshot(model, I, DL, ML, P, E_norm, GT, tcfg, wloss)
        print(f"Done. Best RMSE(mm)={best:.1f}")

if __name__ == "__main__":
    main()
