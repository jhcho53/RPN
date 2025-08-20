#!/usr/bin/env python3
# K-shot training for MCProp (same model as oneshot; path-based dataset & dataloader)
import os, json, argparse, math, random
from typing import Dict, Any, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

# ---- config / dataset / utils ----
from config.loader import load_config         # DEFAULT_CFG + user json override
from config.schema import MCPropCfg           # dataclass for model cfg (anchor options 포함)
from utils.dataset import KShotDataset, build_kshot_from_paths
from utils.io_utils import save_jet, save_sparse_jet, unfold_neighbors
from utils.loss import l1l2_composite, scale_invariant_log_loss, lidar_consistency, rmse_mm

# ---- model parts (동일) ----
from models.module import TinyFeat, ResidualHead, CurvatureGen, KernelGate, normalize_affinity_list, AnchorHead
from models.affinity import EllipticAffinity, HCLApproxAffinity


# -------------------- shape helpers --------------------
def _strip_extra_batch_dim(x: torch.Tensor) -> torch.Tensor:
    # (B,1,C,H,W) 같이 배치 뒤에 불필요한 1차원이 붙은 경우 제거
    while x.dim() > 4 and x.size(1) == 1:
        x = x.squeeze(1)
    return x

def _as_chw4(x: torch.Tensor) -> torch.Tensor:
    """
    x -> (B,C,H,W) 강제
    - (B,1,C,H,W) -> (B,C,H,W)
    - (B,H,W,C)   -> (B,C,H,W)
    """
    x = _strip_extra_batch_dim(x)
    if x.dim() == 4 and x.size(1) in (1, 3):
        return x
    if x.dim() == 4 and x.size(-1) in (1, 3) and x.size(1) not in (1, 3):
        return x.permute(0, 3, 1, 2).contiguous()
    raise RuntimeError(f"Expected 4D (B,C,H,W) or (B,H,W,C), got {tuple(x.shape)}")

def _as_1ch4(x: torch.Tensor) -> torch.Tensor:
    """
    depth/mask -> (B,1,H,W)
    - (B,1,1,H,W) -> (B,1,H,W)
    - (B,H,W)     -> (B,1,H,W)
    - (B,C,H,W) with C>1 -> 첫 채널
    """
    x = _strip_extra_batch_dim(x)
    if x.dim() == 3:  # (B,H,W)
        return x.unsqueeze(1)
    if x.dim() == 4 and x.size(1) == 1:
        return x
    if x.dim() == 4 and x.size(1) > 1:
        return x[:, :1, ...]
    raise RuntimeError(f"Expected 3D (B,H,W) or 4D with C>=1, got {tuple(x.shape)}")

def _resize_like(x: torch.Tensor, ref: torch.Tensor, mode: str = "bilinear") -> torch.Tensor:
    """x를 ref의 (H,W)에 맞추어 리사이즈. mask류는 nearest."""
    H, W = ref.shape[-2:]
    if x.shape[-2:] == (H, W):
        return x
    if mode == "bilinear":
        return F.interpolate(x, size=(H, W), mode="bilinear", align_corners=False)
    else:
        return F.interpolate(x, size=(H, W), mode="nearest")


# -------------------- Model (oneshot과 동일) --------------------
class MCPropNet(nn.Module):
    def __init__(self, cfg: MCPropCfg):
        super().__init__()
        self.cfg = cfg
        in_ch = 3 + 1 + 1 + 1  # RGB, P/dmax, E_norm, ML
        self.enc = TinyFeat(in_ch, 64)
        self.res = ResidualHead(64, 64) if cfg.use_residual else None
        self.curv = CurvatureGen(64 + 1, cfg.kernels, cfg.kappa_min, cfg.kappa_max)
        self.gate = KernelGate(64, cfg.kernels)

        # geometry-specific affinity
        if cfg.geometry.lower().startswith("ellip"):
            self.aff_head = EllipticAffinity(64, cfg.kernels, c_aff=32, tau_min=0.03, tau_max=0.5)
        else:
            self.aff_head = HCLApproxAffinity(64, cfg.kernels)

        # Learnable anchor
        self.anchor_learnable = bool(cfg.anchor_learnable)
        self.anchor_mode = cfg.anchor_mode.lower()
        self.anchor_init = float(cfg.anchor_alpha)
        if self.anchor_learnable:
            if self.anchor_mode == "scalar":
                self.anchor_param = nn.Parameter(torch.tensor(self.anchor_init, dtype=torch.float32))
                self.anchor_head = None
            elif self.anchor_mode == "map":
                self.anchor_param = None
                self.anchor_head = AnchorHead(64)
            else:
                raise ValueError(f"Unknown anchor_mode: {cfg.anchor_mode}")
        else:
            self.register_buffer("anchor_fixed", torch.tensor(self.anchor_init, dtype=torch.float32))
            self.anchor_param = None
            self.anchor_head = None

    def _alpha_map(self, feat, ML):
        """α map in [0,1], (B,1,H,W)"""
        if self.anchor_learnable:
            if self.anchor_mode == "scalar":
                a = torch.sigmoid(self.anchor_param)
                return a.view(1,1,1,1).expand_as(ML)
            else:
                return self.anchor_head(feat)
        else:
            return self.anchor_fixed.view(1,1,1,1).expand_as(ML)

    def forward(self, I, DL, ML, P, E_norm):
        cfg = self.cfg

        # --- Robust input normalization to (B,C,H,W) ---
        I  = _as_chw4(I.float())
        DL = _as_1ch4(DL.float())
        ML = _as_1ch4(ML.float())
        P  = _as_1ch4(P.float())
        E  = _as_1ch4(E_norm.float())

        # Estimation을 [0,1]로 (혹시 0~255면)
        if E.max() > 1.5:
            E = E / 255.0

        # 해상도 정렬: RGB 기준
        I01 = I / 255.0
        DL  = _resize_like(DL, I, mode="bilinear")
        ML  = _resize_like((ML > 0).float(), I, mode="nearest")
        P   = _resize_like(P,  I, mode="bilinear")
        E   = _resize_like(E,  I, mode="bilinear")

        x_in = torch.cat([I01, P/cfg.dmax, E, ML], dim=1)
        feat = self.enc(x_in)

        D0 = (P + self.res(feat)).clamp(0, cfg.dmax) if self.res is not None else P

        # curvature & affinity
        kappa, scale, bias = self.curv(feat, E)
        if cfg.geometry.lower().startswith("ellip"):
            A_list_raw = self.aff_head(feat, kappa)
        else:
            A_list_raw = self.aff_head(feat, scale, bias)
        A_list = normalize_affinity_list(A_list_raw)  # per-kernel normalized
        sigma  = self.gate(feat)                       # (B,K,H,W)

        alpha = self._alpha_map(feat, ML)              # (B,1,H,W)

        # multi‑kernel propagation
        Dt = D0.clone()
        for _ in range(cfg.steps):
            mix = torch.zeros_like(Dt)
            for idx, k in enumerate(cfg.kernels):
                Ak = A_list[idx]                     # (B, k*k, H, W)
                kk = k*k
                patches = unfold_neighbors(Dt, k)    # (B, kk, H, W)
                center = kk // 2
                patches_center = patches.clone()
                patches_center[:, center:center+1, :, :] = D0  # center ← D0
                Dk = (Ak * patches_center).sum(1, keepdim=True)
                mix = mix + sigma[:, idx:idx+1] * Dk

            if cfg.use_sparse:
                Dt = (1.0 - alpha*ML) * mix + (alpha*ML) * DL
            else:
                Dt = mix
            Dt = Dt.clamp(0, cfg.dmax)

        aux = {"D0": D0, "sigma": sigma, "alpha": alpha}
        return Dt, aux


# -------------------- Train / Val --------------------
def set_seed(seed=1):
    random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)

def move_to_device(batch, device): return [x.to(device, non_blocking=True) for x in batch]

@torch.no_grad()
def evaluate(model: MCPropNet, loader: DataLoader, device, mu_scaleinv=0.1, w_lidar=0.3):
    model.eval()
    tot = {"L":0.0,"L1L2":0.0,"SI":0.0,"LiDAR":0.0,"RMSEmm":0.0}
    n = 0
    for I, DL, ML, P, E, GT in loader:
        I, DL, ML, P, E, GT = move_to_device([I, DL, ML, P, E, GT], device)
        pred, _ = model(I, DL, ML, P, E)

        # GT 해상도 보정(안전)
        if GT is not None and GT.numel() > 0 and GT.shape[-2:] != pred.shape[-2:]:
            GT = F.interpolate(GT, size=pred.shape[-2:], mode="nearest")

        L_l1l2  = l1l2_composite(pred, GT)
        L_si    = scale_invariant_log_loss(E + 1e-3, GT)
        L_lidar = lidar_consistency(pred, DL, ML) if model.cfg.use_sparse else pred.new_tensor(0.0)
        L       = L_l1l2 + mu_scaleinv * L_si + w_lidar * L_lidar

        rm      = rmse_mm(pred, GT)
        tot["L"] += float(L); tot["L1L2"] += float(L_l1l2); tot["SI"] += float(L_si); tot["LiDAR"] += float(L_lidar)
        tot["RMSEmm"] += (0.0 if math.isnan(rm) else rm); n += 1

    avg = {k: v/max(1,n) for k,v in tot.items()}
    return avg


def train_kshot(cfg: Dict[str,Any]):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    ks = cfg.get("kshot", {})
    set_seed(int(ks.get("seed", 1)))

    # ---- K-shot splits from directory paths ----
    # cfg["kshot"]에 train_dirs/val_dirs, K_train/K_val 등 정의되어 있어야 함
    splits = build_kshot_from_paths(cfg)
    train_ds = KShotDataset(splits["train"])
    val_ds   = KShotDataset(splits["val"])

    bs   = int(ks.get("batch_size", 4))
    nw   = int(ks.get("num_workers", 4))
    shuf = bool(ks.get("shuffle", True))
    epochs = int(ks.get("epochs", 60))
    lr     = float(ks.get("lr", 1e-3))
    save_dir = ks.get("save_dir", "runs_kshot"); os.makedirs(save_dir, exist_ok=True)
    tag      = ks.get("tag", "mcprop_kshot")
    preview_every = int(ks.get("preview_every", 5))

    train_loader = DataLoader(train_ds, batch_size=bs, shuffle=shuf, num_workers=nw, pin_memory=True)
    val_loader   = DataLoader(val_ds,   batch_size=bs, shuffle=False, num_workers=nw, pin_memory=True)

    # ---- Model / Optim ----
    mcfg = cfg["model"]
    model = MCPropNet(MCPropCfg(
        dmax=mcfg["dmax"], steps=mcfg["steps"], kernels=tuple(mcfg["kernels"]),
        use_residual=mcfg["use_residual"], use_sparse=mcfg["use_sparse"],
        anchor_alpha=mcfg["anchor_alpha"], anchor_learnable=mcfg["anchor_learnable"],
        anchor_mode=mcfg["anchor_mode"], kappa_min=mcfg["kappa_min"], kappa_max=mcfg["kappa_max"],
        geometry=mcfg["geometry"]
    )).to(device)

    # loss weights (dataclass 안 써도 무방; json 없으면 default 사용)
    mu_scaleinv = float(cfg.get("loss", {}).get("mu_scaleinv", 0.1))
    w_lidar     = float(cfg.get("loss", {}).get("w_lidar", 0.3))
    w_anchor    = float(cfg.get("loss", {}).get("w_anchor_reg", 0.0))

    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-2)

    best_rmse = float("inf")
    best_path = os.path.join(save_dir, f"{tag}_best.pt")

    # ---- Train loop ----
    for ep in range(1, epochs+1):
        model.train()
        tot = {"L":0.0,"L1L2":0.0,"SI":0.0,"LiDAR":0.0,"Anchor":0.0,"RMSEmm":0.0}
        n = 0

        for I, DL, ML, P, E, GT in train_loader:
            I, DL, ML, P, E, GT = move_to_device([I, DL, ML, P, E, GT], device)
            pred, aux = model(I, DL, ML, P, E)

            # GT 해상도 보정(안전)
            if GT is not None and GT.numel() > 0 and GT.shape[-2:] != pred.shape[-2:]:
                GT = F.interpolate(GT, size=pred.shape[-2:], mode="nearest")

            L_l1l2  = l1l2_composite(pred, GT)
            L_si    = scale_invariant_log_loss(E + 1e-3, GT)
            L_lidar = lidar_consistency(pred, DL, ML) if model.cfg.use_sparse else pred.new_tensor(0.0)

            # Anchor regularization: LiDAR 위치에서 α→1 유도(옵션)
            L_anchor = pred.new_tensor(0.0)
            if w_anchor > 0.0 and model.cfg.use_sparse:
                alpha = aux["alpha"].detach() if not model.anchor_learnable else aux["alpha"]
                L_anchor = ((1.0 - alpha) * ML).mean() * w_anchor

            L = L_l1l2 + mu_scaleinv * L_si + w_lidar * L_lidar + L_anchor

            opt.zero_grad(set_to_none=True); L.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()

            rm = rmse_mm(pred, GT)
            tot["L"]+=float(L); tot["L1L2"]+=float(L_l1l2); tot["SI"]+=float(L_si)
            tot["LiDAR"]+=float(L_lidar); tot["Anchor"]+=float(L_anchor)
            tot["RMSEmm"]+= (0.0 if math.isnan(rm) else rm); n+=1

        avg_tr = {k: v/max(1,n) for k,v in tot.items()}
        avg_va = evaluate(model, val_loader, device, mu_scaleinv, w_lidar)

        print(f"[{tag}] Ep {ep:03d} | "
              f"Train L={avg_tr['L']:.4f} L1L2={avg_tr['L1L2']:.4f} SI={avg_tr['SI']:.4f} "
              f"Lidar={avg_tr['LiDAR']:.4f} Anchor={avg_tr['Anchor']:.4f} RMSE={avg_tr['RMSEmm']:.1f} | "
              f"Val L={avg_va['L']:.4f} L1L2={avg_va['L1L2']:.4f} SI={avg_va['SI']:.4f} "
              f"Lidar={avg_va['LiDAR']:.4f} RMSE={avg_va['RMSEmm']:.1f}")

        # 미리보기 저장
        if (ep % max(1, ks.get("preview_every", 5)) == 0) or (ep == 1):
            with torch.no_grad():
                for I, DL, ML, P, E, GT in val_loader:
                    I, DL, ML, P, E, GT = move_to_device([I, DL, ML, P, E, GT], device)
                    pred, aux = model(I, DL, ML, P, E)
                    save_jet(os.path.join(save_dir, f"{tag}_ep{ep:03d}_pred_jet.png"), pred[:1], dmax=model.cfg.dmax)
                    save_jet(os.path.join(save_dir, f"{tag}_ep{ep:03d}_d0_jet.png"),   aux["D0"][:1], dmax=model.cfg.dmax)
                    save_sparse_jet(os.path.join(save_dir, f"{tag}_ep{ep:03d}_sparse_jet.png"), DL[:1], ML[:1], dmax=model.cfg.dmax)
                    break

        # 베스트(Val RMSE)
        if math.isfinite(avg_va["RMSEmm"]) and avg_va["RMSEmm"] < best_rmse:
            best_rmse = avg_va["RMSEmm"]
            torch.save({
                "state_dict": model.state_dict(),
                "cfg": model.cfg.__dict__,
                "val_rmse_mm": best_rmse
            }, best_path)
            print(f"  -> best Val RMSE {best_rmse:.1f} mm (saved: {best_path})")

    print(f"[{tag}] Finished. Best Val RMSE(mm)={best_rmse:.1f}")
    return best_rmse


# -------------------- Entry --------------------
def main():
    parser = argparse.ArgumentParser("K-shot training for MCProp (path-based)")
    parser.add_argument("--config", type=str, default="", help="JSON/YAML config path (optional)")
    args = parser.parse_args()

    cfg = load_config(args.config)   # DEFAULT_CFG + user override
    train_kshot(cfg)

if __name__ == "__main__":
    main()
