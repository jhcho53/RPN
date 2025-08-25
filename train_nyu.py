#!/usr/bin/env python3
# NYUv2 One-shot training for MCProp (Images + mm16 Depth + Viz Estimation)
# - 이미지/깊이(mm16)에서 rgb/gt를 읽고,
# - estimation(모노)은 nyu.mono_root에서 *_viz.png(3번 데이터)로 로드
# - Poisson depth completion으로 초기 깊이 P 생성
# - Center crop(기본 228x304) 적용
# - one-shot(train 샷) GT를 16-bit PNG(+jet)로 저장

import os, json, argparse, math, random
from typing import Dict, Any, Tuple, List, Optional

import numpy as np
from PIL import Image
from PIL.PngImagePlugin import PngInfo

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

# >>> NEW: tqdm 진행바
from tqdm.auto import tqdm

# ---- project-common modules ----
from config.loader import load_config
from config.schema import MCPropCfg
from utils.io_utils import save_jet, save_sparse_jet, unfold_neighbors
from utils.loss import l1l2_composite, scale_invariant_log_loss, lidar_consistency, rmse_mm
# from utils.DC.poisson import poisson_complete  # (dataset 내부 구현 사용)

# >>> 변경: 이미지/깊이 전용 데이터로더 <<<
from utils.NYU.dataset import (
    NYUH5OneShotDataset, build_oneshot_from_nyu, dump_oneshot_gt_images
)
from models.module import (
    TinyFeat, ResidualHead, CurvatureGen, KernelGate,
    normalize_affinity_list, AnchorHead
)
from models.affinity import EllipticAffinity, HCLApproxAffinity

# ==================== Shape helpers ====================
def _strip_extra_batch_dim(x: torch.Tensor) -> torch.Tensor:
    while x.dim() > 4 and x.size(1) == 1:
        x = x.squeeze(1)
    return x

def _as_chw4(x: torch.Tensor) -> torch.Tensor:
    x = _strip_extra_batch_dim(x)
    if x.dim() == 4 and x.size(1) in (1, 3):
        return x
    if x.dim() == 4 and x.size(-1) in (1, 3) and x.size(1) not in (1, 3):
        return x.permute(0, 3, 1, 2).contiguous()
    raise RuntimeError(f"Expected 4D (B,C,H,W) or (B,H,W,C), got {tuple(x.shape)}")

def _as_1ch4(x: torch.Tensor) -> torch.Tensor:
    x = _strip_extra_batch_dim(x)
    if x.dim() == 3:
        return x.unsqueeze(1)
    if x.dim() == 4 and x.size(1) == 1:
        return x
    if x.dim() == 4 and x.size(1) > 1:
        return x[:, :1, ...]
    raise RuntimeError(f"Expected 3D (B,H,W) or 4D with C>=1, got {tuple(x.shape)}")

def _resize_like(x: torch.Tensor, ref: torch.Tensor, mode: str = "bilinear") -> torch.Tensor:
    H, W = ref.shape[-2:]
    if x.shape[-2:] == (H, W): return x
    return F.interpolate(
        x, size=(H, W),
        mode=("bilinear" if mode=="bilinear" else "nearest"),
        align_corners=False if mode=="bilinear" else None
    )

# ==================== Model (same as before) ====================


class MCPropNet(nn.Module):
    """
    - step당 unfold 1회(kmax)
    - 작은 커널은 kmax 패치에서 채널 서브셋 추출
    - 센터 치환은 델타 보정(+ w_center*(D0 - Dt))
    - in-place 누적 제거(autograd 안전)
    """
    def __init__(self, cfg: MCPropCfg):
        super().__init__()
        self.cfg = cfg

        # Enc/heads
        in_ch = 3 + 1 + 1 + 1  # RGB, P/dmax, E_norm, ML
        self.enc = TinyFeat(in_ch, 64)
        self.res = ResidualHead(64, 64) if cfg.use_residual else None
        self.curv = CurvatureGen(64 + 1, cfg.kernels, cfg.kappa_min, cfg.kappa_max)
        self.gate = KernelGate(64, cfg.kernels)

        if cfg.geometry.lower().startswith("ellip"):
            self.aff_head = EllipticAffinity(64, cfg.kernels, c_aff=32, tau_min=0.03, tau_max=0.5)
        else:
            self.aff_head = HCLApproxAffinity(64, cfg.kernels)

        # Anchor
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

        # 커널/인덱스 준비
        self.kernels: Tuple[int, ...] = tuple(int(k) for k in cfg.kernels)
        assert all(k % 2 == 1 for k in self.kernels), "All kernels must be odd."
        self.kmax: int = int(max(self.kernels))
        self._center_idx: Dict[int,int] = {k: (k*k)//2 for k in self.kernels}

        def _subset_idx_cpu(k: int, kmax: int) -> torch.Tensor:
            r, rmax = k//2, kmax//2
            idx = []
            for dy in range(-r, r+1):
                row = (dy + rmax) * kmax
                for dx in range(-r, r+1):
                    idx.append(row + (dx + rmax))
            return torch.as_tensor(idx, dtype=torch.long)
        self._idx_subset_cpu: Dict[int, torch.Tensor] = {k: _subset_idx_cpu(k, self.kmax) for k in self.kernels}
        self._idx_device: Dict[str, Dict[int, torch.Tensor]] = {}

    def _alpha_map(self, feat: torch.Tensor, ML: torch.Tensor) -> torch.Tensor:
        if self.anchor_learnable:
            if self.anchor_mode == "scalar":
                a = torch.sigmoid(self.anchor_param)
                return a.view(1,1,1,1).expand_as(ML)
            else:
                return self.anchor_head(feat)
        else:
            return self.anchor_fixed.view(1,1,1,1).expand_as(ML)

    def _get_idx_on_device(self, device: torch.device) -> Dict[int, torch.Tensor]:
        key = str(device)
        cache = self._idx_device.get(key, None)
        if cache is not None: return cache
        cache = {k: v.to(device) for k, v in self._idx_subset_cpu.items()}
        self._idx_device[key] = cache
        return cache

    def forward(self, I, DL, ML, P, E_norm):
        cfg = self.cfg

        I  = _as_chw4(I.float())
        DL = _as_1ch4(DL.float())
        ML = _as_1ch4(ML.float())
        P  = _as_1ch4(P.float())
        E  = _as_1ch4(E_norm.float())

        # Dataset에서 E_norm∈[0,1] 보장이면 아래 분기 제거 가능
        if E.max() > 1.5:
            E = E / 255.0

        I01 = I / 255.0
        x_in = torch.cat([I01, P/cfg.dmax, E, ML], dim=1)

        feat = self.enc(x_in)
        D0 = (P + self.res(feat)).clamp(0, cfg.dmax) if self.res is not None else P

        kappa, scale, bias = self.curv(feat, E)
        if cfg.geometry.lower().startswith("ellip"):
            A_list_raw = self.aff_head(feat, kappa)
        else:
            A_list_raw = self.aff_head(feat, scale, bias)
        A_list = normalize_affinity_list(A_list_raw)
        sigma  = self.gate(feat)  # (B, K, H, W)

        alpha = self._alpha_map(feat, ML) if cfg.use_sparse else None
        idx_on_dev = self._get_idx_on_device(D0.device)

        Dt = D0.clone()
        for _ in range(cfg.steps):
            patches_max = unfold_neighbors(Dt, self.kmax)  # (B, kmax^2, H, W)

            # --- out-of-place 누적 (autograd-safe) ---
            acc = None
            for ki, k in enumerate(self.kernels):
                Ak = A_list[ki]                      # (B, k^2, H, W)
                idxs = idx_on_dev[k]                 # (k^2,)
                patches_k = patches_max.index_select(1, idxs)  # (B, k^2, H, W)

                Dk = (Ak * patches_k).sum(1, keepdim=True)     # (B,1,H,W)
                c  = self._center_idx[k]
                w_c = Ak[:, c:c+1, ...]                         # (B,1,H,W)
                Dk = Dk + w_c * (D0 - Dt)                       # 센터 델타 보정

                term = sigma[:, ki:ki+1, ...] * Dk              # (B,1,H,W)
                acc  = term if acc is None else (acc + term)    # out-of-place add

            mix = acc
            if cfg.use_sparse:
                Dt = (1.0 - alpha*ML) * mix + (alpha*ML) * DL
            else:
                Dt = mix
            Dt = Dt.clamp(0, cfg.dmax)

        return Dt, {"D0": D0, "sigma": sigma, "alpha": (alpha if alpha is not None else torch.tensor(0.0, device=Dt.device))}

# ==================== Train / Val ====================
def set_seed(seed=1):
    random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)

def move_to_device(batch, device): return [x.to(device, non_blocking=True) for x in batch]

@torch.no_grad()
def evaluate(model: MCPropNet, loader: DataLoader, device, mu_scaleinv=0.1, w_lidar=0.3):
    model.eval()
    tot = {"L":0.0,"L1L2":0.0,"SI":0.0,"LiDAR":0.0,"RMSEmm":0.0}
    n = 0

    # >>> tqdm: 검증 진행바
    pbar = tqdm(loader, desc="Val", dynamic_ncols=True, leave=False)
    for I, DL, ML, P, E, GT in pbar:
        I, DL, ML, P, E, GT = move_to_device([I, DL, ML, P, E, GT], device)
        pred, _ = model(I, DL, ML, P, E)

        if GT is not None and GT.numel() > 0 and GT.shape[-2:] != pred.shape[-2:]:
            GT = F.interpolate(GT, size=pred.shape[-2:], mode="nearest")

        L_l1l2  = l1l2_composite(pred, GT)
        L_si    = scale_invariant_log_loss(E + 1e-3, GT)
        L_lidar = lidar_consistency(pred, DL, ML) if model.cfg.use_sparse else pred.new_tensor(0.0)
        L       = L_l1l2 + mu_scaleinv * L_si + w_lidar * L_lidar

        rm      = rmse_mm(pred, GT)
        tot["L"] += float(L); tot["L1L2"] += float(L_l1l2); tot["SI"] += float(L_si); tot["LiDAR"] += float(L_lidar)
        tot["RMSEmm"] += (0.0 if math.isnan(rm) else rm); n += 1

        # 진행 중 러닝 평균을 postfix로 표시
        avg_now = {k: v/max(1,n) for k,v in tot.items()}
        pbar.set_postfix(L=f"{avg_now['L']:.4f}", RMSEmm=f"{avg_now['RMSEmm']:.1f}")

    avg = {k: v/max(1,n) for k,v in tot.items()}
    return avg


def train_oneshot_nyu(cfg: Dict[str,Any]):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    ks = cfg.get("kshot", {})
    set_seed(int(ks.get("seed", 1)))

    # ---- splits ----
    splits = build_oneshot_from_nyu(cfg)

    # --- 원샷 GT 저장 (훈련 시작 전에 한 번) ---
    dmax = float(cfg["model"]["dmax"])
    dump_oneshot_gt_images(splits, cfg, dmax)

    # Scales & Poisson & crop
    nyu = cfg["nyu"]
    mono_scale   = nyu.get("mono_scale", None)      # (viz 사용이면 미사용)
    sparse_scale = nyu.get("sparse_scale", None)
    fix_sparse   = bool(nyu.get("fix_sparse", True))
    crop_h = nyu.get("crop_h", 228)
    crop_w = nyu.get("crop_w", 304)
    crop_hw = (crop_h, crop_w) if (crop_h is not None and crop_w is not None) else None
    est_cmap = nyu.get("est_cmap", "inferno")

    pconf   = cfg.get("poisson", {})
    p_lam   = float(pconf.get("lambda", 800.0))
    p_iters = int(pconf.get("iters", 300))
    p_hard  = bool(pconf.get("hard", False))

    train_ds = NYUH5OneShotDataset(
        splits["train"], dmax=dmax,
        mono_scale=mono_scale, sparse_scale=sparse_scale, fix_sparse=fix_sparse,
        poisson_lam=p_lam, poisson_iters=p_iters, poisson_hard=p_hard,
        crop_hw=crop_hw, est_cmap=est_cmap
    )
    val_ds   = NYUH5OneShotDataset(
        splits["val"], dmax=dmax,
        mono_scale=mono_scale, sparse_scale=sparse_scale, fix_sparse=True,
        poisson_lam=p_lam, poisson_iters=p_iters, poisson_hard=p_hard,
        crop_hw=crop_hw, est_cmap=est_cmap
    )

    bs   = int(ks.get("batch_size", 1))
    nw   = int(ks.get("num_workers", 4))
    shuf = bool(ks.get("shuffle", True))
    epochs = int(ks.get("epochs", 60))
    lr     = float(ks.get("lr", 1e-3))
    save_dir = ks.get("save_dir", "runs_nyu_1shot"); os.makedirs(save_dir, exist_ok=True)
    tag      = ks.get("tag", "mcprop_nyu_1shot")
    preview_every = int(ks.get("preview_every", 1))

    train_loader = DataLoader(train_ds, batch_size=bs, shuffle=shuf, num_workers=nw, pin_memory=True)
    val_loader   = DataLoader(val_ds,   batch_size=bs, shuffle=False, num_workers=nw, pin_memory=True)

    # ---- model/optim ----
    mcfg = cfg["model"]
    model = MCPropNet(MCPropCfg(
        dmax=mcfg["dmax"], steps=mcfg["steps"], kernels=tuple(mcfg["kernels"]),
        use_residual=mcfg["use_residual"], use_sparse=mcfg["use_sparse"],
        anchor_alpha=mcfg["anchor_alpha"], anchor_learnable=mcfg["anchor_learnable"],
        anchor_mode=mcfg["anchor_mode"], kappa_min=mcfg["kappa_min"], kappa_max=mcfg["kappa_max"],
        geometry=mcfg["geometry"]
    )).to(device)

    mu_scaleinv = float(cfg.get("loss", {}).get("mu_scaleinv", 0.1))
    w_lidar     = float(cfg.get("loss", {}).get("w_lidar", 0.3))
    w_anchor    = float(cfg.get("loss", {}).get("w_anchor_reg", 0.0))

    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-2)

    best_rmse = float("inf")
    best_path = os.path.join(save_dir, f"{tag}_best.pt")

    # ---- loop ----
    for ep in range(1, epochs+1):
        model.train()
        tot = {"L":0.0,"L1L2":0.0,"SI":0.0,"LiDAR":0.0,"Anchor":0.0,"RMSEmm":0.0}
        n = 0

        # >>> tqdm: 학습 진행바
        pbar = tqdm(train_loader, desc=f"Train Ep {ep:03d}", dynamic_ncols=True, leave=False)
        for I, DL, ML, P, E, GT in pbar:
            I, DL, ML, P, E, GT = move_to_device([I, DL, ML, P, E, GT], device)
            pred, aux = model(I, DL, ML, P, E)

            if GT is not None and GT.numel() > 0 and GT.shape[-2:] != pred.shape[-2:]:
                GT = F.interpolate(GT, size=pred.shape[-2:], mode="nearest")

            L_l1l2  = l1l2_composite(pred, GT)
            L_si    = scale_invariant_log_loss(E + 1e-3, GT)
            L_lidar = lidar_consistency(pred, DL, ML) if model.cfg.use_sparse else pred.new_tensor(0.0)

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

            # 러닝 평균을 진행바에 표시
            avg_now = {k: v/max(1,n) for k,v in tot.items()}
            pbar.set_postfix(L=f"{avg_now['L']:.4f}",
                             L1L2=f"{avg_now['L1L2']:.4f}",
                             SI=f"{avg_now['SI']:.4f}",
                             RMSEmm=f"{avg_now['RMSEmm']:.1f}")

        avg_tr = {k: v/max(1,n) for k,v in tot.items()}
        avg_va = evaluate(model, val_loader, device, mu_scaleinv, w_lidar)

        print(f"[{tag}] Ep {ep:03d} | "
              f"Train L={avg_tr['L']:.4f} L1L2={avg_tr['L1L2']:.4f} SI={avg_tr['SI']:.4f} "
              f"Lidar={avg_tr['LiDAR']:.4f} Anchor={avg_tr['Anchor']:.4f} RMSE={avg_tr['RMSEmm']:.1f} | "
              f"Val L={avg_va['L']:.4f} L1L2={avg_va['L1L2']:.4f} SI={avg_va['SI']:.4f} "
              f"Lidar={avg_va['LiDAR']:.4f} RMSE={avg_va['RMSEmm']:.1f}")

        # preview
        if (ep % max(1, preview_every) == 0) or (ep == 1):
            with torch.no_grad():
                for I, DL, ML, P, E, GT in val_loader:
                    I, DL, ML, P, E, GT = move_to_device([I, DL, ML, P, E, GT], device)
                    pred, aux = model(I, DL, ML, P, E)
                    save_jet(os.path.join(save_dir, f"{tag}_ep{ep:03d}_pred_jet.png"), pred[:1], dmax=model.cfg.dmax)
                    save_jet(os.path.join(save_dir, f"{tag}_ep{ep:03d}_d0_jet.png"),   aux["D0"][:1], dmax=model.cfg.dmax)
                    save_sparse_jet(os.path.join(save_dir, f"{tag}_ep{ep:03d}_sparse_jet.png"), DL[:1], ML[:1], dmax=model.cfg.dmax)
                    break

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


# ==================== Entry ====================
def main():
    parser = argparse.ArgumentParser("NYUv2 One-shot for MCProp (Images + Viz Estimation)")
    parser.add_argument("--config", type=str, default="", help="JSON/YAML config path (optional)")
    args = parser.parse_args()

    cfg = load_config(args.config)
    train_oneshot_nyu(cfg)

if __name__ == "__main__":
    main()
