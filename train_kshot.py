# train_kshot.py
# python3 train_kshot.py --config config/kshot/kshot.json
# torchrun --nproc_per_node=2 train_kshot.py --config config/kshot/kshot.json --ddp

import os, json, argparse, math, random
from typing import Dict, Any, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

import torch.distributed as dist

from config.loader import load_config
from config.schema import MCPropCfg, LossW  # TrainCfg는 1-shot용이므로 여기선 쓰지 않음
from utils.dataset import KShotDataset, build_kshot_from_paths
from utils.io_utils import save_jet, save_sparse_jet, unfold_neighbors
from utils.loss import l1l2_composite, scale_invariant_log_loss, lidar_consistency, rmse_mm

from models.module import (
    TinyFeat, ResidualHead, CurvatureGen, KernelGate,
    normalize_affinity_list, AnchorHead
)
from models.affinity import EllipticAffinity, HCLApproxAffinity

# -------------------- DDP helpers --------------------
def is_dist_avail_and_initialized():
    return dist.is_available() and dist.is_initialized()

def get_rank():
    return dist.get_rank() if is_dist_avail_and_initialized() else 0

def get_world_size():
    return dist.get_world_size() if is_dist_avail_and_initialized() else 1

def is_main_process():
    return get_rank() == 0

def print0(*args, **kwargs):
    if is_main_process():
        print(*args, **kwargs)

def setup_ddp(backend="nccl"):
    """
    torchrun 으로 실행 시 환경변수(LOCAL_RANK, RANK, WORLD_SIZE)를 사용.
    """
    if is_dist_avail_and_initialized():
        return int(os.environ.get("LOCAL_RANK", 0))
    if "LOCAL_RANK" not in os.environ:
        # 단일 GPU 또는 DP 모드일 수 있음
        return 0
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    dist.init_process_group(backend=backend, init_method="env://")
    return local_rank

def cleanup_ddp():
    if is_dist_avail_and_initialized():
        dist.barrier()
        dist.destroy_process_group()

# -------------------- shape helpers --------------------
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
    raise RuntimeError(f"Expected 4D tensor (B,C,H,W) or (B,H,W,C), got shape={tuple(x.shape)}")

def _as_1ch4(x: torch.Tensor) -> torch.Tensor:
    x = _strip_extra_batch_dim(x)
    if x.dim() == 3:
        return x.unsqueeze(1)
    if x.dim() == 4 and x.size(1) == 1:
        return x
    if x.dim() == 4 and x.size(1) > 1:
        return x[:, :1, ...]
    raise RuntimeError(f"Expected 3D (B,H,W) or 4D with C>=1, got shape={tuple(x.shape)}")

def _resize_like(x: torch.Tensor, ref: torch.Tensor, mode: str = "bilinear") -> torch.Tensor:
    H, W = ref.shape[-2:]
    if x.shape[-2:] == (H, W):
        return x
    if mode == "bilinear":
        return F.interpolate(x, size=(H, W), mode="bilinear", align_corners=False)
    else:
        return F.interpolate(x, size=(H, W), mode="nearest")

# -------------------- Model --------------------
class MCPropNet(nn.Module):
    def __init__(self, cfg: MCPropCfg):
        super().__init__()
        self.cfg = cfg
        in_ch = 3+1+1+1
        self.enc = TinyFeat(in_ch, 64)
        self.res = ResidualHead(64, 64) if cfg.use_residual else None
        self.curv = CurvatureGen(64+1, cfg.kernels, cfg.kappa_min, cfg.kappa_max)
        self.gate = KernelGate(64, cfg.kernels)

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

        I  = _as_chw4(I.float())
        DL = _as_1ch4(DL.float())
        ML = _as_1ch4(ML.float())
        P  = _as_1ch4(P.float())
        E  = _as_1ch4(E_norm.float())

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
        sigma  = self.gate(feat)

        alpha = self._alpha_map(feat, ML)

        Dt = D0.clone()
        for _ in range(cfg.steps):
            mix = torch.zeros_like(Dt)
            for idx, k in enumerate(cfg.kernels):
                Ak = A_list[idx]
                kk = k*k
                patches = unfold_neighbors(Dt, k)
                center = kk // 2
                patches_center = patches.clone()
                patches_center[:, center:center+1, :, :] = D0
                Dk = (Ak * patches_center).sum(1, keepdim=True)
                mix = mix + sigma[:, idx:idx+1] * Dk

            if cfg.use_sparse:
                Dt = (1.0 - alpha*ML) * mix + (alpha*ML) * DL
            else:
                Dt = mix
            Dt = Dt.clamp(0, cfg.dmax)

        return Dt, {"D0": D0, "sigma": sigma, "alpha": alpha}

# -------------------- Train / Val --------------------
def set_seed(seed=1):
    random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)

def move_to_device(batch, device): return [x.to(device, non_blocking=True) for x in batch]

@torch.no_grad()
def evaluate(model, loader, device, loss_w: LossW = None, distributed: bool=False):
    model.eval()
    # 총합(샘플 수로 가중 평균)
    sum_L = sum_L1 = sum_SI = sum_LiDAR = 0.0
    sum_RMSE = 0.0
    n_samples = 0

    for I, DL, ML, P, E, GT in loader:
        I, DL, ML, P, E, GT = move_to_device([I, DL, ML, P, E, GT], device)
        B = I.size(0)
        pred, aux = model(I, DL, ML, P, E)

        if GT is not None and GT.numel() > 0 and GT.shape[-2:] != pred.shape[-2:]:
            GT = F.interpolate(GT, size=pred.shape[-2:], mode="nearest")

        L_l1l2  = l1l2_composite(pred, GT)
        L_si    = scale_invariant_log_loss(E + 1e-3, GT)
        L_lidar = lidar_consistency(pred, DL, ML) if model.cfg.use_sparse else pred.new_tensor(0.0)
        L       = L_l1l2 + 0.1*L_si + 0.3*L_lidar

        # per-sample RMSE 평균
        rms = 0.0; cnt=0
        for b in range(B):
            rmb = rmse_mm(pred[b:b+1], GT[b:b+1])
            if not math.isnan(rmb):
                rms += rmb; cnt += 1

        sum_L      += float(L) * B
        sum_L1     += float(L_l1l2) * B
        sum_SI     += float(L_si) * B
        sum_LiDAR  += float(L_lidar) * B
        sum_RMSE   += (rms if cnt>0 else 0.0)
        n_samples  += B

    # DDP reduce
    if distributed and is_dist_avail_and_initialized():
        t = torch.tensor([sum_L, sum_L1, sum_SI, sum_LiDAR, sum_RMSE, n_samples], device=device, dtype=torch.float64)
        dist.all_reduce(t, op=dist.ReduceOp.SUM)
        sum_L, sum_L1, sum_SI, sum_LiDAR, sum_RMSE, n_samples = [float(x) for x in t.tolist()]

    n = max(1, int(n_samples))
    return {
        "L":       sum_L/n,
        "L1L2":    sum_L1/n,
        "SI":      sum_SI/n,
        "LiDAR":   sum_LiDAR/n,
        "RMSEmm":  sum_RMSE/max(1, n_samples)  # per-sample 평균
    }

def _save_epoch_viz(save_dir: str, tag: str, ep: int, split_name: str,
                    I, DL, ML, P, E, pred, aux, dmax: float, max_n: int = 4, viz_dir: str = "viz"):
    out_root = os.path.join(save_dir, viz_dir, f"ep{ep:03d}", split_name)
    os.makedirs(out_root, exist_ok=True)
    B = I.shape[0]
    N = min(B, max_n)
    for j in range(N):
        save_jet(os.path.join(out_root, f"pred_{j:03d}.png"), pred[j:j+1], dmax=dmax)
        save_jet(os.path.join(out_root, f"d0_{j:03d}.png"),   aux["D0"][j:j+1], dmax=dmax)
        save_sparse_jet(os.path.join(out_root, f"sparse_{j:03d}.png"), DL[j:j+1], ML[j:j+1], dmax=dmax)

def train_kshot(cfg: Dict[str,Any], use_ddp: bool=False, use_dp: bool=False, local_rank: int=0):
    # device
    device = torch.device(f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu")

    ks = cfg.get("kshot", {})
    set_seed(int(ks.get("seed", 1)))

    # ---- 샘플 빌드(경로 기반) ----
    splits = build_kshot_from_paths(cfg)
    train_ds = KShotDataset(splits["train"])
    val_ds   = KShotDataset(splits["val"])

    bs   = int(ks.get("batch_size", 4))
    nw   = int(ks.get("num_workers", 4))
    shuf = bool(ks.get("shuffle", True))
    epochs = int(ks.get("epochs", 60))
    lr     = float(ks.get("lr", 1e-3))
    save_dir = ks.get("save_dir", "runs_kshot"); os.makedirs(save_dir, exist_ok=True) if is_main_process() else None
    tag      = ks.get("tag", "mcprop_kshot")
    preview_every = int(ks.get("preview_every", 5))

    viz_every = int(ks.get("viz_every", 10))
    viz_dir   = ks.get("viz_dir", "viz")
    viz_max   = int(ks.get("viz_max_per_split", 4))

    # ---- Sampler (DDP) ----
    if use_ddp and get_world_size() > 1:
        train_sampler = torch.utils.data.distributed.DistributedSampler(train_ds, shuffle=shuf, drop_last=False)
        val_sampler   = torch.utils.data.distributed.DistributedSampler(val_ds,   shuffle=False, drop_last=False)
        shuffle_flag  = False   # sampler가 셔플 담당
    else:
        train_sampler = None
        val_sampler   = None
        shuffle_flag  = shuf

    train_loader = DataLoader(train_ds, batch_size=bs, shuffle=shuffle_flag,
                              num_workers=nw, pin_memory=True, sampler=train_sampler)
    val_loader   = DataLoader(val_ds,   batch_size=bs, shuffle=False,
                              num_workers=nw, pin_memory=True, sampler=val_sampler)

    # ---- 모델/옵티마 ----
    mcfg = cfg["model"]
    base_model = MCPropNet(MCPropCfg(
        dmax=mcfg["dmax"], steps=mcfg["steps"], kernels=tuple(mcfg["kernels"]),
        use_residual=mcfg["use_residual"], use_sparse=mcfg["use_sparse"],
        anchor_alpha=mcfg["anchor_alpha"], anchor_learnable=mcfg["anchor_learnable"],
        anchor_mode=mcfg["anchor_mode"], kappa_min=mcfg["kappa_min"], kappa_max=mcfg["kappa_max"],
        geometry=mcfg["geometry"]
    )).to(device)

    # wrap (DDP/DP)
    if use_ddp and get_world_size() > 1:
        model = nn.parallel.DistributedDataParallel(base_model, device_ids=[local_rank], output_device=local_rank)
    elif use_dp and torch.cuda.device_count() > 1:
        print0(f"[INFO] Using DataParallel with {torch.cuda.device_count()} GPUs")
        model = nn.DataParallel(base_model)
    else:
        model = base_model

    loss_w = LossW(
        mu_scaleinv=cfg["loss"].get("mu_scaleinv", 0.1),
        w_lidar=cfg["loss"].get("w_lidar", 0.3),
        w_anchor_reg=cfg["loss"].get("w_anchor_reg", 0.0)
    )
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-2)

    best_rmse = float("inf")
    best_path = os.path.join(save_dir, f"{tag}_best.pt")

    # ---- 학습 루프 ----
    for ep in range(1, epochs+1):
        if train_sampler is not None:
            train_sampler.set_epoch(ep)

        model.train()
        tot = {"L":0.0,"L1L2":0.0,"SI":0.0,"LiDAR":0.0,"Anchor":0.0,"RMSEmm":0.0}; n=0

        for I, DL, ML, P, E, GT in train_loader:
            I, DL, ML, P, E, GT = move_to_device([I, DL, ML, P, E, GT], device)
            pred, aux = model(I, DL, ML, P, E)

            if GT is not None and GT.numel() > 0 and GT.shape[-2:] != pred.shape[-2:]:
                GT = F.interpolate(GT, size=pred.shape[-2:], mode="nearest")

            L_l1l2  = l1l2_composite(pred, GT)
            L_si = scale_invariant_log_loss(pred.clamp_min(1e-3), GT)
            L_lidar = lidar_consistency(pred, DL, ML) if base_model.cfg.use_sparse else pred.new_tensor(0.0)

            L_anchor = pred.new_tensor(0.0)
            if loss_w.w_anchor_reg > 0.0 and base_model.cfg.use_sparse:
                alpha = aux["alpha"].detach() if not base_model.anchor_learnable else aux["alpha"]
                L_anchor = ((1.0 - alpha) * ML).mean() * loss_w.w_anchor_reg

            L = L_l1l2 + loss_w.mu_scaleinv * L_si + loss_w.w_lidar * L_lidar + L_anchor

            opt.zero_grad(set_to_none=True); L.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0); opt.step()

            rm = rmse_mm(pred, GT)
            tot["L"]+=float(L); tot["L1L2"]+=float(L_l1l2); tot["SI"]+=float(L_si)
            tot["LiDAR"]+=float(L_lidar); tot["Anchor"]+=float(L_anchor); tot["RMSEmm"]+= (0.0 if math.isnan(rm) else rm)
            n+=1

        # DDP에서 train 로그 평균은 대략적인 보고만(정확한 합산은 생략)
        avg = {k: v/max(1,n) for k,v in tot.items()}

        logs_val = evaluate(base_model if isinstance(model, (nn.DataParallel, nn.parallel.DistributedDataParallel)) else model,
                            val_loader, device, loss_w,
                            distributed=(use_ddp and get_world_size()>1))

        print0(f"[{tag}] Ep {ep:03d} | "
              f"Train L={avg['L']:.4f} L1L2={avg['L1L2']:.4f} SI={avg['SI']:.4f} "
              f"Lidar={avg['LiDAR']:.4f} Anchor={avg['Anchor']:.4f} RMSE={avg['RMSEmm']:.1f} | "
              f"Val L={logs_val['L']:.4f} L1L2={logs_val['L1L2']:.4f} SI={logs_val['SI']:.4f} "
              f"Lidar={logs_val['LiDAR']:.4f} RMSE={logs_val['RMSEmm']:.1f}")

        # 기존 미리보기(유지, rank0만)
        if is_main_process() and ((ep % max(1, preview_every) == 0) or (ep == 1)):
            with torch.no_grad():
                for I, DL, ML, P, E, GT in val_loader:
                    I, DL, ML, P, E, GT = move_to_device([I, DL, ML, P, E, GT], device)
                    pred, aux = model(I, DL, ML, P, E)
                    save_jet(os.path.join(save_dir, f"{tag}_ep{ep:03d}_pred_jet.png"), pred[:1], dmax=base_model.cfg.dmax)
                    save_jet(os.path.join(save_dir, f"{tag}_ep{ep:03d}_d0_jet.png"),   aux["D0"][:1], dmax=base_model.cfg.dmax)
                    save_sparse_jet(os.path.join(save_dir, f"{tag}_ep{ep:03d}_sparse_jet.png"), DL[:1], ML[:1], dmax=base_model.cfg.dmax)
                    break

        # 10 epoch 단위 viz (rank0만)
        if is_main_process() and (viz_every > 0 and (ep % viz_every == 0)):
            with torch.no_grad():
                (model.eval() if hasattr(model, "eval") else None)
                for I, DL, ML, P, E, GT in train_loader:
                    I, DL, ML, P, E, GT = move_to_device([I, DL, ML, P, E, GT], device)
                    pred, aux = model(I, DL, ML, P, E)
                    _save_epoch_viz(save_dir, tag, ep, "train", I, DL, ML, P, E, pred, aux, base_model.cfg.dmax, max_n=viz_max, viz_dir=viz_dir)
                    break
                for I, DL, ML, P, E, GT in val_loader:
                    I, DL, ML, P, E, GT = move_to_device([I, DL, ML, P, E, GT], device)
                    pred, aux = model(I, DL, ML, P, E)
                    _save_epoch_viz(save_dir, tag, ep, "val", I, DL, ML, P, E, pred, aux, base_model.cfg.dmax, max_n=viz_max, viz_dir=viz_dir)
                    break
                model.train()

        # 베스트(Val RMSE, rank0만)
        if is_main_process() and math.isfinite(logs_val["RMSEmm"]) and logs_val["RMSEmm"] < best_rmse:
            best_rmse = logs_val["RMSEmm"]
            to_save = base_model if isinstance(model, (nn.DataParallel, nn.parallel.DistributedDataParallel)) else model
            torch.save({"state_dict": to_save.state_dict(),
                        "cfg": to_save.cfg.__dict__,
                        "val_rmse_mm": best_rmse}, best_path)
            print0(f"  -> best Val RMSE {best_rmse:.1f} mm (saved: {best_path})")

    print0(f"[{tag}] Finished. Best Val RMSE(mm)={best_rmse:.1f}")
    return best_rmse

# -------------------- Entry --------------------
def main():
    parser = argparse.ArgumentParser("K-shot training for MCProp (path-based)")
    parser.add_argument("--config", type=str, default="", help="JSON/YAML config (optional)")
    parser.add_argument("--ddp", action="store_true", help="Use DistributedDataParallel (recommended)")
    parser.add_argument("--data-parallel", action="store_true", help="Use DataParallel (fallback)")
    args = parser.parse_args()

    # DDP setup (if requested)
    local_rank = 0
    if args.ddp:
        local_rank = setup_ddp(backend="nccl")
        print0(f"[DDP] rank={get_rank()} world={get_world_size()} local_rank={local_rank}")

    cfg = load_config(args.config)   # DEFAULT_CFG + user override

    try:
        train_kshot(cfg, use_ddp=args.ddp, use_dp=args.data_parallel, local_rank=local_rank)
    finally:
        if args.ddp:
            cleanup_ddp()

if __name__ == "__main__":
    main()
