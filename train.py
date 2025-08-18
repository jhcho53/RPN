# train_oneshot.py
import os, json, argparse
import torch

from config.loader import load_config, to_dataclasses
from utils.io_utils import (
    load_rgb, load_depth16_mm_as_m, load_pseudo_auto,
    load_estimation_8bit_norm, save_jet, save_sparse_jet, unfold_neighbors
)
from utils.loss import l1l2_composite, scale_invariant_log_loss, lidar_consistency, rmse_mm
from models.module import TinyFeat, ResidualHead, CurvatureGen, KernelGate, normalize_affinity_list, AnchorHead
from models.affinity import EllipticAffinity, HCLApproxAffinity
from config.loader import load_config, to_dataclasses
from config.schema import MCPropCfg, TrainCfg, LossW, OneShotPaths
from config.defaults import DEFAULT_CFG
import torch.nn as nn

# -------------------- Full model --------------------
class MCPropNet(nn.Module):
    def __init__(self, cfg: MCPropCfg):
        super().__init__()
        self.cfg = cfg
        in_ch = 3+1+1+1
        self.enc = TinyFeat(in_ch, 64)
        self.res = ResidualHead(64, 64) if cfg.use_residual else None
        self.curv = CurvatureGen(64+1, cfg.kernels, cfg.kappa_min, cfg.kappa_max)
        self.gate = KernelGate(64, cfg.kernels)

        # geometry-specific affinity
        if cfg.geometry.lower().startswith("ellip"):
            self.aff_head = EllipticAffinity(64, cfg.kernels, c_aff=32, tau_min=0.03, tau_max=0.5)
        else:
            self.aff_head = HCLApproxAffinity(64, cfg.kernels)

        # ---- Learnable anchor ----
        self.anchor_learnable = bool(cfg.anchor_learnable)
        self.anchor_mode = cfg.anchor_mode.lower()
        self.anchor_init = float(cfg.anchor_alpha)

        if self.anchor_learnable:
            if self.anchor_mode == "scalar":
                # single global parameter
                self.anchor_param = nn.Parameter(torch.tensor(self.anchor_init, dtype=torch.float32))
                self.anchor_head = None
            elif self.anchor_mode == "map":
                self.anchor_param = None
                self.anchor_head = AnchorHead(64)
            else:
                raise ValueError(f"Unknown anchor_mode: {cfg.anchor_mode}")
        else:
            # fixed buffer for reproducibility
            self.register_buffer("anchor_fixed", torch.tensor(self.anchor_init, dtype=torch.float32))
            self.anchor_param = None
            self.anchor_head = None

    def _alpha_map(self, feat, ML):
        """return α map in [0,1], shape (B,1,H,W)"""
        if self.anchor_learnable:
            if self.anchor_mode == "scalar":
                a = torch.sigmoid(self.anchor_param)  # (0,1)
                return a.view(1,1,1,1).expand_as(ML)
            else:  # "map"
                return self.anchor_head(feat)
        else:
            return self.anchor_fixed.view(1,1,1,1).expand_as(ML)

    def forward(self, I, DL, ML, P, E_norm):
        cfg = self.cfg
        I01 = I / 255.0
        x_in = torch.cat([I01, P/cfg.dmax, E_norm, ML], dim=1)
        feat = self.enc(x_in)

        D0 = (P + self.res(feat)).clamp(0, cfg.dmax) if self.res is not None else P

        kappa, scale, bias = self.curv(feat, E_norm)
        if cfg.geometry.lower().startswith("ellip"):
            A_list_raw = self.aff_head(feat, kappa)
        else:
            A_list_raw = self.aff_head(feat, scale, bias)
        A_list = normalize_affinity_list(A_list_raw)
        sigma  = self.gate(feat)

        alpha = self._alpha_map(feat, ML)  # (B,1,H,W) in [0,1]

        Dt = D0.clone()
        for _ in range(cfg.steps):
            mix_k = []
            for idx, k in enumerate(cfg.kernels):
                Ak = A_list[idx]
                kk = k*k
                patches = unfold_neighbors(Dt, k)
                center = kk // 2
                patches_center = patches.clone()
                patches_center[:, center:center+1, :, :] = D0
                D_next_k = (Ak * patches_center).sum(1, keepdim=True)
                mix_k.append(D_next_k)

            Dmix = torch.zeros_like(Dt)
            for idx, Dk in enumerate(mix_k):
                Dmix = Dmix + (sigma[:, idx:idx+1] * Dk)

            if self.cfg.use_sparse:
                Dt = (1.0 - alpha*ML) * Dmix + (alpha*ML) * DL
            else:
                Dt = Dmix
            Dt = Dt.clamp(0, cfg.dmax)

        aux = {"D0": D0, "sigma": sigma, "alpha": alpha}
        return Dt, aux

def load_oneshot(paths: OneShotPaths, device="cuda"):
    I  = load_rgb(paths.rgb).to(device)
    DL = load_depth16_mm_as_m(paths.sparse).to(device)
    ML = (DL>0).float()
    P  = load_pseudo_auto(paths.pseudo).to(device)
    E  = load_estimation_8bit_norm(paths.estim).to(device)
    GT = load_depth16_mm_as_m(paths.gt).to(device) if (paths.gt and os.path.isfile(paths.gt)) else None
    return I, DL, ML, P, E, GT

def train_oneshot(model: MCPropNet, I, DL, ML, P, E_norm, GT, tcfg: TrainCfg, w: LossW) -> float:
    os.makedirs(tcfg.save_dir, exist_ok=True)
    opt = torch.optim.AdamW(model.parameters(), lr=tcfg.lr, weight_decay=1e-2)
    best = float('inf')
    best_path = os.path.join(tcfg.save_dir, f"{tcfg.tag}_best.pt")

    save_sparse_jet(os.path.join(tcfg.save_dir, f"{tcfg.tag}_sparse_jet.png"), DL, ML, dmin=0.5, dmax=model.cfg.dmax)

    for ep in range(1, tcfg.epochs+1):
        model.train()
        pred, aux = model(I, DL, ML, P, E_norm)

        L_l1l2  = l1l2_composite(pred, GT)
        L_si    = scale_invariant_log_loss(E_norm + 1e-3, GT)
        L_lidar = lidar_consistency(pred, DL, ML) if model.cfg.use_sparse else pred.new_tensor(0.0)

        # NEW: anchor regularization (encourage α→1 at LiDAR locations)
        L_anchor = pred.new_tensor(0.0)
        if w.w_anchor_reg > 0.0 and model.cfg.use_sparse:
            alpha = aux["alpha"].detach() if not model.anchor_learnable else aux["alpha"]
            # toward-one at LiDAR pixels:
            L_anchor = ((1.0 - alpha) * ML).mean() * w.w_anchor_reg

        L = L_l1l2 + w.mu_scaleinv * L_si + w.w_lidar * L_lidar + L_anchor

        opt.zero_grad(set_to_none=True); L.backward(); opt.step()

        with torch.no_grad():
            rm = rmse_mm(pred, GT)
        print(f"[{tcfg.tag}] ep {ep:03d} | L={float(L):.4f}  L1L2={float(L_l1l2):.4f}  "
              f"SI={float(L_si):.4f}  LiDAR={float(L_lidar):.4f}  Anchor={float(L_anchor):.4f}  RMSE(mm)={rm:.1f}")

        if ep % max(1, tcfg.epochs//10) == 0 or ep==1:
            save_jet(os.path.join(tcfg.save_dir, f"{tcfg.tag}_ep{ep:03d}_pred_jet.png"), pred, dmax=model.cfg.dmax)
            save_jet(os.path.join(tcfg.save_dir, f"{tcfg.tag}_ep{ep:03d}_d0_jet.png"),   aux["D0"], dmax=model.cfg.dmax)

        if torch.isfinite(torch.tensor(rm)) and rm < best:
            best = rm
            torch.save({"state_dict": model.state_dict(),
                        "cfg": model.cfg.__dict__}, best_path)
            print(f"  -> best RMSE {best:.1f} mm  (saved: {best_path})")

    return best

def main():
    import json
    parser = argparse.ArgumentParser("1-shot multi-geometry propagation (with learnable anchor)")
    parser.add_argument("--config", type=str, default="", help="JSON/YAML config path (optional)")
    args = parser.parse_args()

    cfg = load_config(args.config)
    paths_dc, train_dc, model_dc, loss_dc, exps = to_dataclasses(cfg)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    I, DL, ML, P, E_norm, GT = load_oneshot(paths_dc, device=device)
    os.makedirs(train_dc.save_dir, exist_ok=True)

    # ablation runs
    if exps:
        for exp in exps:
            m = dict(cfg["model"]); m["geometry"] = exp.get("geometry", m["geometry"])
            model = MCPropNet(MCPropCfg(
                dmax=m["dmax"], steps=m["steps"], kernels=tuple(m["kernels"]),
                use_residual=m["use_residual"], use_sparse=m["use_sparse"],
                anchor_alpha=m["anchor_alpha"], anchor_learnable=m["anchor_learnable"],
                anchor_mode=m["anchor_mode"], kappa_min=m["kappa_min"], kappa_max=m["kappa_max"],
                geometry=m["geometry"]
            )).to(device)

            tcfg = TrainCfg(**cfg["train"]); tcfg.tag = exp.get("tag", f"mcprop_1shot_{m['geometry']}")
            best = train_oneshot(model, I, DL, ML, P, E_norm, GT, tcfg, loss_dc)
            print(f"[{tcfg.tag}] Best RMSE(mm)={best:.1f}")
    else:
        model = MCPropNet(model_dc).to(device)
        best = train_oneshot(model, I, DL, ML, P, E_norm, GT, train_dc, loss_dc)
        print(f"Done. Best RMSE(mm)={best:.1f}")

if __name__ == "__main__":
    main()