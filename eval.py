#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os, sys, argparse, glob, csv, json
from dataclasses import dataclass
from typing import Tuple, List, Dict, Any, Optional

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm

# ===== 프로젝트 모듈 =====
try:
    from utils.io_utils import (
        load_depth16_mm_as_m, load_estimation_8bit_norm, load_pseudo_auto, load_rgb,
        save_jet, save_sparse_jet
    )
    from models.module import TinyFeat, ResidualHead, CurvatureGen, KernelGate, normalize_affinity_list
    from models.affinity import EllipticAffinity, HCLApproxAffinity
except Exception as e:
    print("[ERROR] 프로젝트 모듈을 찾을 수 없습니다. PYTHONPATH 또는 모듈 배치를 확인하세요.")
    print("예: export PYTHONPATH=$PYTHONPATH:$(pwd)")
    raise

# -------------------- 유틸: dict merge & overrides --------------------

def _deep_update(dst: dict, src: dict) -> dict:
    """중첩 dict 재귀 병합 (src가 dst를 덮어씀)."""
    for k, v in src.items():
        if isinstance(v, dict) and isinstance(dst.get(k), dict):
            _deep_update(dst[k], v)
        else:
            dst[k] = v
    return dst

def _parse_value(val: str):
    """문자열을 bool/int/float/그대로 순으로 파싱."""
    if val.lower() in ("true","false"):
        return val.lower() == "true"
    try:
        if "." in val:
            return float(val)
        return int(val)
    except ValueError:
        return val

def _apply_overrides(cfg: dict, kv_list: List[str]) -> dict:
    """
    --set a.b.c=1 형태를 cfg에 적용.
    """
    for kv in kv_list:
        if "=" not in kv:
            print(f"[WARN] ignore override without '=': {kv}")
            continue
        key, val = kv.split("=", 1)
        val = _parse_value(val)
        # 점 표기 접근
        node = cfg
        parts = key.split(".")
        for p in parts[:-1]:
            if p not in node or not isinstance(node[p], dict):
                node[p] = {}
            node = node[p]
        node[parts[-1]] = val
    return cfg

# -------------------- 모델 정의 (학습 시와 동일) --------------------

def unfold_neighbors(x: torch.Tensor, k: int) -> torch.Tensor:
    pad = k // 2
    return F.unfold(x, kernel_size=k, padding=pad).view(x.size(0), k*k, x.size(2), x.size(3))

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
    # (선택) anchor 관련(learnable 저장 호환)
    anchor_mode: str = "scalar"
    anchor_learnable: bool = False

class MCPropNet(nn.Module):
    """
    Multi-geometry (hyperbolic approx. / elliptic) learnable propagation
    """
    def __init__(self, cfg: MCPropCfg):
        super().__init__()
        self.cfg = cfg
        in_ch = 3+1+1+1  # RGB, P/dmax, E_norm, ML

        self.enc = TinyFeat(in_ch, 64)
        self.res = ResidualHead(64, 64) if cfg.use_residual else None

        self.curv = CurvatureGen(64+1, cfg.kernels, cfg.kappa_min, cfg.kappa_max)  # +1: E_norm concat
        self.gate = KernelGate(64, cfg.kernels)

        if cfg.geometry.lower().startswith("ellip"):
            self.aff_head = EllipticAffinity(64, cfg.kernels, c_aff=32, tau_min=0.03, tau_max=0.5)
        else:
            self.aff_head = HCLApproxAffinity(64, cfg.kernels)

        # (선택) anchor learnable 호환 파라미터
        self.anchor_head = None     # map 모드 head
        self.alpha_raw   = None     # scalar learnable
        self.alpha_const = None     # scalar fixed(ckpt에 저장될 수 있음)

    def set_anchor_modules_from_ckpt(self, sd: Dict[str, torch.Tensor]):
        """ckpt에 anchor 관련 모듈 파라미터가 있으면 등록(호환용)."""
        if any("anchor_head" in k for k in sd.keys()) and self.anchor_head is None:
            self.anchor_head = nn.Conv2d(64, 1, kernel_size=3, padding=1)
            self.add_module("anchor_head", self.anchor_head)
        if any("alpha_raw" in k for k in sd.keys()) and (self.alpha_raw is None):
            self.alpha_raw = nn.Parameter(torch.tensor(0.0))
            self.register_parameter("alpha_raw", self.alpha_raw)

    def _anchor_blend(self, Dt, D0, DL, ML, feat):
        """LiDAR soft-Dirichlet 블렌딩."""
        if not self.cfg.use_sparse:
            return Dt
        if self.anchor_head is not None:       # map 모드
            alpha_eff = torch.sigmoid(self.anchor_head(feat)) * (ML.detach())
        elif self.alpha_raw is not None:       # scalar learnable
            alpha_eff = torch.sigmoid(self.alpha_raw) * (ML.detach())
        elif self.alpha_const is not None:     # scalar fixed
            alpha_eff = self.alpha_const * (ML.detach())
        else:                                  # cfg anchor_alpha (고정)
            alpha_eff = Dt.new_tensor(self.cfg.anchor_alpha) * (ML.detach())
        return (1.0 - alpha_eff) * Dt + alpha_eff * DL

    def forward(self, I, DL, ML, P, E_norm):
        cfg = self.cfg
        I01 = I / 255.0
        x_in = torch.cat([I01, P/cfg.dmax, E_norm, ML], dim=1)
        feat = self.enc(x_in)  # (B,64,H,W)

        D0 = (P + self.res(feat)).clamp(0, cfg.dmax) if self.res is not None else P

        kappa, scale, bias = self.curv(feat, E_norm)
        if cfg.geometry.lower().startswith("ellip"):
            A_list_raw = self.aff_head(feat, kappa)
        else:
            A_list_raw = self.aff_head(feat, scale, bias)

        A_list = normalize_affinity_list(A_list_raw)
        sigma  = self.gate(feat)  # (B,K,H,W)

        Dt = D0.clone()
        for _ in range(cfg.steps):
            mix_k = []
            for idx, k in enumerate(cfg.kernels):
                Ak = A_list[idx]
                kk = k*k
                patches = unfold_neighbors(Dt, k)
                center = kk // 2
                patches_c = patches.clone()
                patches_c[:, center:center+1, :, :] = D0  # screened-like
                D_next_k = (Ak * patches_c).sum(1, keepdim=True)
                mix_k.append(D_next_k)
            Dmix = torch.zeros_like(Dt)
            for idx, Dk in enumerate(mix_k):
                Dmix = Dmix + sigma[:, idx:idx+1] * Dk
            Dt = self._anchor_blend(Dmix, D0, DL, ML, feat).clamp(0, cfg.dmax)

        return Dt, {"D0": D0, "sigma": sigma, "A_list": A_list, "kappa": kappa}

# -------------------- 체크포인트 + 설정 로더 --------------------

def load_json(path: str) -> dict:
    with open(path, "r") as f:
        return json.load(f)

def load_model_from_ckpt_with_cfg(ckpt_path: str,
                                  cfg_override: Optional[dict],
                                  set_list: Optional[List[str]],
                                  device: torch.device) -> MCPropNet:
    ckpt = torch.load(ckpt_path, map_location=device)
    if "cfg" not in ckpt:
        raise RuntimeError("Checkpoint에 cfg가 없습니다. 학습 시 torch.save({'state_dict', 'cfg'}) 형태로 저장했는지 확인하세요.")
    base_cfg = {"model": dict(ckpt["cfg"])}  # 모델 cfg는 'model' 섹션으로 감싸 병합

    # override: --cfg 파일이 있으면 병합
    if cfg_override:
        _deep_update(base_cfg, cfg_override)

    # override: --set key=value 들 적용
    if set_list:
        _apply_overrides(base_cfg, set_list)

    # 최종 model cfg 구성
    mcfg = base_cfg.get("model", {})
    mcfg_dc = MCPropCfg(
        dmax=mcfg.get("dmax", 80.0),
        steps=mcfg.get("steps", 6),
        kernels=tuple(mcfg.get("kernels", [3,5,7])),
        use_residual=mcfg.get("use_residual", True),
        use_sparse=mcfg.get("use_sparse", True),
        anchor_alpha=mcfg.get("anchor_alpha", 0.7),
        kappa_min=mcfg.get("kappa_min", 1e-3),
        kappa_max=mcfg.get("kappa_max", 1.0),
        geometry=mcfg.get("geometry", "hyperbolic"),
        anchor_mode=mcfg.get("anchor_mode", "scalar"),
        anchor_learnable=mcfg.get("anchor_learnable", False),
    )
    model = MCPropNet(mcfg_dc).to(device)
    # anchor 모듈 호환 생성
    model.set_anchor_modules_from_ckpt(ckpt["state_dict"])
    model.load_state_dict(ckpt["state_dict"], strict=False)
    model.eval()
    return model, base_cfg

# -------------------- KITTI 경로/리스트 유틸 --------------------

def find_sparse_list(root: str) -> List[str]:
    pat02 = os.path.join(root, "data_depth_velodyne", "val", "**", "proj_depth", "velodyne_raw", "image_02", "*.png")
    pat03 = os.path.join(root, "data_depth_velodyne", "val", "**", "proj_depth", "velodyne_raw", "image_03", "*.png")
    return sorted(glob.glob(pat02, recursive=True)) + sorted(glob.glob(pat03, recursive=True))

def path_map(root: str, sp_path: str) -> Dict[str, str]:
    base_vel = os.path.join(root, "data_depth_velodyne", "val")
    rel = os.path.relpath(sp_path, base_vel)  # e.g. 2011.../proj_depth/velodyne_raw/image_02/0000.png
    rel_img = rel.replace(os.path.join("proj_depth","velodyne_raw")+os.sep, os.path.join("proj_depth")+os.sep)
    rel_gt  = rel.replace(os.path.join("proj_depth","velodyne_raw")+os.sep, os.path.join("proj_depth","groundtruth")+os.sep)
    return {
        "rel": rel,
        "rgb":   os.path.join(root, "kitti_raw",     "val", rel_img),
        "pseudo":os.path.join(root, "pseudo_depth_map", "val", rel),
        "estim": os.path.join(root, "kitti_raw_da",  "val", rel_img),
        "gt":    os.path.join(root, "data_depth_annotated", "val", rel_gt),
        "sparse": sp_path
    }

# -------------------- 지표 --------------------

def kitti_metrics_per_frame(pred_m: torch.Tensor, gt_m: torch.Tensor) -> Dict[str, float]:
    pred = pred_m.detach().cpu().numpy().astype(np.float64).squeeze()
    gt   = gt_m.detach().cpu().numpy().astype(np.float64).squeeze()
    mask = (gt > 0.0)
    n = int(mask.sum())
    if n == 0:
        return {'se':0.0,'ae':0.0,'ise':0.0,'iae':0.0,'n':0}
    pred = np.clip(pred[mask], 1e-6, 80.0)
    gt   = np.clip(gt[mask],   1e-6, 80.0)
    err = pred - gt
    inv_err = (1.0/pred) - (1.0/gt)
    return {
        'se':float(np.sum(err*err)),
        'ae':float(np.sum(np.abs(err))),
        'ise':float(np.sum(inv_err*inv_err)),
        'iae':float(np.sum(np.abs(inv_err))),
        'n':n
    }

def finalize_metrics(acc: Dict[str,float]) -> Dict[str,float]:
    n = max(1, int(acc.get('n',0)))
    rmse  = np.sqrt(acc['se'] / n) * 1000.0
    mae   = (acc['ae']  / n) * 1000.0
    irmse = np.sqrt(acc['ise']/ n) * 1000.0
    imae  = (acc['iae']/ n) * 1000.0
    return {'RMSE(mm)':rmse, 'MAE(mm)':mae, 'iRMSE(1/km)':irmse, 'iMAE(1/km)':imae}

# -------------------- 평가 루프 --------------------

@torch.no_grad()
def evaluate(root: str, ckpt: str, out_root: str, device: torch.device,
             cfg_override: Optional[dict], set_list: Optional[List[str]],
             save_viz: bool, per_image: bool, max_frames: int = -1):
    os.makedirs(out_root, exist_ok=True)
    viz_root = os.path.join(out_root, "viz", "val")
    dep_root = os.path.join(out_root, "pred", "val")

    # 모델+최종 cfg 로드
    model, final_cfg = load_model_from_ckpt_with_cfg(ckpt, cfg_override, set_list, device)
    dmax = float(model.cfg.dmax)

    # root 결정: 우선 CLI --root, 없으면 cfg(paths.root)
    if root is None or root == "":
        root = final_cfg.get("paths", {}).get("root", "")
    if not root:
        raise ValueError("--root 를 주거나, --cfg 내 paths.root 를 지정해야 합니다.")
    print(f"[INFO] dataset root = {root}")
    print(f"[INFO] model cfg    = {model.cfg}")

    # 목록
    sparse_list = find_sparse_list(root)
    if max_frames > 0:
        sparse_list = sparse_list[:max_frames]

    # CSV
    csv_path = os.path.join(out_root, "metrics_val.csv")
    os.makedirs(os.path.dirname(csv_path), exist_ok=True)
    csv_file = open(csv_path, 'w', newline='')
    writer = csv.writer(csv_file)
    writer.writerow(["rel_path", "n_valid", "RMSE(mm)", "MAE(mm)", "iRMSE(1/km)", "iMAE(1/km)"])

    acc = {'se':0.0,'ae':0.0,'ise':0.0,'iae':0.0,'n':0}
    missing = {"rgb":0, "pseudo":0, "estim":0, "gt":0}

    pbar = tqdm(sparse_list, desc="Eval (val)")
    for sp in pbar:
        paths = path_map(root, sp)
        rel   = paths["rel"]

        # 존재 확인
        if not os.path.exists(paths["rgb"]):   missing["rgb"] += 1;   continue
        if not os.path.exists(paths["pseudo"]):missing["pseudo"] += 1;continue
        if not os.path.exists(paths["estim"]): missing["estim"] += 1; continue
        has_gt = os.path.exists(paths["gt"])
        if not has_gt: missing["gt"] += 1

        # 로드
        I  = load_rgb(paths["rgb"]).to(device)
        DL = load_depth16_mm_as_m(paths["sparse"]).to(device)
        ML = (DL > 0).float()
        P  = load_pseudo_auto(paths["pseudo"]).to(device)
        E  = load_estimation_8bit_norm(paths["estim"]).to(device)  # [0,1]
        GT = load_depth16_mm_as_m(paths["gt"]).to(device) if has_gt else None

        # 추론
        pred, aux = model(I, DL, ML, P, E)

        # 저장
        out_path = os.path.join(dep_root, rel)
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        cv2.imwrite(out_path, (pred.clamp(0, dmax).cpu().numpy().squeeze() * 256.0).astype(np.uint16))

        if save_viz:
            base_noext = os.path.splitext(os.path.join(viz_root, rel))[0]
            os.makedirs(os.path.dirname(base_noext), exist_ok=True)
            save_jet(base_noext + "_pred_jet.png", pred, dmax=dmax)
            save_jet(base_noext + "_d0_jet.png",   aux["D0"], dmax=dmax)
            save_sparse_jet(base_noext + "_sparse_jet.png", DL, ML, dmax=dmax)

        # 지표
        if has_gt:
            m = kitti_metrics_per_frame(pred, GT)
            for k in acc: acc[k] += m[k]
            fm = finalize_metrics(m)
            writer.writerow([rel, m['n'],
                             f"{fm['RMSE(mm)']:.3f}", f"{fm['MAE(mm)']:.3f}",
                             f"{fm['iRMSE(1/km)']:.3f}", f"{fm['iMAE(1/km)']:.3f}"])
            if per_image:
                tqdm.write(f"{rel} | n={m['n']:6d}  RMSE={fm['RMSE(mm)']:8.2f}  MAE={fm['MAE(mm)']:7.2f}  "
                           f"iRMSE={fm['iRMSE(1/km)']:7.2f}  iMAE={fm['iMAE(1/km)']:7.2f}")

    csv_file.close()

    # 요약
    print(f"[INFO] missing: rgb={missing['rgb']}, pseudo={missing['pseudo']}, "
          f"estim={missing['estim']}, gt(missing-not-evaluated)={missing['gt']}")
    if acc['n'] > 0:
        tot = finalize_metrics(acc)
        print("[val] KITTI metrics (dataset summary)")
        for k,v in tot.items():
            print(f"  {k:>12}: {v:10.3f}")
    else:
        print("[val] No GT found — metrics not computed.")

# -------------------- 엔트리 --------------------

def parse_args():
    ap = argparse.ArgumentParser("Evaluate 1-shot MCPropNet on KITTI val (cfg-aware)")
    ap.add_argument("--root", default="", help="Datasets root (없으면 --cfg의 paths.root 사용)")
    ap.add_argument("--ckpt", required=True, help="Path to checkpoint (.pt)")
    ap.add_argument("--out",  required=True, help="Output root for predictions & viz")
    ap.add_argument("--cfg",  default="", help="Optional JSON config to override ckpt cfg")
    ap.add_argument("--set",  nargs="*", default=[], help="Inline overrides, e.g., model.steps=8 model.geometry=elliptic")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--save-viz", action="store_true", help="Save jet visualizations")
    ap.add_argument("--per-image", action="store_true", help="Print per-frame metrics")
    ap.add_argument("--max-frames", type=int, default=-1, help="Limit frames for quick test")
    return ap.parse_args()

def main():
    args = parse_args()
    device = torch.device(args.device)

    cfg_override = None
    if args.cfg and os.path.isfile(args.cfg):
        cfg_override = load_json(args.cfg)
        print(f"[INFO] loaded cfg file: {args.cfg}")

    evaluate(root=args.root, ckpt=args.ckpt, out_root=args.out, device=device,
             cfg_override=cfg_override, set_list=args.set,
             save_viz=args.save_viz, per_image=args.per_image, max_frames=args.max_frames)

if __name__ == "__main__":
    main()
