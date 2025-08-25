#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Evaluate MCPropNet on NYUv2 validation set with sparse->Poisson initialization
- Reads sparse (DL/ML) from validation HDF5, or deterministically samples 500 from GT if missing
- Loads estimation E (file), runs Poisson(E, DL, ML) -> P
- Center crop (default 228x304) to match training
- Reports RMSE/MAE (mm) and iRMSE/iMAE (1/m)
- Saves 16-bit PNG predictions with scale meta (scale=1000), optional jet previews
"""

import os, argparse, json, csv, random, hashlib
from typing import Dict, Any, List, Optional, Tuple
import numpy as np
from tqdm import tqdm

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from PIL import Image
from PIL.PngImagePlugin import PngInfo
import h5py

# ---------- config / model ----------
from config.loader import load_config
from config.schema import MCPropCfg
from train_nyu import MCPropNet  # ensure identical architecture

# ---------- io utils (for viz only) ----------
from utils.io_utils import save_jet, save_sparse_jet

# =============================================================================
# IO helpers: PNG16 <-> meters
# =============================================================================
def save_depth_png16_with_scale(path: str, depth_m: np.ndarray, scale_mm: float = 1000.0):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    d = np.nan_to_num(depth_m, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
    u16 = np.clip(d * scale_mm + 0.5, 0, 65535).astype(np.uint16)
    im = Image.fromarray(u16, mode="I;16")
    meta = PngInfo()
    meta.add_text("scale", f"{scale_mm}")
    meta.add_text("offset", "0.0")
    im.save(path, pnginfo=meta)

def _read_u16(path: str) -> Tuple[np.ndarray, dict]:
    img = Image.open(path)
    text = dict(img.text) if hasattr(img, "text") else {}
    arr = np.array(img)
    if arr.dtype != np.uint16:
        arr = arr.astype(np.uint16)
    return arr, text

def _u16_to_meters(u16: np.ndarray, text: dict, dmax: float, default_scale_mm: Optional[float]) -> np.ndarray:
    if ("scale" in text) or ("offset" in text):
        s = float(text.get("scale", "1000.0"))
        t = float(text.get("offset", "0.0"))
        depth = (u16.astype(np.float32) - t) / max(1e-12, s)
        return np.clip(depth, 0.0, dmax)
    if default_scale_mm is not None:
        depth = u16.astype(np.float32) / float(default_scale_mm)
        return np.clip(depth, 0.0, dmax)
    depth = (u16.astype(np.float32) / 65535.0) * dmax
    return np.clip(depth, 0.0, dmax)

def _read_float_or_u16_to_meters(path: str, dmax: float, scale_mm: Optional[float]) -> np.ndarray:
    ext = os.path.splitext(path)[1].lower()
    if ext == ".npy":
        arr = np.load(path).astype(np.float32)
        return np.clip(arr, 0.0, dmax)
    u16, text = _read_u16(path)
    return _u16_to_meters(u16, text, dmax=dmax, default_scale_mm=scale_mm)

# =============================================================================
# H5 helpers (robust to different key names/shapes)
# =============================================================================
RGB_KEYS      = ["rgb", "image", "images", "color", "colors"]
DEPTH_KEYS    = ["gt", "depth", "depths", "gt_depth", "ground_truth"]
DL_KEYS       = ["dl", "sparse_depth", "sparse_dl", "lidar_depth"]
ML_KEYS       = ["ml", "mask", "sparse_mask", "valid_mask"]
SPARSE_X_KEYS = ["sparse_x", "x"]
SPARSE_Y_KEYS = ["sparse_y", "y"]
SPARSE_D_KEYS = ["sparse_d", "d", "depth_values"]
SPARSE_BUNDLE = ["sparse", "points"]  # (K,3) with (x,y,d)
ID_KEYS       = ["ids", "id", "names", "name"]

def _h5_find_first(f: h5py.File, keys: List[str]) -> Optional[str]:
    for k in keys:
        if k in f: return k
    for k in keys:
        if "/" in k:
            grp, ds = k.split("/", 1)
            if grp in f and ds in f[grp]: return k
    return None

def _h5_get(f: h5py.File, key: str):
    if "/" in key:
        grp, ds = key.split("/", 1)
        return f[grp][ds]
    return f[key]

def _as_chw_uint8(arr: np.ndarray) -> np.ndarray:
    if arr.ndim == 2:
        arr = np.stack([arr, arr, arr], axis=0)
    elif arr.ndim == 3 and arr.shape[-1] == 3:
        arr = np.transpose(arr, (2, 0, 1))
    elif arr.ndim == 3 and arr.shape[0] == 3:
        pass
    elif arr.ndim == 3 and arr.shape[-1] == 1:
        g = arr[..., 0]; arr = np.stack([g, g, g], axis=0)
    elif arr.ndim == 3 and arr.shape[0] == 1:
        g = arr[0]; arr = np.stack([g, g, g], axis=0)
    else:
        raise RuntimeError(f"Unexpected RGB shape: {arr.shape}")
    if arr.dtype != np.uint8:
        a = arr.astype(np.float32)
        a_max = float(a.max()) if a.size else 1.0
        if a_max <= 1.0 + 1e-6: a = np.clip(a * 255.0, 0, 255)
        else:                   a = np.clip(a, 0, 255)
        arr = a.astype(np.uint8)
    return arr

def _h5_count_records(h5_path: str) -> int:
    with h5py.File(h5_path, "r") as f:
        rgb_k = _h5_find_first(f, RGB_KEYS)
        dep_k = _h5_find_first(f, DEPTH_KEYS)
        for k in [rgb_k, dep_k]:
            if k is None: continue
            ds = _h5_get(f, k)
            if ds.ndim >= 3: return ds.shape[0]
            else:            return 1
    raise RuntimeError(f"Cannot infer record count from {h5_path}.")

def _h5_ids_or_index(h5_path: str, N: int) -> List[str]:
    base = os.path.splitext(os.path.basename(h5_path))[0]
    with h5py.File(h5_path, "r") as f:
        id_k = _h5_find_first(f, ID_KEYS)
        if id_k is None:
            return [base] if N == 1 else [str(i).zfill(5) for i in range(N)]
        ds = _h5_get(f, id_k)
        out = []
        for i in range(N):
            v = ds[i]
            if isinstance(v, bytes):
                out.append(v.decode("utf-8", errors="ignore"))
            elif hasattr(v, "astype"):
                try: out.append(str(v.astype(str)))
                except Exception: out.append(str(v))
            else:
                out.append(str(v))
        return out

def _take_sample_2d(ds, idx, expect_hw=None):
    if ds.ndim <= 2:
        arr = np.array(ds[()], dtype=np.float32)
        return arr
    if ds.ndim == 3:
        a = np.array(ds[idx])
        a = np.squeeze(a)
        if a.ndim == 2: return a.astype(np.float32)
    ok = False; arr = None
    for axis in range(ds.ndim):
        slicer = [slice(None)]*ds.ndim; slicer[axis] = idx
        a = np.array(ds[tuple(slicer)]); a = np.squeeze(a)
        if a.ndim == 2:
            arr = a.astype(np.float32); ok = True; break
    if not ok:
        a = np.array(ds[idx]); a = np.squeeze(a)
        if a.ndim == 2: arr = a.astype(np.float32)
        elif (expect_hw is not None) and (a.ndim == 1) and (a.size == expect_hw[0]*expect_hw[1]):
            arr = a.reshape(expect_hw).astype(np.float32)
        else:
            raise RuntimeError(f"Cannot extract 2D slice from dataset with shape {ds.shape}")
    return arr

def _h5_read_record(h5_path: str, idx: int, dmax: float,
                    sparse_scale_mm: Optional[float]) -> Tuple[np.ndarray,np.ndarray,Optional[np.ndarray],Optional[np.ndarray]]:
    with h5py.File(h5_path, "r") as f:
        k_rgb = _h5_find_first(f, RGB_KEYS)
        if k_rgb is None: raise RuntimeError(f"[{h5_path}] RGB dataset not found.")
        ds_rgb = _h5_get(f, k_rgb)
        if ds_rgb.ndim <= 2: rgb_raw = np.array(ds_rgb[()])
        elif ds_rgb.ndim == 3 and 3 in ds_rgb.shape: rgb_raw = np.array(ds_rgb[()])
        else: rgb_raw = np.array(ds_rgb[idx])
        rgb = _as_chw_uint8(rgb_raw)  # (3,H,W)
        H, W = rgb.shape[1], rgb.shape[2]

        k_gt = _h5_find_first(f, DEPTH_KEYS)
        if k_gt is None: raise RuntimeError(f"[{h5_path}] GT depth dataset not found.")
        ds_gt = _h5_get(f, k_gt)
        gt = _take_sample_2d(ds_gt, idx, expect_hw=(H, W))
        gt = np.clip(gt, 0.0, dmax)

        DL_np, ML_np = None, None
        k_dl = _h5_find_first(f, DL_KEYS)
        if k_dl is not None:
            ds_dl = _h5_get(f, k_dl)
            dl = _take_sample_2d(ds_dl, idx, expect_hw=(H, W))
            if sparse_scale_mm is not None: dl = dl / float(sparse_scale_mm)
            dl = np.clip(dl, 0.0, dmax); DL_np = dl
            k_ml = _h5_find_first(f, ML_KEYS)
            if k_ml is not None:
                ds_ml = _h5_get(f, k_ml)
                ml = _take_sample_2d(ds_ml, idx, expect_hw=(H, W)).astype(np.float32)
                ML_np = (ml > 0).astype(np.uint8)
            else:
                ML_np = (dl > 0).astype(np.uint8)
        else:
            # points (x,y,d) 형태 지원
            k_bundle = _h5_find_first(f, SPARSE_BUNDLE)
            if k_bundle is not None:
                pts = _h5_get(f, k_bundle)[idx]
                if pts.ndim == 1 and pts.size == 0:
                    DL_np = np.zeros((H,W), dtype=np.float32); ML_np = np.zeros((H,W), dtype=np.uint8)
                else:
                    x = pts[:,0].astype(np.int32); y = pts[:,1].astype(np.int32)
                    d = pts[:,2].astype(np.float32)
                    if sparse_scale_mm is not None: d = d / float(sparse_scale_mm)
                    x = np.clip(x, 0, W-1); y = np.clip(y, 0, H-1); d = np.clip(d, 0.0, dmax)
                    DL_np = np.zeros((H,W), dtype=np.float32); ML_np = np.zeros((H,W), dtype=np.uint8)
                    DL_np[y, x] = d; ML_np[y, x] = 1
        return rgb, gt, DL_np, ML_np

# =============================================================================
# Center crop + deterministic sparse sampling (fallback)
# =============================================================================
def _center_crop_slices(H: int, W: int, out_h: int, out_w: int):
    if out_h > H or out_w > W:
        raise RuntimeError(f"Crop size ({out_w}x{out_h}) exceeds input ({W}x{H}).")
    top  = (H - out_h) // 2
    left = (W - out_w) // 2
    return slice(top, top + out_h), slice(left, left + out_w)

def _center_crop_tensor(x: torch.Tensor, out_h: int, out_w: int) -> torch.Tensor:
    if x.dim() == 2:
        H, W = x.shape; ys, xs = _center_crop_slices(H, W, out_h, out_w); return x[ys, xs]
    elif x.dim() == 3:
        C, H, W = x.shape; ys, xs = _center_crop_slices(H, W, out_h, out_w); return x[:, ys, xs]
    else:
        raise RuntimeError(f"Expected 2D/3D tensor, got {tuple(x.shape)}")

def _stable_int_seed(*parts) -> int:
    h = hashlib.sha256("|".join(str(p) for p in parts).encode("utf-8")).digest()
    return int.from_bytes(h[:8], "little") & 0x7fffffff

def _make_sparse_from_gt_deterministic(GT_m: torch.Tensor, N: int, key_tuple) -> Tuple[torch.Tensor, torch.Tensor]:
    H, W = GT_m.shape[-2:]
    valid = (GT_m > 0.0).view(-1)
    idx_all = valid.nonzero(as_tuple=False).view(-1)
    if idx_all.numel() == 0:
        ML = torch.zeros((1,H,W), dtype=torch.float32, device=GT_m.device)
        DL = torch.zeros_like(GT_m)
        return DL, ML
    seed = _stable_int_seed(*key_tuple, H, W)
    g = torch.Generator(device=GT_m.device); g.manual_seed(seed)
    sel = idx_all[torch.randperm(idx_all.numel(), generator=g)[:min(N, idx_all.numel())]]
    ML = torch.zeros((H*W,), dtype=torch.float32, device=GT_m.device); ML[sel] = 1.0
    ML = ML.view(1, H, W); DL = GT_m * ML
    return DL, ML

# =============================================================================
# Poisson completion (import or fallback)
# =============================================================================
try:
    from utils.DC.poisson import poisson_complete as _poisson_external
    def poisson_complete(E_m, DL, ML, lam, iters, hard, dmax):  # thin wrapper
        return _poisson_external(E_m, DL, ML, lam=lam, iters=iters, hard=hard, dmax=dmax)
except Exception:
    @torch.no_grad()
    def poisson_complete(E_m: torch.Tensor, DL: torch.Tensor, ML: torch.Tensor,
                         lam: float = 800.0, iters: int = 300, hard: bool = False,
                         dmax: float = 10.0) -> torch.Tensor:
        def neigh_sum(X):
            up    = F.pad(X, (0,0,1,0), mode='replicate')[:,:,:-1,:]
            down  = F.pad(X, (0,0,0,1), mode='replicate')[:,:,1:,:]
            left  = F.pad(X, (1,0,0,0), mode='replicate')[:,:,:,:-1]
            right = F.pad(X, (0,1,0,0), mode='replicate')[:,:,:,1:]
            return up + down + left + right
        lap_E = neigh_sum(E_m) - 4.0 * E_m
        b = (-lap_E) + lam * ML * DL
        denom = 4.0 + lam * ML
        D = E_m.clone()
        for _ in range(int(iters)):
            sumN = neigh_sum(D)
            D_new = (sumN + b) / denom
            if hard: D_new = ML * DL + (1.0 - ML) * D_new
            D = D_new
        return D.clamp_(0.0, float(dmax))

# =============================================================================
# Split builder (uses nyu.* in cfg; supports h5_val/_dir/_glob, mono_fmt)
# =============================================================================
import glob
def _collect_h5_list(nyu: Dict[str,Any], split: str) -> List[str]:
    key_file = f"h5_{split}"
    key_glob = f"h5_{split}_glob"
    key_dir  = f"h5_{split}_dir"
    paths = []
    p_file = nyu.get(key_file, "")
    p_glob = nyu.get(key_glob, "")
    p_dir  = nyu.get(key_dir, "")
    if p_file and os.path.isfile(p_file): return [p_file]
    if p_glob:
        paths += [p for p in glob.glob(p_glob, recursive=True) if os.path.isfile(p) and p.lower().endswith(".h5")]
    if p_dir and os.path.isdir(p_dir):
        for r,_,fs in os.walk(p_dir):
            for fn in fs:
                if fn.lower().endswith(".h5"): paths.append(os.path.join(r, fn))
    return sorted(list(dict.fromkeys(paths)))

def build_val_entries_from_cfg(cfg: Dict[str,Any]) -> List[dict]:
    nyu = cfg["nyu"]; mono_fmt = nyu.get("mono_fmt", "")
    val_files = _collect_h5_list(nyu, "val")
    if not val_files: raise RuntimeError("No validation H5 found. Set nyu.h5_val or h5_val_dir or h5_val_glob.")
    ents = []
    for hp in val_files:
        N = _h5_count_records(hp)
        ids = _h5_ids_or_index(hp, N)
        stem = os.path.splitext(os.path.basename(hp))[0]
        for i in range(N):
            rid_i = ids[i] if i < len(ids) else f"{i:05d}"
            rid = stem if N == 1 else f"{stem}_{rid_i}"
            mono = mono_fmt.format(id=rid) if mono_fmt else None
            ents.append({"h5": hp, "idx": i, "id": rid, "mono": mono})
    # K_val 제한 (optional)
    ks = cfg.get("kshot", {})
    K_val = ks.get("K_val", None)
    if K_val is not None:
        random.seed(int(ks.get("seed", 1)))
        if len(ents) > int(K_val):
            ents = random.sample(ents, k=int(K_val))
    return ents

# =============================================================================
# Dataset: reads sparse, estimation; runs Poisson to produce P
# =============================================================================
class NYUValDatasetPoisson(Dataset):
    def __init__(self, entries: List[dict], dmax: float,
                 mono_scale: Optional[float],
                 sparse_scale: Optional[float],
                 crop_hw: Optional[Tuple[int,int]],
                 poisson_lam: float, poisson_iters: int, poisson_hard: bool):
        super().__init__()
        self.entries = entries
        self.dmax = float(dmax)
        self.mono_scale = mono_scale
        self.sparse_scale = sparse_scale
        self.crop_hw = crop_hw
        self.p_lam = float(poisson_lam)
        self.p_iters = int(poisson_iters)
        self.p_hard = bool(poisson_hard)

    def __len__(self): return len(self.entries)

    def __getitem__(self, i: int):
        ent = self.entries[i]
        # H5
        rgb_chw, gt_hw, dl_hw, ml_hw = _h5_read_record(ent["h5"], ent["idx"], self.dmax, self.sparse_scale)
        I  = torch.from_numpy(rgb_chw)            # (3,H,W) uint8
        GT = torch.from_numpy(gt_hw).unsqueeze(0) # (1,H,W) m

        # center crop
        if self.crop_hw is not None:
            ch, cw = self.crop_hw
            I  = _center_crop_tensor(I,  ch, cw)
            GT = _center_crop_tensor(GT, ch, cw)

        # sparse
        key = (ent["h5"], ent["idx"], *(self.crop_hw if self.crop_hw else (-1,-1)))
        if dl_hw is None or ml_hw is None:
            DL, ML = _make_sparse_from_gt_deterministic(GT, 500, key)
        else:
            DL = torch.from_numpy(dl_hw).unsqueeze(0)
            ML = torch.from_numpy(ml_hw.astype(np.float32)).unsqueeze(0)
            if self.crop_hw is not None:
                DL = _center_crop_tensor(DL, ch, cw)
                ML = _center_crop_tensor(ML, ch, cw)

        # estimation
        E = torch.zeros_like(GT)
        if ent.get("mono") and os.path.isfile(ent["mono"]):
            E_np = _read_float_or_u16_to_meters(ent["mono"], dmax=self.dmax, scale_mm=self.mono_scale)
            E = torch.from_numpy(E_np).unsqueeze(0) if E_np.ndim == 2 else torch.from_numpy(E_np)
            if self.crop_hw is not None:
                E = _center_crop_tensor(E, ch, cw)

        # Poisson completion (meters)
        P = poisson_complete(E.unsqueeze(0), DL.unsqueeze(0), ML.unsqueeze(0),
                             lam=self.p_lam, iters=self.p_iters, hard=self.p_hard, dmax=self.dmax)[0]

        E_norm = torch.clamp(E / self.dmax, 0.0, 1.0)
        return I, DL, ML, P, E_norm, GT

# =============================================================================
# Metrics
# =============================================================================
def _seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**31
    np.random.seed(worker_seed); random.seed(worker_seed)

@torch.no_grad()
def _resize_like(x: torch.Tensor, ref: torch.Tensor) -> torch.Tensor:
    H, W = ref.shape[-2:]
    if x.shape[-2:] == (H, W): return x
    return F.interpolate(x, size=(H, W), mode="nearest")

@torch.no_grad()
def compute_metrics(pred: torch.Tensor, gt: torch.Tensor):
    assert pred.shape == gt.shape
    valid = (gt > 0.0)
    sse = torch.sum(((pred - gt) ** 2)[valid]).item()
    sae = torch.sum((pred - gt).abs()[valid]).item()
    cnt = int(valid.sum().item())

    inv_pred = 1.0 / torch.clamp(pred, min=1e-6)
    inv_gt   = 1.0 / torch.clamp(gt,   min=1e-6)
    inv_valid = valid & torch.isfinite(inv_pred) & torch.isfinite(inv_gt)
    is2 = torch.sum(((inv_pred - inv_gt) ** 2)[inv_valid]).item()
    ia1 = torch.sum((inv_pred - inv_gt).abs()[inv_valid]).item()
    icnt = int(inv_valid.sum().item())
    return {"sse":sse, "sae":sae, "cnt":cnt, "is2":is2, "ia1":ia1, "icnt":icnt}

def merge_metrics(M: List[Dict[str, float]]):
    sse  = sum(m["sse"]  for m in M); sae  = sum(m["sae"]  for m in M); cnt  = sum(m["cnt"]  for m in M)
    is2  = sum(m["is2"]  for m in M); ia1  = sum(m["ia1"]  for m in M); icnt = sum(m["icnt"] for m in M)
    rmse_m = (sse / max(1, cnt)) ** 0.5 if cnt>0 else float("nan")
    mae_m  = (sae / max(1, cnt))        if cnt>0 else float("nan")
    irmse  = (is2 / max(1, icnt)) ** 0.5 if icnt>0 else float("nan")
    imae   = (ia1 / max(1, icnt))        if icnt>0 else float("nan")
    return {"RMSE_mm": rmse_m * 1000.0, "MAE_mm": mae_m * 1000.0, "iRMSE": irmse, "iMAE": imae, "valid_px": cnt}

# =============================================================================
# Evaluation core
# =============================================================================
@torch.no_grad()
def run_eval(cfg: Dict[str,Any], ckpt_path: str, out_root: str,
             device: torch.device, save_viz: bool, per_image: bool,
             max_frames: int = -1, preview_every: int = 50):

    os.makedirs(out_root, exist_ok=True)
    viz_root  = os.path.join(out_root, "viz");  os.makedirs(viz_root, exist_ok=True)
    pred_root = os.path.join(out_root, "pred"); os.makedirs(pred_root, exist_ok=True)

    # --- load checkpoint first (to get exact dmax etc.) ---
    ckpt = torch.load(ckpt_path, map_location="cpu")
    if "state_dict" not in ckpt or "cfg" not in ckpt:
        raise RuntimeError("Checkpoint must contain 'state_dict' and 'cfg' (saved in train_nyu.py).")
    mcfg_ckpt = ckpt["cfg"]
    model = MCPropNet(MCPropCfg(
        dmax=mcfg_ckpt["dmax"], steps=mcfg_ckpt["steps"], kernels=tuple(mcfg_ckpt["kernels"]),
        use_residual=mcfg_ckpt["use_residual"], use_sparse=mcfg_ckpt["use_sparse"],
        anchor_alpha=mcfg_ckpt["anchor_alpha"], anchor_learnable=mcfg_ckpt["anchor_learnable"],
        anchor_mode=mcfg_ckpt["anchor_mode"], kappa_min=mcfg_ckpt["kappa_min"], kappa_max=mcfg_ckpt["kappa_max"],
        geometry=mcfg_ckpt["geometry"]
    )).to(device)
    model.load_state_dict(ckpt["state_dict"], strict=True)
    model.eval()
    dmax = float(model.cfg.dmax)  # use model's dmax for dataset normalization/clamp

    # --- build val entries from cfg (supports h5_val/_dir/_glob, mono_fmt) ---
    val_entries = build_val_entries_from_cfg(cfg)
    if max_frames > 0:
        val_entries = val_entries[:max_frames]

    # --- dataset/loader options (from eval cfg) ---
    nyu  = cfg["nyu"]
    mono_scale   = nyu.get("mono_scale", None)
    sparse_scale = nyu.get("sparse_scale", None)
    crop_h = nyu.get("crop_h", 228)
    crop_w = nyu.get("crop_w", 304)
    crop_hw = (crop_h, crop_w) if (crop_h is not None and crop_w is not None) else None

    pconf   = cfg.get("poisson", {})
    p_lam   = float(pconf.get("lambda", 800.0))
    p_iters = int(pconf.get("iters", 300))
    p_hard  = bool(pconf.get("hard", False))

    val_ds = NYUValDatasetPoisson(
        val_entries, dmax=dmax,
        mono_scale=mono_scale, sparse_scale=sparse_scale, crop_hw=crop_hw,
        poisson_lam=p_lam, poisson_iters=p_iters, poisson_hard=p_hard
    )

    ks = cfg.get("kshot", {})
    nw   = int(ks.get("num_workers", 4))
    bs   = int(ks.get("batch_size", 1))
    pw   = bool(ks.get("persistent_workers", True)) if nw > 0 else False

    val_loader = DataLoader(
        val_ds, batch_size=bs, shuffle=False, num_workers=nw, pin_memory=True,
        persistent_workers=pw, worker_init_fn=_seed_worker
    )

    # --- CSV ---
    csv_path = os.path.join(out_root, "val_metrics_per_image.csv")
    csv_f = open(csv_path, "w", newline="")
    csv_w = csv.writer(csv_f)
    csv_w.writerow(["id", "RMSE_mm", "MAE_mm", "iRMSE(1/m)", "iMAE(1/m)", "valid_px"])

    # --- iterate ---
    all_mets = []; n_total = 0
    pbar = tqdm(val_loader, desc="Eval (NYUv2 val)")
    for it, batch in enumerate(pbar, start=1):
        I, DL, ML, P, E, GT = [t.to(device) for t in batch]
        pred, _ = model(I, DL, ML, P, E)

        if GT.shape[-2:] != pred.shape[-2:]:
            GT = _resize_like(GT, pred)

        mets = compute_metrics(pred, GT)
        all_mets.append(mets); n_total += I.size(0)

        rid = val_entries[it-1].get("id", f"{it:05d}") if (it-1) < len(val_entries) else f"{it:05d}"
        merged_i = merge_metrics([mets])
        csv_w.writerow([rid,
                        f"{merged_i['RMSE_mm']:.3f}",
                        f"{merged_i['MAE_mm']:.3f}",
                        f"{merged_i['iRMSE']:.6f}",
                        f"{merged_i['iMAE']:.6f}",
                        merged_i["valid_px"]])

        # save prediction (16-bit PNG with scale meta)
        pred_np = pred[0,0].detach().cpu().numpy()
        save_depth_png16_with_scale(os.path.join(pred_root, f"{rid}_pred16.png"), pred_np, scale_mm=1000.0)

        if save_viz and (preview_every <= 0 or it % preview_every == 0):
            save_jet(os.path.join(viz_root, f"{rid}_pred_jet.png"), pred[:1].cpu(), dmax=dmax)
            save_sparse_jet(os.path.join(viz_root, f"{rid}_sparse_jet.png"), DL[:1].cpu(), ML[:1].cpu(), dmax=dmax)

    csv_f.close()

    # --- summary ---
    merged = merge_metrics(all_mets)
    print("\n=== Validation (NYUv2) ===")
    print(f"Samples      : {n_total}")
    print(f"Valid pixels : {merged['valid_px']}")
    print(f"RMSE (mm)    : {merged['RMSE_mm']:.3f}")
    print(f"MAE  (mm)    : {merged['MAE_mm']:.3f}")
    print(f"iRMSE (1/m)  : {merged['iRMSE']:.6f}")
    print(f"iMAE  (1/m)  : {merged['iMAE']:.6f}")
    print(f"Per-image CSV: {csv_path}")

# =============================================================================
# CLI
# =============================================================================
def parse_args():
    ap = argparse.ArgumentParser("Evaluate MCPropNet on NYUv2 val (sparse+Poisson init)")
    ap.add_argument("--config", required=True, help="평가용 JSON/YAML config (nyu.*, poisson.*, kshot.*)")
    ap.add_argument("--ckpt",   required=True, help="체크포인트 경로 (*.pt, saved by train_nyu.py)")
    ap.add_argument("--out",    required=True, help="출력 루트 (pred/, viz/, CSV 등 저장)")

    # 런타임 오버라이드
    ap.add_argument("--set",  nargs="*", default=[], help="오버라이드(예: nyu.crop_h=240 nyu.crop_w=320)")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--save-viz", action="store_true", help="JET 시각화 저장")
    ap.add_argument("--per-image", action="store_true", help="프레임별 지표 로그")
    ap.add_argument("--max-frames", type=int, default=-1, help="평가 프레임 제한(디버깅용)")
    ap.add_argument("--preview-every", type=int, default=50, help="JET 저장 주기(N프레임마다 1장; <=0이면 매장)")

    # 검증 입력 오버라이드 — 본 스크립트 빌더가 처리
    ap.add_argument("--h5_val",      default="", help="단일 HDF5 파일")
    ap.add_argument("--h5_val_dir",  default="", help="검증 HDF5 폴더(재귀)")
    ap.add_argument("--h5_val_glob", default="", help="검증 HDF5 글롭 패턴(예: /path/val/**/*.h5)")
    ap.add_argument("--mono_fmt",    default="", help="estimation 경로 포맷(예: /.../val/{id}_da_vitl_16bit.png)")
    return ap.parse_args()

def _apply_overrides(cfg: dict, kv_list: List[str]) -> dict:
    def _parse_value(val: str):
        if isinstance(val, str) and val.lower() in ("true","false"):
            return val.lower() == "true"
        if isinstance(val, str):
            try:
                if "." in val: return float(val)
                return int(val)
            except ValueError:
                return val
        return val
    for kv in kv_list:
        if "=" not in kv:
            print(f"[WARN] ignore override without '=': {kv}"); continue
        key, val = kv.split("=", 1); val = _parse_value(val)
        node = cfg; parts = key.split(".")
        for p in parts[:-1]:
            if p not in node or not isinstance(node[p], dict): node[p] = {}
            node = node[p]
        node[parts[-1]] = val
    return cfg

def main():
    args = parse_args()
    device = torch.device(args.device)

    # load & overrides
    cfg = load_config(args.config)
    if args.set:
        cfg = _apply_overrides(cfg, args.set)

    # override nyu inputs if provided
    nyu = cfg.setdefault("nyu", {})
    if args.h5_val:      nyu["h5_val"] = args.h5_val
    if args.h5_val_dir:  nyu["h5_val_dir"] = args.h5_val_dir
    if args.h5_val_glob: nyu["h5_val_glob"] = args.h5_val_glob
    if args.mono_fmt:    nyu["mono_fmt"] = args.mono_fmt

    run_eval(cfg, args.ckpt, args.out, device,
             save_viz=args.save_viz, per_image=args.per_image,
             max_frames=args.max_frames, preview_every=args.preview_every)

if __name__ == "__main__":
    main()
