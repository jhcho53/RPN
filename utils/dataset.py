# utils/dataset.py
import os, glob, random
from typing import List, Dict, Any, Tuple
import torch
from torch.utils.data import Dataset

from .io_utils import (
    load_rgb, load_depth16_mm_as_m, load_pseudo_auto, load_estimation_8bit_norm
)

# ----------------- 1) CSV 기반(이전 호환: 필요시 사용) -----------------
def _resolve_path(base_dir: str, p: str) -> str:
    if not p: return ""
    if os.path.isabs(p): return p
    return os.path.normpath(os.path.join(base_dir, p))

def _read_list_file(list_path: str) -> List[Dict[str, str]]:
    lines = []
    base_dir = os.path.abspath(os.path.dirname(list_path))
    with open(list_path, "r") as f:
        raw = [x.strip() for x in f.readlines() if len(x.strip()) > 0]
    if not raw: return []

    header = None
    first = raw[0]
    if any(h in first.lower() for h in ["rgb","sparse","pseudo","estim","gt"]):
        header = [h.strip() for h in first.replace("\t", ",").split(",")]
        rows = raw[1:]
    else:
        header = ["rgb","sparse","pseudo","estim","gt"]
        rows = raw

    out = []
    for r in rows:
        cols = [c.strip() for c in r.replace("\t", ",").split(",")]
        while len(cols) < len(header): cols.append("")
        item = dict(zip(header, cols))
        for k in ["rgb","sparse","pseudo","estim","gt"]:
            item[k] = _resolve_path(base_dir, item.get(k,""))
        out.append(item)
    return out

# ----------------- 2) 경로 기반 스캐너 -----------------
def _scan_dir(root: str, pattern: str) -> Dict[str, str]:
    """root 아래 pattern 재귀 검색 → {relpath: fullpath}"""
    root = os.path.abspath(root)
    paths = glob.glob(os.path.join(root, "**", pattern), recursive=True)
    out = {}
    for p in paths:
        if os.path.isdir(p): continue
        rel = os.path.relpath(p, root)  # 디렉터리 구조 유지
        out[rel] = os.path.abspath(p)
    return out

def _scan_dir_stem(root: str, pattern: str) -> Dict[str, str]:
    """파일명(stem) → fullpath (동명파일 충돌 시 마지막만 남음)"""
    root = os.path.abspath(root)
    paths = glob.glob(os.path.join(root, "**", pattern), recursive=True)
    out = {}
    for p in paths:
        if os.path.isdir(p): continue
        stem = os.path.splitext(os.path.basename(p))[0]
        out[stem] = os.path.abspath(p)
    return out

def _common_keys_by_mode(mod_maps: Dict[str, Dict[str,str]], match_mode: str) -> List[str]:
    """modality dict들의 key 교집합 반환"""
    keys_sets = [set(d.keys()) for d in mod_maps.values() if len(d)>0]
    if not keys_sets: return []
    common = set.intersection(*keys_sets)
    return sorted(list(common))

def _index_modality_dirs(dirs: Dict[str,str], pattern: str, match_mode: str) -> Dict[str, Dict[str,str]]:
    """
    dirs: {"rgb_dir": "...", "sparse_dir": "...", "pseudo_dir": "...", "estim_dir": "...", "gt_dir": "..."}
    반환: {"rgb": {key:full}, "sparse":{...}, ...}
    """
    use_stem = (match_mode.lower() == "stem")
    scan = _scan_dir_stem if use_stem else _scan_dir

    out = {}
    for mkey, dkey in [("rgb","rgb_dir"), ("sparse","sparse_dir"), ("pseudo","pseudo_dir"),
                       ("estim","estim_dir"), ("gt","gt_dir")]:
        d = dirs.get(dkey, "")
        if d and os.path.isdir(d):
            out[mkey] = scan(d, pattern)
        else:
            out[mkey] = {}
    return out

def _pair_samples_from_dirs(dirs: Dict[str,str], K: int, seed: int, prefer_all: bool=False) -> List[Dict[str,str]]:
    """
    dirs에서 modality별 파일을 스캔하여 공통 key로 페어링 후 K개 샘플 선택
    prefer_all=True면 K<=0 시 전체 사용
    """
    pattern    = dirs.get("pattern", "*.png")
    match_mode = dirs.get("match_mode", "relpath")

    mods = _index_modality_dirs(dirs, pattern, match_mode)
    # 최소한 rgb, sparse, pseudo, estim 은 있어야 유효
    required = ["rgb","sparse","pseudo","estim"]
    if not all(len(mods[m])>0 for m in required):
        return []

    common = _common_keys_by_mode({k:mods[k] for k in required}, match_mode)
    # gt는 선택
    random.Random(seed).shuffle(common)

    if K is None or K <= 0:
        keys = common if prefer_all else common  # 동일
    else:
        keys = common[:K]

    out = []
    for k in keys:
        rec = {
            "rgb":    mods["rgb"][k],
            "sparse": mods["sparse"][k],
            "pseudo": mods["pseudo"][k],
            "estim":  mods["estim"][k],
            "gt":     mods["gt"].get(k, "") if len(mods.get("gt",{}))>0 else ""
        }
        out.append(rec)
    return out

def build_kshot_from_paths(cfg: Dict[str,Any]) -> Dict[str, List[Dict[str,str]]]:
    """
    cfg['kshot']['train_dirs'] / cfg['kshot']['val_dirs'] 기반으로
    K_train/K_val 샘플 목록을 생성.
    - val_dirs가 비어있으면 train 풀에서 홀드아웃으로 K_val 선택.
    """
    ks = cfg.get("kshot", {})
    seed    = int(ks.get("seed", 1))
    K_train = int(ks.get("K_train", 10))
    K_val   = int(ks.get("K_val", 5))

    train_dirs = ks.get("train_dirs", {})
    val_dirs   = ks.get("val_dirs", {})

    # 1) Train pool
    train_samples_full = _pair_samples_from_dirs(train_dirs, K=0, seed=seed, prefer_all=True)
    if not train_samples_full:
        # 폴백: oneshot 경로를 K_train/1 로 복제
        oneshot = cfg["paths"]
        t_rec = {"rgb": oneshot["rgb"], "sparse": oneshot["sparse"], "pseudo": oneshot["pseudo"],
                 "estim": oneshot["estim"], "gt": oneshot.get("gt","")}
        train_samples = [t_rec for _ in range(max(1, K_train))]
        val_samples   = [t_rec]
        return {"train": train_samples, "val": val_samples}

    # 2) 홀드아웃 또는 별도 val_dirs
    rng = random.Random(seed)
    rng.shuffle(train_samples_full)

    if val_dirs and any(val_dirs.values()):
        train_samples = _pair_samples_from_dirs(train_dirs, K=K_train, seed=seed)
        val_samples   = _pair_samples_from_dirs(val_dirs,   K=K_val,   seed=seed)
    else:
        # 같은 풀에서 분할
        n_tr = K_train if K_train > 0 else len(train_samples_full)
        n_va = K_val   if K_val   > 0 else max(1, int(0.2*len(train_samples_full)))
        n_tr = min(n_tr, len(train_samples_full))
        rest = train_samples_full[n_tr:]
        rng.shuffle(rest)
        n_va = min(n_va, len(rest))
        train_samples = train_samples_full[:n_tr]
        val_samples   = rest[:n_va] if n_va>0 else train_samples_full[-1:]

    return {"train": train_samples, "val": val_samples}

# ----------------- Dataset -----------------
class KShotDataset(Dataset):
    """
    각 샘플을 3D 텐서(C,H,W)로 반환하여 DataLoader가 (B,C,H,W)로 묶이도록 정규화.
    - RGB : (3,H,W)
    - DL/ML/P/E/GT : (1,H,W)
    - 필요 시 참조 해상도(I의 H×W)로 안전하게 리사이즈(깊이류는 nearest, 이미지류는 bilinear)
    """
    def __init__(self, samples, resize_to=None):
        """
        samples: List[Dict[str,str]] with keys: rgb, sparse, pseudo, estim, (optional) gt
        resize_to: (H,W) or None. None이면 RGB의 해상도를 참조 기준으로 사용.
        """
        super().__init__()
        self.samples = samples
        self.resize_to = resize_to  # (H, W) or None

    def __len__(self):
        return len(self.samples)

    # --------- shape helpers ---------
    @staticmethod
    def _as_chw3(x: torch.Tensor) -> torch.Tensor:
        """
        (1,3,H,W) -> (3,H,W), (H,W,3) -> (3,H,W), (3,H,W) 그대로
        """
        if x.dim() == 4 and x.size(0) == 1:  # (1,3,H,W)
            x = x.squeeze(0)
        if x.dim() == 3 and x.size(0) == 3:  # (3,H,W)
            return x.contiguous()
        if x.dim() == 3 and x.size(2) == 3:  # (H,W,3) 방어
            x = x.permute(2, 0, 1)
            return x.contiguous()
        raise RuntimeError(f"RGB tensor must be (1,3,H,W)/(3,H,W)/(H,W,3), got shape={tuple(x.shape)}")

    @staticmethod
    def _as_chw1(x: torch.Tensor) -> torch.Tensor:
        """
        (1,1,H,W) -> (1,H,W), (H,W) -> (1,H,W), (1,H,W) 그대로
        """
        if x.dim() == 4 and x.size(0) == 1:   # (1,1,H,W)
            x = x.squeeze(0)
        if x.dim() == 2:                      # (H,W)
            x = x.unsqueeze(0)
        if x.dim() == 3 and x.size(0) == 1:   # (1,H,W)
            return x.contiguous()
        raise RuntimeError(f"Depth-like tensor must be (1,1,H,W)/(1,H,W)/(H,W), got shape={tuple(x.shape)}")

    # --------- one sample loader ---------
    def _load_one(self, rec):
        # 1) 로드 (현재 load_*는 (1,*,H,W)로 반환)
        I  = load_rgb(rec["rgb"]).float()                 # (1,3,H,W)
        DL = load_depth16_mm_as_m(rec["sparse"]).float()  # (1,1,H,W)
        ML = (DL > 0).float()                             # (1,1,H,W)
        P  = load_pseudo_auto(rec["pseudo"]).float()      # (1,1,H,W) in meters
        E  = load_estimation_8bit_norm(rec["estim"]).float()  # (1,1,H,W) in [0,1]

        if rec.get("gt","") and os.path.isfile(rec["gt"]):
            GT = load_depth16_mm_as_m(rec["gt"]).float()  # (1,1,H,W)
        else:
            GT = torch.zeros_like(DL)

        # 2) 배치 차원 제거 → (C,H,W)
        I  = self._as_chw3(I)   # (3,H,W)
        DL = self._as_chw1(DL)  # (1,H,W)
        ML = self._as_chw1(ML)  # (1,H,W)
        P  = self._as_chw1(P)   # (1,H,W)
        E  = self._as_chw1(E)   # (1,H,W)
        GT = self._as_chw1(GT)  # (1,H,W)

        return I, DL, ML, P, E, GT

    def __getitem__(self, idx):
        return self._load_one(self.samples[idx])
