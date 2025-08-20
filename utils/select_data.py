#!/usr/bin/env python3
import argparse, json, os, re, shutil, csv
from pathlib import Path
from collections import defaultdict
from typing import Dict, List, Tuple, Optional

import cv2  # NEW: size check

# 예) 
# python3 utils/select_data.py --rgb-root /home/vip/Desktop/DC/DenseLiDAR/datasets/kitti_raw/train --sparse-root /home/vip/Desktop/DC/DenseLiDAR/datasets/data_depth_velodyne/train --pseudo-root /home/vip/Desktop/DC/DenseLiDAR/datasets/pseudo_depth_map/train --estim-root /home/vip/Desktop/DC/DenseLiDAR/datasets/kitti_raw_da/train --gt-root /home/vip/Desktop/DC/DenseLiDAR/datasets/data_depth_annotated/train --k 100 --out-ds "kitti_k100_dataset"
# --------------------- 경로/유틸 ---------------------

def _norm(p: str) -> str:
    return str(Path(p).resolve()).replace("\\", "/")

def _find_seq_id(parts: Tuple[str, ...]) -> Optional[str]:
    """
    KITTI류 경로에서 시퀀스 ID 추출.
    기본: 'proj_depth' 앞 디렉터리, 없으면 뒤에서 'drive_.*_sync' 탐색.
    """
    try:
        idx = parts.index("proj_depth")
        if idx > 0:
            return parts[idx - 1]
    except ValueError:
        pass
    for p in reversed(parts):
        if re.search(r"drive_.*_sync", p):
            return p
    return None

def _is_image02(parts: Tuple[str, ...]) -> bool:
    return any(p == "image_02" for p in parts)

def _select_min_stem(common_ids: List[str]) -> str:
    numeric = [int(s) for s in common_ids if re.fullmatch(r"\d+", s)]
    if numeric:
        min_int = min(numeric)
        # 원 문자열 반환
        for s in common_ids:
            if s.isdigit() and int(s) == min_int:
                return s
    return sorted(common_ids)[0]

def _collect_by_sequence_image02(root: Path) -> Dict[str, Dict[str, str]]:
    """
    root 아래 *.png 순회, image_02만 선택.
    seq_id -> {frame_stem: full_path}
    """
    seq2frames = defaultdict(dict)
    for p in root.rglob("*.png"):
        parts = p.parts
        if not _is_image02(parts):
            continue
        seq_id = _find_seq_id(parts)
        if not seq_id:
            continue
        stem = p.stem
        seq2frames[seq_id][stem] = str(p.resolve())
    return seq2frames

# ---------- NEW: 해상도 확인 유틸 ----------

def _read_hw(path: str) -> Optional[Tuple[int, int]]:
    """
    이미지의 (H,W) 반환. 읽기 실패 시 None.
    """
    try:
        im = cv2.imread(path, cv2.IMREAD_UNCHANGED)
        if im is None:
            return None
        h, w = im.shape[:2]
        return (h, w)
    except Exception:
        return None

def _meets_size(path: str, req_h: int, req_w: int) -> bool:
    hw = _read_hw(path)
    return (hw is not None) and (hw[0] == req_h) and (hw[1] == req_w)


def _first_common_id_with_size(
    sid: str,
    maps: Dict[str, Dict[str, Dict[str, str]]],
    require_modalities: List[str],
    req_h: int,
    req_w: int
) -> Optional[str]:
    """
    시퀀스 sid에서 모든 모달리티에 공통으로 존재하는 프레임 ID들 중
    (req_h, req_w) 해상도 조건을 '모두' 만족하는 가장 이른(숫자 기준) 프레임의
    '원래 문자열 키'를 반환. 없으면 None.
    """
    # 공통 프레임 집합
    common_ids = None
    for m in require_modalities:
        ids = set(maps[m][sid].keys())
        common_ids = ids if common_ids is None else (common_ids & ids)
    if not common_ids:
        return None

    # 숫자/비숫자 분리: 정렬은 숫자 기준, 하지만 '원래 문자열'을 항상 유지
    numeric_pairs = [(int(s), s) for s in common_ids if re.fullmatch(r"\d+", s)]
    others        = [s for s in common_ids if not re.fullmatch(r"\d+", s)]
    ordered_ids   = [s for _, s in sorted(numeric_pairs, key=lambda x: x[0])] + sorted(others)

    # 순서대로 해상도 조건 검사
    for cand_str in ordered_ids:
        ok = True
        for m in require_modalities:
            p = maps[m][sid].get(cand_str, None)
            if p is None:
                ok = False
                break
            if not _meets_size(p, req_h, req_w):
                ok = False
                break
        if ok:
            return cand_str
    return None

def _common_first_frames(
    across_mod_maps: Dict[str, Dict[str, Dict[str, str]]],
    require_modalities: List[str],
    req_h: int,
    req_w: int
) -> Dict[str, Dict[str, str]]:
    """
    모달리티별 {seq_id -> {stem: path}} → 공통 시퀀스들 중
    해상도 조건을 만족하는 첫 프레임을 선택하여 {seq_id -> {mod: path}} 반환.
    """
    seq_sets = [set(across_mod_maps[m].keys()) for m in require_modalities]
    common_seq_ids = set.intersection(*seq_sets) if seq_sets else set()

    out = {}
    for sid in sorted(common_seq_ids):
        chosen = _first_common_id_with_size(sid, across_mod_maps, require_modalities, req_h, req_w)
        if chosen is None:
            continue
        sample = {m: across_mod_maps[m][sid][chosen] for m in require_modalities}
        out[sid] = sample
    return out

def _fill_to_k(samples_dict: Dict[str, Dict[str, str]], k: int) -> List[Dict[str, str]]:
    """
    samples_dict: {seq_id: {mod: path}}
    k개가 될 때까지 앞에서부터 순환 복제.
    """
    keys = sorted(samples_dict.keys())
    if not keys:
        return []
    out = []
    i = 0
    while len(out) < k:
        sid = keys[i % len(keys)]
        out.append(samples_dict[sid])
        i += 1
    return out

def _extract_seq_and_frame(path: str) -> Tuple[str, str]:
    p = Path(path)
    seq = _find_seq_id(p.parts) or "seq"
    stem = p.stem
    return seq, stem

def _ensure_empty_dir(d: Path, overwrite: bool):
    if d.exists():
        if not overwrite:
            raise FileExistsError(f"Output dir already exists: {d}. Use --overwrite to reuse.")
        # cleanup
        for x in d.iterdir():
            if x.is_symlink() or x.is_file():
                x.unlink(missing_ok=True)
            elif x.is_dir():
                shutil.rmtree(x)
    d.mkdir(parents=True, exist_ok=True)

def _link_or_copy(src: str, dst: str, mode: str = "symlink"):
    Path(dst).parent.mkdir(parents=True, exist_ok=True)
    if mode == "symlink":
        try:
            os.symlink(src, dst)
            return
        except Exception:
            pass  # fallback to copy
    elif mode == "hardlink":
        try:
            os.link(src, dst)
            return
        except Exception:
            pass  # fallback to copy
    shutil.copy2(src, dst)

# --------------------- 샘플 선택 로직 ---------------------

def select_first_k_image02(
    rgb_root: str,
    sparse_root: str,
    pseudo_root: str,
    estim_root: str,
    gt_root: Optional[str],
    k: int,
    req_w: int,
    req_h: int,
) -> Tuple[Dict[str, List[str]], Dict[str, int]]:
    rgb_map    = _collect_by_sequence_image02(Path(rgb_root))
    sparse_map = _collect_by_sequence_image02(Path(sparse_root))
    pseudo_map = _collect_by_sequence_image02(Path(pseudo_root))
    estim_map  = _collect_by_sequence_image02(Path(estim_root))

    maps = {"rgb": rgb_map, "sparse": sparse_map, "pseudo": pseudo_map, "estim": estim_map}
    require = ["rgb", "sparse", "pseudo", "estim"]

    if gt_root:
        gt_map = _collect_by_sequence_image02(Path(gt_root))
        maps["gt"] = gt_map
        require.append("gt")

    # ---- 해상도 조건을 만족하는 '첫' 공통 프레임 선택 ----
    common = _common_first_frames(maps, require, req_h=req_h, req_w=req_w)
    selected = _fill_to_k(common, k)

    out_lists = {m: [] for m in require}
    for item in selected:
        for m in require:
            out_lists[m].append(item[m])

    stats = {
        "n_common_sequences": len(common),
        "n_selected": len(selected),
    }
    return out_lists, stats

# --------------------- 데이터셋 구축 ---------------------

def build_split(out_root: Path,
                prefix: str,
                lists: Dict[str, List[str]],
                link_mode: str = "symlink",
                start_index: int = 0,
                req_w: Optional[int] = None,
                req_h: Optional[int] = None) -> Dict[str, str]:
    """
    out_root/prefix/{rgb,sparse,pseudo,estim,gt}/<index>_<seq>_<frame>.png 생성
    (옵션) req_w/h가 주어지면 여기서도 2차 확인
    """
    n = len(next(iter(lists.values())))
    for m in lists:
        if len(lists[m]) != n:
            raise ValueError(f"Modalities have different counts. '{m}'={len(lists[m])}, expected {n}.")

    mapping_rows = []
    for i in range(n):
        idx = i + start_index
        for m, src in lists.items():
            s = src[i]
            # 2차 크기 검증(안전장치)
            if (req_w is not None) and (req_h is not None):
                if not _meets_size(s, req_h, req_w):
                    raise ValueError(f"[{prefix}] size mismatch at {m}: {s} (required {req_w}x{req_h})")
            seq, stem = _extract_seq_and_frame(s)
            fname = f"{idx:06d}_{seq}_{stem}.png"
            dst = out_root / prefix / m / fname
            _link_or_copy(s, str(dst), mode=link_mode)
            mapping_rows.append({
                "split": prefix, "index": idx, "modality": m,
                "seq": seq, "frame": stem, "src": s, "dst": str(dst)
            })

    # 매핑 CSV/JSON 저장
    manifest_dir = out_root / prefix
    manifest_dir.mkdir(parents=True, exist_ok=True)

    csv_path = manifest_dir / "mapping.csv"
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["split","index","modality","seq","frame","src","dst"])
        w.writeheader()
        for r in mapping_rows: w.writerow(r)

    json_path = manifest_dir / "manifest.json"
    with open(json_path, "w") as f:
        json.dump({"samples": mapping_rows}, f, indent=2)

    return {"csv": str(csv_path), "json": str(json_path)}

# --------------------- CLI ---------------------

def main():
    ap = argparse.ArgumentParser("Build K-shot dataset from KITTI-style roots (image_02 only, size-filtered)")
    # train roots
    ap.add_argument("--rgb-root",    required=True, help="RGB root (e.g., kitti_raw/train)")
    ap.add_argument("--sparse-root", required=True, help="LiDAR sparse root (e.g., data_depth_velodyne/train)")
    ap.add_argument("--pseudo-root", required=True, help="Pseudo depth root")
    ap.add_argument("--estim-root",  required=True, help="Estimation root (8-bit source)")
    ap.add_argument("--gt-root",     default="",   help="GT root (optional)")
    ap.add_argument("--k", type=int, default=10,   help="K shots for train")

    # optional val roots
    ap.add_argument("--val-rgb-root")
    ap.add_argument("--val-sparse-root")
    ap.add_argument("--val-pseudo-root")
    ap.add_argument("--val-estim-root")
    ap.add_argument("--val-gt-root", default="")
    ap.add_argument("--k-val", type=int, default=0, help="K shots for val (0=skip)")

    # size requirement (defaults: 1242x375)
    ap.add_argument("--require-w", type=int, default=1242, help="Required width (default: 1242)")
    ap.add_argument("--require-h", type=int, default=375,  help="Required height (default: 375)")

    # output
    ap.add_argument("--out-ds", required=True, help="Output dataset root directory")
    ap.add_argument("--train-prefix", default="train", help="Subdir name for train split")
    ap.add_argument("--val-prefix",   default="val",   help="Subdir name for val split")
    ap.add_argument("--link-mode", choices=["symlink","hardlink","copy"], default="symlink")
    ap.add_argument("--overwrite", action="store_true", help="Overwrite existing output directory")
    ap.add_argument("--start-index", type=int, default=0, help="Start numbering from this index")
    args = ap.parse_args()

    out_root = Path(args.out_ds)
    _ensure_empty_dir(out_root, overwrite=args.overwrite)

    # ---- train ----
    train_lists, train_stats = select_first_k_image02(
        rgb_root=args.rgb_root,
        sparse_root=args.sparse_root,
        pseudo_root=args.pseudo_root,
        estim_root=args.estim_root,
        gt_root=args.gt_root if args.gt_root else None,
        k=args.k,
        req_w=args.require_w, req_h=args.require_h
    )
    if len(next(iter(train_lists.values()), [])) == 0:
        raise RuntimeError("No train samples matched the size requirement. "
                           f"Check roots or relax --require-w/--require-h.")
    paths_train = build_split(
        out_root, args.train_prefix, train_lists,
        link_mode=args.link_mode, start_index=args.start_index,
        req_w=args.require_w, req_h=args.require_h
    )
    print(f"[OK] Built TRAIN split at: {out_root/args.train_prefix}")
    print(f"  - #common sequences (size-ok): {train_stats['n_common_sequences']}")
    print(f"  - #selected (after fill):     {args.k}")
    print(f"  - manifests: {paths_train['csv']} , {paths_train['json']}")

    # ---- val (optional) ----
    if args.k_val > 0 and all([
        args.val_rgb_root, args.val_sparse_root, args.val_pseudo_root, args.val_estim_root
    ]):
        val_lists, val_stats = select_first_k_image02(
            rgb_root=args.val_rgb_root,
            sparse_root=args.val_sparse_root,
            pseudo_root=args.val_pseudo_root,
            estim_root=args.val_estim_root,
            gt_root=args.val_gt_root if args.val_gt_root else None,
            k=args.k_val,
            req_w=args.require_w, req_h=args.require_h
        )
        if len(next(iter(val_lists.values()), [])) == 0:
            print("[WARN] No val samples matched the size requirement; skipping val build.")
        else:
            start_val = args.start_index + args.k
            paths_val = build_split(
                out_root, args.val_prefix, val_lists,
                link_mode=args.link_mode, start_index=start_val,
                req_w=args.require_w, req_h=args.require_h
            )
            print(f"[OK] Built VAL split at: {out_root/args.val_prefix}")
            print(f"  - #common sequences (size-ok): {val_stats['n_common_sequences']}")
            print(f"  - #selected (after fill):     {args.k_val}")
            print(f"  - manifests: {paths_val['csv']} , {paths_val['json']}")
    else:
        print("[INFO] Val split skipped (provide --val-*-root and --k-val > 0 to build).")

    # 최상위 요약 저장
    summary = {
        "out_root": _norm(str(out_root)),
        "link_mode": args.link_mode,
        "train": {"k": args.k, "prefix": args.train_prefix},
        "val": {"k": args.k_val, "prefix": args.val_prefix} if args.k_val > 0 else None,
        "require_size": {"w": args.require_w, "h": args.require_h}
    }
    with open(out_root / "dataset_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(f"[DONE] Summary: {out_root/'dataset_summary.json'}")

if __name__ == "__main__":
    main()
