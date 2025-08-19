# config/defaults.py
from typing import Dict, Any

DEFAULT_CFG: Dict[str, Any] = {
    "paths": {
        "rgb":      "/home/vip/Desktop/DC/DenseLiDAR/datasets/kitti_raw/train/2011_09_26_drive_0001_sync/proj_depth/image_02/0000000005.png",
        "sparse":   "/home/vip/Desktop/DC/DenseLiDAR/datasets/data_depth_velodyne/train/2011_09_26_drive_0001_sync/proj_depth/velodyne_raw/image_02/0000000005.png",
        "pseudo":   "/home/vip/Desktop/DC/DenseLiDAR/datasets/pseudo_depth_map/train/2011_09_26_drive_0001_sync/proj_depth/velodyne_raw/image_02/0000000005.png",
        "estim":    "/home/vip/Desktop/DC/DenseLiDAR/datasets/kitti_raw_da/train/2011_09_26_drive_0001_sync/proj_depth/image_02/0000000005.png",
        "gt":       "/home/vip/Desktop/DC/DenseLiDAR/datasets/data_depth_annotated/train/2011_09_26_drive_0001_sync/proj_depth/groundtruth/image_02/0000000005.png"
    },
    "train": {
        "epochs": 300,
        "lr": 1e-3,
        "save_dir": "runs_oneshot",
        "tag": "mcprop_1shot"
    },
    "model": {
        "dmax": 80.0,
        "steps": 6,
        "use_residual": True,
        "use_sparse": False,

        # --- anchor(Dirichlet) ---
        "anchor_alpha": 0.7,         # 초기값 (learnable이 False면 고정)
        "anchor_learnable": True,   # True면 학습
        "anchor_mode": "map",     # "scalar" or "map"

        # curvature / geometry
        "kappa_min": 1e-3,
        "kappa_max": 1.0,
        "kernels": [3,5,7],
        "geometry": "hyperbolic"
    },
    "loss": {
        "mu_scaleinv": 0.1,
        "w_lidar": 0.3,
        "w_anchor_reg": 0.0          # >0이면 LiDAR 위치에서 α→1로 유도
    },
    "experiments": [
        {"geometry": "hyperbolic", "tag": "mcprop_1shot_hyp"},
        {"geometry": "elliptic",   "tag": "mcprop_1shot_ellip"}
    ]
}
