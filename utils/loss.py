from typing import Optional

import torch
import torch.nn.functional as F

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

def rmse_mm(pred: torch.Tensor, gt: Optional[torch.Tensor]) -> float:
    if gt is None: return float('nan')
    M = (gt>0)
    if M.sum()==0: return float('nan')
    e2 = (pred[M]-gt[M])**2
    return float(torch.sqrt(e2.mean()).item()*1000.0)