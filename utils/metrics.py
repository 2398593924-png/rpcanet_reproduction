import torch
import numpy as np
from sklearn.metrics import roc_curve, auc

# Segment
def compute_iou(pred, gt, thresh=0.5, eps=1e-6):
    # 这里官方用的是output > 0
    pred = (torch.sigmoid(pred) > thresh).float()
    gt = gt.float()
    inter = (pred * gt).sum(dim=[1, 2, 3])
    union = pred.sum(dim=[1, 2, 3]) + gt.sum(dim=[1, 2, 3]) - inter
    return ((inter + eps) / (union + eps)).mean().item()


def compute_f1(pred, gt, thresh=0.5, eps=1e-6):
    pred = (torch.sigmoid(pred) > thresh).float()
    gt = gt.float()
    tp = (pred * gt).sum(dim=[1, 2, 3])
    fp = (pred * (1 - gt)).sum(dim=[1, 2, 3])
    fn = ((1 - pred) * gt).sum(dim=[1, 2, 3])
    prec = (tp + eps) / (tp + fp + eps)
    rec  = (tp + eps) / (tp + fn + eps)
    f1 = 2 * prec * rec / (prec + rec + eps)
    return f1.mean().item()


def compute_pd_fa(pred, gt, thresh=0.5):
    pred_bin = (torch.sigmoid(pred) > thresh).float()
    pd_list, fa_list = [], []
    for p, g in zip(pred_bin, gt):
        if g.sum() > 0:
            pd_list.append(1.0 if p.sum() > 0 else 0.0)
        else:
            fa_list.append(1.0 if p.sum() > 0 else 0.0)
    Pd = sum(pd_list) / max(len(pd_list), 1)
    Fa = sum(fa_list) / max(len(fa_list), 1)
    return Pd, Fa


def compute_auc(pred, gt):
    p = pred.detach().flatten().cpu().numpy()
    g = gt.detach().flatten().cpu().numpy().astype(int)
    if g.max() == g.min():
        return 0.5
    fpr, tpr, _ = roc_curve(g, p)
    return auc(fpr, tpr)


def compute_acc_sen_spe(pred, gt, thresh=0.5, eps=1e-6):
    pred = (torch.sigmoid(pred) > thresh).float()
    gt = gt.float()
    tp = (pred * gt).sum().item()
    tn = ((1 - pred) * (1 - gt)).sum().item()
    fp = (pred * (1 - gt)).sum().item()
    fn = ((1 - pred) * gt).sum().item()
    Acc = (tp + tn) / (tp + tn + fp + fn + eps)
    Sen = tp / (tp + fn + eps)
    Spe = tn / (tn + fp + eps)
    return Acc, Sen, Spe


def compute_mae(pred, gt):
    return (pred - gt).abs().mean().item()


def compute_psnr(pred, gt, max_val=1.0, eps=1e-8):
    mse = ((pred - gt) ** 2).mean().item()
    if mse < eps:
        return 100.0
    return 10 * np.log10(max_val ** 2 / mse)


def low_rank_index(B_k):
    if B_k.dim() == 3:
        B_k = B_k.mean(dim=0)
    _, S, _ = torch.linalg.svd(B_k.float(), full_matrices=False)
    return S


def sparsity_rate(O_k, eps=1e-6):
    return (O_k.abs() > eps).float().sum().item() / O_k.numel()


def compute_all(pred_mask, gt_mask, pred_img=None, gt_img=None, thresh=0.5):
    out = {}
    out["IoU"] = compute_iou(pred_mask, gt_mask, thresh)
    out["F1"]  = compute_f1(pred_mask, gt_mask, thresh)
    out["AUC"] = compute_auc(pred_mask, gt_mask)

    if pred_img is not None and gt_img is not None:
        out["MAE"]  = compute_mae(pred_img, gt_img)
        out["PSNR"] = compute_psnr(pred_img, gt_img)

    return out