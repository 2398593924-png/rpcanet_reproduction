import torch
import torch.nn as nn

class SoftIOU(nn.Module):
    def __init__(self, sigma=0.01, eps=1e-6):
        super().__init__()
        self.sigma = sigma
        self.eps = eps

    def forward(self, pred_mask, gt_mask, pred_img, gt_img):
        pred_mask = torch.sigmoid(pred_mask)
        p = pred_mask.reshape(pred_mask.size(0), -1)
        g = gt_mask.reshape(gt_mask.size(0), -1)

        tp = (p * g).sum(dim=1)
        fp = (p * (1 - g)).sum(dim=1)
        fn = ((1 - p) * g).sum(dim=1)

        iou = (tp + self.eps) / (tp + fp + fn + self.eps)
        loss_iou = 1 - iou.mean()

        loss_mse = ((pred_img - gt_img) ** 2).sum(dim=[1, 2, 3]).mean() / (gt_img.shape[2] * gt_img.shape[3])

        return loss_iou + self.sigma * loss_mse