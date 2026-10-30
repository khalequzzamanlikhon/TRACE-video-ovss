"""Metrics at native resolution.

mIoU   : one confusion matrix over all labelled frames; nanmean over classes with TP+FP+FN > 0.
TC/TA  : motion-compensated consistency between consecutive frames (RAFT warp + forward-backward mask).
         TC = mean over pairs of class-IoU(P_t, warp(P_{t-1})); TA = pixel agreement on valid pixels.
VC_n   : VSPW video consistency on pixels whose GT is constant over n frames.
VCc_n  : (ours) same, but the prediction must also be constant AND correct -> not gameable by constant output.
"""
import numpy as np
import torch


class ConfusionMIoU:
    def __init__(self, num_classes, ignore_index):
        self.C, self.ignore = num_classes, ignore_index
        self.cm = np.zeros((num_classes, num_classes), np.int64)

    def update(self, pred, gt):
        m = (gt != self.ignore) & (gt < self.C)
        idx = self.C * gt[m].astype(np.int64) + pred[m].astype(np.int64)
        self.cm += np.bincount(idx, minlength=self.C ** 2).reshape(self.C, self.C)

    def result(self):
        tp = np.diag(self.cm).astype(np.float64)
        den = self.cm.sum(0) + self.cm.sum(1) - tp
        iou = np.where(den > 0, tp / np.maximum(den, 1), np.nan)
        acc = tp.sum() / max(self.cm.sum(), 1)
        return {'mIoU': float(np.nanmean(iou) * 100), 'aAcc': float(acc * 100),
                'per_class': [None if np.isnan(x) else float(x * 100) for x in iou]}


def pair_consistency(p_t, p_prev_w, valid, num_classes):
    """p_t, p_prev_w: (H,W) long tensors; valid (H,W) bool. Returns (TC_pair, TA_pair) in [0,1]."""
    a, b = p_t[valid], p_prev_w[valid]
    if a.numel() == 0:
        return float('nan'), float('nan')
    oa = torch.nn.functional.one_hot(a.clamp_max(num_classes - 1), num_classes).bool()
    ob = torch.nn.functional.one_hot(b.clamp_max(num_classes - 1), num_classes).bool()
    inter, union = (oa & ob).sum(0).float(), (oa | ob).sum(0).float()
    ious = inter[union > 0] / union[union > 0]
    return ious.mean().item(), (a == b).float().mean().item()


def video_consistency(preds, gts, n, ignore_index):
    """preds, gts: lists of (H,W) uint8 arrays of ONE sequence (consecutive labelled frames).
    Returns (VC_n, VCc_n) in [0,1] averaged over windows, or (nan, nan)."""
    vc, vcc = [], []
    for i in range(len(preds) - n + 1):
        g = np.stack(gts[i:i + n])
        p = np.stack(preds[i:i + n])
        gm = (g == g[0]).all(0) & (g[0] != ignore_index)
        tot = gm.sum()
        if tot == 0:
            continue
        pm = (p == p[0]).all(0)
        vc.append((gm & pm).sum() / tot)
        vcc.append((gm & pm & (p[0] == g[0])).sum() / tot)
    if not vc:
        return float('nan'), float('nan')
    return float(np.mean(vc)), float(np.mean(vcc))
