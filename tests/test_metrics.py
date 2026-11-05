"""Metric sanity checks on synthetic data (CPU only). Run: python tests/test_metrics.py"""
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from tovss.metrics import ConfusionMIoU, video_consistency, pair_consistency   # noqa: E402
from tovss.flow import backward_warp, fb_valid                                 # noqa: E402


def test_miou():
    rng = np.random.default_rng(0)
    gt = rng.integers(0, 5, (50, 60)).astype(np.uint8)
    gt[:5] = 255
    m = ConfusionMIoU(5, 255)
    m.update(gt.copy(), gt)
    assert abs(m.result()['mIoU'] - 100) < 1e-9
    m2 = ConfusionMIoU(5, 255)
    m2.update(np.zeros_like(gt), gt)
    r = m2.result()
    assert r['mIoU'] < 25, r


def test_vc_constant_not_gameable():
    rng = np.random.default_rng(1)
    gts = [rng.integers(0, 3, (20, 20)).astype(np.uint8)] * 8
    const = [np.zeros((20, 20), np.uint8)] * 8
    vc, vcc = video_consistency(const, gts, 8, 255)
    acc = (gts[0] == 0).mean()
    assert abs(vc - 1.0) < 1e-9, vc               # VC is fooled by constant output
    assert abs(vcc - acc) < 1e-9, (vcc, acc)      # VCc is not


def test_warped_consistency():
    """Scene translating 3 px/frame: same-pixel agreement < 1 but warped TA == 1 on valid pixels."""
    H, W = 40, 64
    base = torch.zeros(H, W, dtype=torch.long)
    base[:, 20:30] = 1
    base[10:20, 40:50] = 2
    prev = base
    cur = torch.roll(base, 3, dims=1)              # content moved right by 3 px
    flow_tp = torch.zeros(1, 2, H, W)
    flow_tp[0, 0] = -3.0                           # pixel x in t comes from x-3 in t-1
    flow_pt = torch.zeros(1, 2, H, W)
    flow_pt[0, 0] = 3.0
    w, inb, grid = backward_warp(prev[None, None].float(), flow_tp, mode='nearest')
    valid = inb & fb_valid(flow_tp, flow_pt, grid)
    tc, ta = pair_consistency(cur, w[0, 0].long(), valid, 3)
    same = (cur == prev).float().mean().item()
    assert same < 0.95 and abs(ta - 1) < 1e-6 and abs(tc - 1) < 1e-6, (same, tc, ta)


if __name__ == '__main__':
    test_miou()
    test_vc_constant_not_gameable()
    test_warped_consistency()
    print('metric tests: PASS')
