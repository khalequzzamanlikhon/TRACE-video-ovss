"""Per-frame scores and flow-aligned temporal aggregation (RAFT-EMA and the bidirectional flow smoother).
Per-frame logits and RAFT flows are cached per sequence (half native resolution), then the aggregation rules
are evaluated on the same cache.

Aggregators (X_t = per-frame class map, in prob or logit space; w_t = per-pixel weight):
  none     : X_t
  causal   : F_t = w_t X_t + g * valid * warp(F_{t-1});  normalised by the same recursion on w  (RAFT-EMA)
  bidir    : causal + anti-causal recursion (fixed-interval smoother), lag = unlimited or `lag` frames
w_t = 1 (plain) or the per-pixel confidence 1 - H(P_t)/log C (conf).

usage: python pipeline/temporal.py --pf '{...}' --aggs '[{...}]' --out results/temporal.json
"""
import argparse
import hashlib
import json
import math
import os
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from perframe import DinoKeyHook, predict                       # noqa: E402
from tovss.data import build_dataset, load_rgb                           # noqa: E402
from tovss.flow import Raft, backward_warp, fb_valid                     # noqa: E402
from tovss.metrics import ConfusionMIoU, video_consistency               # noqa: E402
from tovss.models import Encoders                                        # noqa: E402
from tovss.text import IMAGENET_TEMPLATES                                # noqa: E402


@torch.no_grad()
def cache_sequence(enc, hook, raft, text, owner, C, frames, pf, scale):
    """Returns logits list [(C,h,w) half], flows list of (f_t->t-1, f_t-1->t) for t>=1, native size."""
    logits, flows, prev = [], [None], None
    for f in frames:
        rgb = load_rgb(f['img_path'])
        H, W = rgb.shape[:2]
        h, w = int(round(H * scale)), int(round(W * scale))
        img = torch.from_numpy(rgb).cuda().permute(2, 0, 1)[None].float() / 255.0
        s = pf['short'] / min(H, W)
        hi, wi = int(round(H * s / enc.patch)) * enc.patch, int(round(W * s / enc.patch)) * enc.patch
        x = F.interpolate(img, (hi, wi), mode='bilinear', align_corners=False, antialias=True)
        lg = predict(enc, hook, x, text, owner, C, pf)
        logits.append(F.interpolate(lg, (h, w), mode='bilinear', align_corners=False)[0].half())
        small = F.interpolate(img, (h, w), mode='bilinear', align_corners=False, antialias=True)
        if prev is not None:
            flows.append((raft(small, prev), raft(prev, small)))
        prev = small
    return logits, flows, (H, W)


def to_space(lg, a):
    lg = lg.float()
    if a.get('space', 'prob') == 'prob':
        return torch.softmax(lg / a.get('T', 1.0), 0)
    return lg / 100.0


def weight(lg, a):
    if not a.get('conf', False):
        return torch.ones_like(lg[:1].float())
    P = torch.softmax(lg.float(), 0)
    H = -(P.clamp_min(1e-8) * P.clamp_min(1e-8).log()).sum(0, keepdim=True)
    return (1 - H / math.log(P.shape[0])).clamp_min(1e-3) ** a.get('conf_pow', 1.0)


def warp_valid(x, f_back, f_fwd):
    """x defined in the other frame; f_back: flow cur->other, f_fwd: other->cur. Returns warped x, valid."""
    xw, inb, grid = backward_warp(x[None], f_back, mode='bilinear')
    return xw[0], (inb & fb_valid(f_back, f_fwd, grid)).float()[None]


def magnification(fb):
    """Local area ratio of the warp x -> x + fb(x): >1 where the other frame saw the surface larger."""
    u, v = fb[0, 0], fb[0, 1]
    du_dy, du_dx = torch.gradient(u)
    dv_dy, dv_dx = torch.gradient(v)
    return ((1 + du_dx) * (1 + dv_dy) - du_dy * dv_dx)[None]


def recursion(logits, a, flows, g, order):
    """Normalised recursive aggregation along `order` (list of frame indices).
    Yields (t, S_t, Z_t) on GPU; only the previous state is kept on GPU.
    a['zoom'] = kappa > 0: zoom-aware weighting, the propagated evidence is scaled by magnification**kappa."""
    prev, S, Z = None, None, None
    for t in order:
        x, w = to_space(logits[t], a), weight(logits[t], a)
        s, z = w * x, w
        if prev is not None:
            if prev == t - 1:          # forward: other = t-1, flows[t] = (t->t-1, t-1->t)
                fb, ff = flows[t]
            else:                      # backward: other = t+1, flows[t+1] = (t+1->t, t->t+1)
                ff, fb = flows[t + 1]
            fb, ff = fb.float(), ff.float()    # flows may be stored in fp16
            sw, v = warp_valid(S, fb, ff)
            zw, _ = warp_valid(Z, fb, ff)
            if a.get('zoom'):
                m = F.avg_pool2d(magnification(fb)[None], 5, 1, 2)[0].clamp(0.25, 4.0)
                v = v * m ** a['zoom']
            s, z = s + g * v * sw, z + g * v * zw
        S, Z, prev = s, z, t
        yield t, S, Z


def prior_em(logits, a):
    """Video-level label-shift correction (Saerens et al. 2002 EM) on the per-frame posteriors.
    Returns logits + strength * log(pi / pi0); online=True uses only frames seen so far."""
    T = a.get('em_T', 1.0)                 # posterior temperature: CLIP's x100 logits are near one-hot
    sub = [torch.softmax(lg.float()[:, ::8, ::8] / T, 0).flatten(1) for lg in logits]
    s = a.get('prior_em', 1.0)
    out = []
    for t, lg in enumerate(logits):
        P = torch.cat(sub[:t + 1] if a.get('online') else sub, 1)          # (C, M)
        pi0 = torch.full_like(P[:, 0], 1.0 / P.shape[0])   # zero-shot classifier: no training prior -> uniform
        pi = P.mean(1).clamp_min(1e-6)
        for _ in range(a.get('em_iters', 10)):
            q = P * (pi / pi0)[:, None]
            q = q / q.sum(0, keepdim=True)
            pi = q.mean(1).clamp_min(1e-6)
        out.append((lg.float() + T * s * torch.log(pi / pi0)[:, None, None]).half())
        if not a.get('online'):
            corr = out[-1].float() - lg.float()
            return [(l.float() + corr).half() for l in logits]
    return out


def aggregate(logits, flows, a):
    """Yields the aggregated class map of every frame, in frame order, on GPU."""
    if a.get('prior_em'):
        logits = prior_em(logits, a)
    n = len(logits)
    if a['type'] == 'none':
        for lg in logits:
            yield lg.float()
        return
    g = a['g']
    if a['type'] == 'causal':
        for _, s, z in recursion(logits, a, flows, g, list(range(n))):
            yield s / z
        return
    bwd = [None] * n                    # anti-causal sums kept on CPU in fp16
    for t, s, z in recursion(logits, a, flows, a.get('gb', g), list(range(n - 1, -1, -1))):
        bwd[t] = (s.half().cpu(), z.half().cpu())
    for t, s, z in recursion(logits, a, flows, g, list(range(n))):
        sb, zb = bwd[t][0].cuda().float(), bwd[t][1].cuda().float()
        bwd[t] = None
        x, w = to_space(logits[t], a), weight(logits[t], a)
        yield (s + sb - w * x) / (z + zb - w)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--pf', required=True, help='per-frame config (see pipeline/perframe.py)')
    ap.add_argument('--aggs', required=True)
    ap.add_argument('--split', default='dev')
    ap.add_argument('--seqs', default=None)
    ap.add_argument('--max_frames', type=int, default=None)
    ap.add_argument('--scale', type=float, default=0.5)
    ap.add_argument('--out', required=True)
    ap.add_argument('--cache_dir', default=None)
    ap.add_argument('--final', default=None, help='frozen config file (committed); required for val/test')
    ap.add_argument('--dataset', default='kitti_step')
    a = ap.parse_args()
    if a.dataset != 'kitti_step' or a.split not in ('dev', 'train'):
        assert a.final and os.path.exists(a.final), 'val/test only with a committed frozen config (--final)'
        assert json.loads(a.pf) == json.load(open(a.final))['pf'], 'per-frame config must match the frozen one'
    pf, aggs = json.loads(a.pf), json.loads(a.aggs)
    kw = {'seqs': a.seqs.split(',')} if a.seqs else {}
    ds = build_dataset(a.dataset, a.split, **kw)
    C = len(ds.classes)
    enc = Encoders(pf.get('clip', 'ViT-B-16'), 'openai', pf.get('dino', 'dinov2_vits14'), 'cuda')
    hook = DinoKeyHook(enc.dino)
    text, owner = enc.encode_queries(ds.queries, IMAGENET_TEMPLATES)
    raft = Raft('cuda')
    mi = [ConfusionMIoU(C, ds.ignore_index) for _ in aggs]
    vcs = [[] for _ in aggs]
    t0 = time.time()
    for seq, frames in ds.sequences:
        frames = frames[:a.max_frames]
        cpath = None
        if a.cache_dir:
            tag = hashlib.md5(json.dumps([pf, a.scale, a.max_frames], sort_keys=True).encode()).hexdigest()[:8]
            cpath = os.path.join(a.cache_dir, f'{seq}_{tag}.pt')
        if cpath and os.path.exists(cpath):
            d = torch.load(cpath, map_location='cuda')
            logits, flows, (H, W) = d['logits'], d['flows'], d['size']
        else:
            logits, flows, (H, W) = cache_sequence(enc, hook, raft, text, owner, C, frames, pf, a.scale)
            if cpath:
                os.makedirs(a.cache_dir, exist_ok=True)
                torch.save({'logits': logits, 'flows': [None] + [(x.half(), y.half()) for x, y in flows[1:]],
                            'size': (H, W)}, cpath)
        flows = [None] + [(x.float(), y.float()) for x, y in flows[1:]]
        gts = [ds.load_gt(f) if f['gt_path'] else None for f in frames]
        for i, ag in enumerate(aggs):
            preds = [F.interpolate(o[None], (H, W), mode='bilinear', align_corners=False).argmax(1)[0]
                     .to(torch.uint8).cpu().numpy() for o in aggregate(logits, flows, ag)]
            for p, g in zip(preds, gts):
                if g is not None:
                    mi[i].update(p, g)
            for s in range(0, len(preds) - 7):
                if all(g is not None for g in gts[s:s + 8]):
                    vc, vcc = video_consistency(preds[s:s + 8], gts[s:s + 8], 8, ds.ignore_index)
                    if not np.isnan(vc):
                        vcs[i].append((vc, vcc))
        print(f'seq {seq} done ({len(frames)} frames, {time.time() - t0:.0f}s)', flush=True)
        del logits, flows
        torch.cuda.empty_cache()
    rows = []
    for ag, m, v in zip(aggs, mi, vcs):
        r = m.result()
        rows.append({'pf': pf, 'agg': ag, 'mIoU': r['mIoU'], 'per_class': dict(zip(ds.classes, r['per_class'])),
                     'VC8': float(np.mean([x[0] for x in v]) * 100), 'VCc8': float(np.mean([x[1] for x in v]) * 100)})
        print(json.dumps({'agg': ag, 'mIoU': round(r['mIoU'], 2), 'VC8': round(rows[-1]['VC8'], 2),
                          'VCc8': round(rows[-1]['VCc8'], 2)}), flush=True)
    old = json.load(open(a.out)) if os.path.exists(a.out) else []
    json.dump(old + rows, open(a.out, 'w'), indent=1)


if __name__ == '__main__':
    main()
