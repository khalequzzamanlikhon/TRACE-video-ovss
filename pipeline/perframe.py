"""how strong can the per-frame training-free model get on KITTI-STEP?

Compares whole-image vs sliding-window inference, softmax(S/tau) vs ProxyCLIP-style attention, DINO token vs
last-block key features, and DINO grid resolution (CLIP values upsampled to the finer DINO grid).

usage: python pipeline/perframe.py --configs '[{...}, ...]' --out results/perframe.json
"""
import argparse
import json
import os
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from tovss.data import build_dataset, load_rgb                              # noqa: E402
from tovss.metrics import ConfusionMIoU                                      # noqa: E402
from tovss.models import Encoders, query_logits_to_class, _norm, IN_MEAN, IN_STD   # noqa: E402
from tovss.text import IMAGENET_TEMPLATES                                    # noqa: E402


class DinoKeyHook:
    """Captures last-block qkv of a DINO/DINOv2 ViT to read the keys (ProxyCLIP uses keys)."""

    def __init__(self, dino):
        self.out = None
        dino.blocks[-1].attn.qkv.register_forward_hook(self._hook)

    def _hook(self, m, i, o):
        self.out = o


def dino_feats(enc, hook, img, grid, kind):
    gh, gw = grid
    p = enc.dino_patch
    x = F.interpolate(img, (p * gh, p * gw), mode='bilinear', align_corners=False)
    x = _norm(x, IN_MEAN, IN_STD)
    n_extra = 1 + getattr(enc.dino, 'num_register_tokens', 0)
    if enc.dino_name.startswith('dinov2'):
        t = enc.dino.forward_features(x)['x_norm_patchtokens']
    else:
        t = enc.dino.get_intermediate_layers(x, n=1)[0][:, 1:]
    if kind == 'key':
        qkv = hook.out                                   # (B, n_extra+N, 3D)
        D = qkv.shape[-1] // 3
        t = qkv[:, n_extra:, D:2 * D]
    return F.normalize(t.float(), dim=-1)                # (B, N, Dd)


def attention(E, mode, tau, beta, gamma):
    S = E @ E.transpose(1, 2)                            # (B,N,N)
    if mode == 'softmax':
        return torch.softmax(S / tau, -1)
    # ProxyCLIP: (S - beta*mean(S)) * gamma, negatives masked
    S = (S - S.mean(dim=(1, 2), keepdim=True) * beta) * gamma
    eye = torch.eye(S.shape[-1], dtype=torch.bool, device=S.device)
    S = S.masked_fill((S < 0) & ~eye, float('-inf')).masked_fill((S < 0) & eye, 0.0)   # keep self: no empty rows
    return torch.softmax(S, -1)


@torch.no_grad()
def window_logits(enc, hook, crops, text, owner, C, c):
    """crops (B,3,h,w) -> class logits (B,C,h',w') on the (possibly upsampled) grid."""
    q, k, v, blk, (gh, gw) = enc.clip_tokens(crops)
    head = c.get('head', 'proxy')
    m = c.get('dino_up', 1) if head == 'proxy' else 1
    fg = (gh * m, gw * m)
    B, D = v.shape[0], v.shape[-1]
    vp = v[:, 1:]
    if m > 1:
        vp = F.interpolate(vp.transpose(1, 2).reshape(B, D, gh, gw), fg, mode='bilinear',
                           align_corners=False).reshape(B, D, -1).transpose(1, 2)
    if head == 'proxy':
        E = dino_feats(enc, hook, crops, fg, c.get('feat', 'tok'))
        A = attention(E, c.get('attn', 'softmax'), c.get('tau', 0.1), c.get('beta', 1.2), c.get('gamma', 3.0))
        vp = A @ vp
    elif head in ('sclip', 'naclip'):
        # SCLIP: correlative self-attention softmax(qq) + softmax(kk); NACLIP: kk + Gaussian neighbourhood prior
        nh = blk.attn.num_heads
        N, dh = vp.shape[1], D // nh
        split = lambda x: x[:, 1:].reshape(B, N, nh, dh).transpose(1, 2)        # (B,nh,N,dh)
        qh, kh, vh = split(q), split(k), split(v)
        sc = dh ** -0.5
        if head == 'sclip':
            A = 0.5 * (torch.softmax(qh @ qh.transpose(-1, -2) * sc, -1) +
                       torch.softmax(kh @ kh.transpose(-1, -2) * sc, -1))
        else:
            ys, xs = torch.meshgrid(torch.arange(gh, device=v.device), torch.arange(gw, device=v.device),
                                    indexing='ij')
            p = torch.stack([ys.flatten(), xs.flatten()], 1).float()
            G = -torch.cdist(p, p).pow(2) / (2 * c.get('sigma', 5.0) ** 2)
            A = torch.softmax(kh @ kh.transpose(-1, -2) * sc + G, -1)
        vp = (A @ vh).transpose(1, 2).reshape(B, N, D)
    z = enc.V.ln_post(blk.attn.out_proj(vp))
    if enc.V.proj is not None:
        z = z @ enc.V.proj
    z = F.normalize(z, dim=-1)
    ql = 100.0 * z @ text.T                                                   # (B,N,Q)
    lg = torch.stack([query_logits_to_class(x, owner, C) for x in ql])        # (B,N,C)
    return lg.transpose(1, 2).reshape(B, C, *fg)


@torch.no_grad()
def predict(enc, hook, img, text, owner, C, c):
    """img (1,3,H,W) at inference resolution -> logits (1,C,H,W)."""
    H, W = img.shape[2:]
    if c.get('mode', 'whole') == 'whole':
        lg = window_logits(enc, hook, img, text, owner, C, c)
        return F.interpolate(lg, (H, W), mode='bilinear', align_corners=False)
    cs, st = c['crop'], c['stride']
    ys = list(range(0, max(H - cs, 0) + 1, st))
    xs = list(range(0, max(W - cs, 0) + 1, st))
    if ys[-1] + cs < H:
        ys.append(H - cs)
    if xs[-1] + cs < W:
        xs.append(W - cs)
    boxes = [(y, x) for y in ys for x in xs]
    out = img.new_zeros(1, C, H, W)
    cnt = img.new_zeros(1, 1, H, W)
    bs = c.get('bs', 16)
    for i in range(0, len(boxes), bs):
        bb = boxes[i:i + bs]
        crops = torch.cat([img[:, :, y:y + cs, x:x + cs] for y, x in bb])
        lg = F.interpolate(window_logits(enc, hook, crops, text, owner, C, c), (cs, cs), mode='bilinear',
                           align_corners=False)
        for j, (y, x) in enumerate(bb):
            out[:, :, y:y + cs, x:x + cs] += lg[j:j + 1]
            cnt[:, :, y:y + cs, x:x + cs] += 1
    return out / cnt


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--configs', required=True)
    ap.add_argument('--split', default='dev')
    ap.add_argument('--fstride', type=int, default=5)
    ap.add_argument('--out', required=True)
    a = ap.parse_args()
    cfgs = json.loads(a.configs)
    ds = build_dataset('kitti_step', a.split)
    assert a.split in ('dev', 'train'), 'exploration never reads val/test'
    frames = [f for _, fr in ds.sequences for f in fr[::a.fstride] if f['gt_path']]
    results = json.load(open(a.out)) if os.path.exists(a.out) else []
    enc, cur_dino, hook = None, None, None
    for c in cfgs:
        dino = c.get('dino', 'dinov2_vits14')
        clip = c.get('clip', 'ViT-B-16')
        if (dino, clip) != cur_dino:
            enc = Encoders(clip, 'openai', dino, 'cuda')
            hook = DinoKeyHook(enc.dino)
            text, owner = enc.encode_queries(ds.queries, IMAGENET_TEMPLATES)
            cur_dino = (dino, clip)
        miou = ConfusionMIoU(len(ds.classes), ds.ignore_index)
        t0 = time.time()
        for f in frames:
            rgb = load_rgb(f['img_path'])
            H, W = rgb.shape[:2]
            s = c['short'] / min(H, W)
            mult = enc.patch
            hi, wi = int(round(H * s / mult)) * mult, int(round(W * s / mult)) * mult
            img = torch.from_numpy(rgb).cuda().permute(2, 0, 1)[None].float() / 255.0
            img = F.interpolate(img, (hi, wi), mode='bilinear', align_corners=False, antialias=True)
            lg = predict(enc, hook, img, text, owner, len(ds.classes), c)
            pred = F.interpolate(lg, (H, W), mode='bilinear', align_corners=False).argmax(1)[0]
            miou.update(pred.to(torch.uint8).cpu().numpy(), ds.load_gt(f))
        r = miou.result()
        row = {'cfg': c, 'mIoU': r['mIoU'], 'aAcc': r['aAcc'],
               'per_class': dict(zip(ds.classes, r['per_class'])), 'n': len(frames),
               'sec_per_frame': (time.time() - t0) / len(frames)}
        results.append(row)
        print(json.dumps({'cfg': c, 'mIoU': round(r['mIoU'], 2), 'spf': round(row['sec_per_frame'], 3)}), flush=True)
        json.dump(results, open(a.out, 'w'), indent=1)


if __name__ == '__main__':
    main()
