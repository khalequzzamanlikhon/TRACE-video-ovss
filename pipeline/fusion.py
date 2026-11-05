"""Fuse dense training-free OVSS logits with SAM2 space-time tracks.

For pixel x of frame t covered by track k (id map from pipeline/tracks.py), with dense map D_t (per-frame or the
bidirectional RAFT smoother of pipeline/temporal.py):
  log P(x) = log softmax(D_t(x)) + a * log p_pool(k) + b * log p_crop(k)
  p_pool(k): dense probabilities averaged over track k in the current frame ('frame') or over all its frames ('tube')
  p_crop(k): CLIP image embeddings of the K largest views of k (masked / context crops), area-weighted, vs text
Uncovered pixels keep the dense prediction. Everything runs at the cache resolution (half native); id maps are
nearest-downsampled; the fused map is upsampled to native resolution for evaluation.

Temporal vocabulary pruning ('presence'): per clip, a class is present if some track's multi-view crop-CLIP
probability for it reaches `thr` (src 'crop'), or if it occurs in the GT of the clip (src 'oracle', upper bound).
Evidence is pooled over +-`win` neighbouring clips. Absent classes get `pen` added to their log-probabilities
before pooling.

  python pipeline/fusion.py --pf '{...}' --cfgs '[...]' --out results/x.json
"""
import argparse
import hashlib
import json
import os
import sys

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from temporal import aggregate                                   # noqa: E402
from tovss.data import build_dataset                                     # noqa: E402
from tovss.metrics import ConfusionMIoU, video_consistency               # noqa: E402
from tovss.text import IMAGENET_TEMPLATES                                # noqa: E402


@torch.no_grad()
def text_embeddings(queries):
    import open_clip
    model, _, _ = open_clip.create_model_and_transforms('ViT-B-16', pretrained='openai')
    model = model.cuda().eval()
    tok = open_clip.get_tokenizer('ViT-B-16')
    embs, owner = [], []
    for c, names in enumerate(queries):
        for n in names:
            e = F.normalize(model.encode_text(tok([t.format(n) for t in IMAGENET_TEMPLATES]).cuda()).float(), dim=-1)
            embs.append(F.normalize(e.mean(0), dim=-1))
            owner.append(c)
    return torch.stack(embs), torch.tensor(owner, device='cuda')


def crop_probs(tracks, T, owner, C, c):
    """(max_id+1, C) crop-CLIP class probabilities per track (row 0 and missing tracks: uniform)."""
    n = max(tracks) + 1 if tracks else 1
    P = torch.full((n, C), 1.0 / C, device='cuda')
    kind, alpha = c.get('crop', 'both'), c.get('view_alpha', 0.5)
    gids, L = [], []
    for gid, t in tracks.items():
        if 'emb_mask' not in t:
            continue
        w = torch.tensor([ar for _, ar in t['views']], device='cuda').float() ** alpha
        w = (w / w.sum())[:, None]
        e = {'mask': t['emb_mask'], 'ctx': t['emb_ctx']}.get(kind)
        e = (t['emb_mask'].float() + t['emb_ctx'].float()) if e is None else e.float()
        e = F.normalize((w * F.normalize(e.cuda(), dim=-1)).sum(0), dim=-1)
        ql = c.get('crop_scale', 100.0) * e @ T.T
        L.append(torch.full((C,), float('-inf'), device='cuda').scatter_reduce(0, owner, ql, reduce='amax'))
        gids.append(gid)
    if gids:
        L = torch.stack(L)
        if c.get('crop_center'):      # video-level class debiasing: remove each class's mean logit over all tracks
            L = L - c['crop_center'] * L.mean(0, keepdim=True)
        P[torch.tensor(gids, device='cuda')] = torch.softmax(L, 1)
    if c.get('lp') and c['lp'].get('target', 'crop') in ('crop', 'both'):
        P = graph_propagate(P, tracks, c['lp'])
    return P


def track_descriptors(tracks):
    """Visual descriptor per track: normalised mean of its (masked + context) crop embeddings."""
    gids, X = [], []
    for gid, t in tracks.items():
        if 'emb_mask' in t:
            e = F.normalize(t['emb_mask'].float(), dim=-1) + F.normalize(t['emb_ctx'].float(), dim=-1)
            X.append(F.normalize(e.mean(0), dim=-1))
            gids.append(gid)
    return gids, (torch.stack(X).cuda() if X else None)


def graph_propagate(P, tracks, lp):
    """Track-graph label propagation over all tracks of the video: kNN graph on track descriptors,
    P <- (1 - alpha) P0 + alpha W P (iterated). Rows of tracks without descriptors are unchanged."""
    gids, X = track_descriptors(tracks)
    if X is None or len(gids) < 3:
        return P
    g = torch.tensor(gids, device='cuda')
    S = X @ X.T
    S.fill_diagonal_(float('-inf'))
    k = min(lp.get('k', 10), len(gids) - 1)
    val, idx = S.topk(k, dim=1)
    W = torch.zeros_like(S).scatter_(1, idx, torch.softmax(val / lp.get('tau', 0.05), 1))
    P0 = P[g].clone()
    Q = P0.clone()
    for _ in range(lp.get('iters', 10)):
        Q = (1 - lp['alpha']) * P0 + lp['alpha'] * W @ Q
    P = P.clone()
    P[g] = Q / Q.sum(1, keepdim=True)
    return P


def causal_crop_tables(tracks, T, owner, C, c, n_frames):
    """Online crop-CLIP: for each frame t, (gids, probs) of tracks alive at t, using only views at frames <= t
    (the K largest seen so far, area-weighted). Requires tracks built with pipeline/tracks.py --causal."""
    kind, alpha, K, scale = c.get('crop', 'both'), c.get('view_alpha', 0.5), c.get('K', 5), c.get('crop_scale', 100.0)
    per_frame = [([], []) for _ in range(n_frames)]

    def embs(t):
        em, ec = F.normalize(t['emb_mask'].float().cuda(), dim=-1), F.normalize(t['emb_ctx'].float().cuda(), dim=-1)
        e_all = {'mask': em, 'ctx': ec}.get(kind, None)
        return F.normalize(em + ec, dim=-1) if e_all is None else e_all

    def class_logits(e):
        return torch.full((C,), float('-inf'), device='cuda').scatter_reduce(0, owner, scale * e @ T.T, reduce='amax')

    # online class debiasing: mean first-view logits of all tracks from EARLIER clips (strictly causal;
    # the first clip has no history and is not debiased)
    mu = {}
    if c.get('crop_center'):
        first = {}
        for t in tracks.values():
            if 'emb_mask' in t:
                j0 = min(range(len(t['views'])), key=lambda j: t['views'][j][0])
                first.setdefault(t['clip'], []).append(class_logits(embs(t)[j0]))
        run, cnt = torch.zeros(C, device='cuda'), 0
        for clip0 in sorted({t['clip'] for t in tracks.values()}):
            mu[clip0] = run / cnt if cnt else torch.zeros(C, device='cuda')
            for lg in first.get(clip0, []):
                run, cnt = run + lg, cnt + 1
    for gid, t in tracks.items():
        if 'emb_mask' not in t:
            continue
        e_all = embs(t)
        views = sorted(range(len(t['views'])), key=lambda j: t['views'][j][0])
        last_f = t['clip'] + max(fi for fi, _ in t['views'])
        for n in range(len(views)):
            seen = views[:n + 1]
            top = sorted(seen, key=lambda j: -t['views'][j][1])[:K]
            w = torch.tensor([t['views'][j][1] for j in top], device='cuda').float() ** alpha
            e = F.normalize(((w / w.sum())[:, None] * e_all[top]).sum(0), dim=-1)
            cl = class_logits(e)
            if c.get('crop_center'):
                cl = cl - c['crop_center'] * mu[t['clip']]
            p = torch.softmax(cl, 0)
            f0 = t['clip'] + t['views'][views[n]][0]
            f1 = t['clip'] + t['views'][views[n + 1]][0] if n + 1 < len(views) else last_f + 1
            for f in range(f0, min(f1, n_frames)):
                per_frame[f][0].append(gid)
                per_frame[f][1].append(p)
    return [(torch.tensor(g, device='cuda', dtype=torch.long), torch.stack(p)) if g else None for g, p in per_frame]


def presence_penalty(n_frames, clip_len, tracks, T, owner, gts, C, pres, logP=None):
    """Per-frame additive log-prob penalty (C,) for classes judged absent in the frame's clip window.
    src 'area': S = fraction of clip pixels whose dense argmax is c; mode 'prior' adds kappa * log(S + eps)."""
    n_clips = (n_frames + clip_len - 1) // clip_len
    S = torch.zeros(n_clips, C, device='cuda')
    if pres['src'] == 'area':
        for t, lp in enumerate(logP):
            S[t // clip_len] += torch.bincount(lp.argmax(0).flatten(), minlength=C).float()
        S = S / S.sum(1, keepdim=True)
        if pres.get('mode') == 'prior':
            w = pres.get('win', 0)
            out = []
            for t in range(n_frames):
                ci = t // clip_len
                s = S[max(0, ci - w):ci + w + 1].mean(0)
                out.append(pres.get('kappa', 0.2) * torch.log(s + pres.get('eps', 1e-4)))
            return out
    elif pres['src'] == 'oracle':
        for t, g in enumerate(gts):
            if g is not None:
                cnt = np.bincount(g[g < C].astype(np.int64), minlength=C)
                S[t // clip_len] = torch.maximum(S[t // clip_len],
                                                 torch.from_numpy(cnt >= pres.get('min_px', 50)).float().cuda())
    else:
        P = crop_probs(tracks, T, owner, C, {'crop': pres.get('crop', 'mask'),
                                             'crop_scale': pres.get('crop_scale', 100.0)})
        for gid, t in tracks.items():
            if 'emb_mask' in t and t['total_area'] >= pres.get('min_area', 0):
                ci = t['clip'] // clip_len
                S[ci] = torch.maximum(S[ci], P[gid])
    w, thr, pen = pres.get('win', 0), pres.get('thr', 0.5), pres.get('pen', -10.0)
    out = []
    for t in range(n_frames):
        ci = t // clip_len
        s = S[max(0, ci - w):ci + w + 1].max(0).values
        out.append(torch.where(s >= thr, torch.zeros_like(s), torch.full_like(s, pen)))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--pf', required=True)
    ap.add_argument('--cfgs', required=True)
    ap.add_argument('--split', default='dev')
    ap.add_argument('--scale', type=float, default=0.5)
    ap.add_argument('--cache_dir', default='cache')
    ap.add_argument('--tracks', default='cache/tracks')
    ap.add_argument('--out', required=True)
    ap.add_argument('--clip_len', type=int, default=20, help='must match pipeline/tracks.py')
    ap.add_argument('--final', default=None, help='frozen config file (committed); required for val/test')
    ap.add_argument('--save_root', default=None, help='write results/<name>/{preds,metrics.json} per cfg')
    ap.add_argument('--dataset', default='kitti_step')
    a = ap.parse_args()
    if a.dataset != 'kitti_step' or a.split not in ('dev', 'train'):
        assert a.final and os.path.exists(a.final), 'val/test only with a committed frozen config (--final)'
        frozen = json.load(open(a.final))
        assert json.loads(a.pf) == frozen['pf'] and all(c in frozen['cfgs'] for c in json.loads(a.cfgs)), \
            'cfgs must match the frozen config file'
    pf, cfgs = json.loads(a.pf), json.loads(a.cfgs)
    ds = build_dataset(a.dataset, a.split)
    C = len(ds.classes)
    T, owner = text_embeddings(ds.queries)
    tag = hashlib.md5(json.dumps([pf, a.scale, None], sort_keys=True).encode()).hexdigest()[:8]
    mi = [ConfusionMIoU(C, ds.ignore_index) for _ in cfgs]
    vcs = [[] for _ in cfgs]
    cover = []
    for seq, frames in ds.sequences:
        d = torch.load(os.path.join(a.cache_dir, f'{seq}_{tag}.pt'), map_location='cuda')
        logits, (H, W) = d['logits'], d['size']
        flows = d['flows']                 # fp16; converted per frame (long sequences do not fit in fp32)
        h, w = logits[0].shape[1:]
        tracks = torch.load(os.path.join(a.tracks, seq, 'tracks.pt'))
        ids = [torch.from_numpy(np.array(Image.open(os.path.join(a.tracks, seq, f'ids_{f["name"]}.png'))).astype(np.int64))
               .cuda() for f in frames]
        ids = [F.interpolate(x[None, None].float(), (h, w), mode='nearest')[0, 0].long() for x in ids]
        cover.append(float(np.mean([(x > 0).float().mean().item() for x in ids])))
        n_ids = max(max(tracks) if tracks else 0, max(int(x.max()) for x in ids)) + 1
        gts = [ds.load_gt(f) if f['gt_path'] else None for f in frames]
        dense_cache = {}
        for i, c in enumerate(cfgs):
            dk = json.dumps(c.get('dense', {'type': 'none'}), sort_keys=True)
            if dk not in dense_cache:
                dense_cache.clear()
                dense_cache[dk] = [x.half() for x in aggregate(logits, flows, {**json.loads(dk), 'space': 'logit'})] \
                    if json.loads(dk)['type'] != 'none' else logits
            D = dense_cache[dk]
            sc = 100.0 if json.loads(dk)['type'] != 'none' else 1.0     # logit-space smoother returns logits / 100
            if c.get('dense_center'):   # video-level class debiasing of the raw dense logits
                mu = torch.stack([x.float().mean((1, 2)) for x in D]).mean(0) * sc
                logP = [torch.log_softmax(x.float() * sc - c['dense_center'] * mu[:, None, None], 0).half() for x in D]
            else:
                logP = [torch.log_softmax(x.float() * sc, 0).half() for x in D]
            if c.get('presence'):
                pen = presence_penalty(len(logP), a.clip_len, tracks, T, owner, gts, C, c['presence'], logP)
                logP = [torch.log_softmax(lp.float() + p[:, None, None], 0).half() for lp, p in zip(logP, pen)]
            pool, causal = c.get('pool', 'none'), c.get('causal', False)
            if causal:
                assert json.loads(dk)['type'] in ('none', 'causal') and not c.get('post'), 'online cfg uses future frames'
            if pool == 'tube':
                sums = torch.zeros(n_ids, C, device='cuda')
                if not causal:
                    for lp, idm in zip(logP, ids):
                        sums.index_add_(0, idm.flatten(), lp.float().exp().flatten(1).T)
                    p_tube = sums / sums.sum(1, keepdim=True).clamp_min(1e-8)
                    if c.get('lp') and c['lp'].get('target', 'crop') in ('tube', 'both'):
                        p_tube = graph_propagate(p_tube, tracks, c['lp'])
            if c.get('b', 0) > 0:
                if causal:
                    crop_tab = causal_crop_tables(tracks, T, owner, C, c, len(logP))
                else:
                    p_crop = crop_probs(tracks, T, owner, C, c)
                    if p_crop.shape[0] < n_ids:
                        p_crop = torch.cat([p_crop, torch.full((n_ids - p_crop.shape[0], C), 1.0 / C, device='cuda')])
            fused = []
            for t, (lp, idm) in enumerate(zip(logP, ids)):
                lp = lp.float()
                out = lp.clone()
                cov = (idm > 0)[None]
                if pool == 'frame':
                    s = torch.zeros(n_ids, C, device='cuda').index_add_(0, idm.flatten(), lp.exp().flatten(1).T)
                    pp = s / s.sum(1, keepdim=True).clamp_min(1e-8)
                    out = out + c['a'] * torch.where(cov, pp.clamp_min(1e-8).log()[idm].permute(2, 0, 1), 0.0)
                elif pool == 'tube':
                    if causal:      # running tube: frames <= t only
                        sums.index_add_(0, idm.flatten(), lp.exp().flatten(1).T)
                        p_tube = sums / sums.sum(1, keepdim=True).clamp_min(1e-8)
                    out = out + c['a'] * torch.where(cov, p_tube.clamp_min(1e-8).log()[idm].permute(2, 0, 1), 0.0)
                if c.get('b', 0) > 0:
                    if causal:
                        p_crop = torch.full((n_ids, C), 1.0 / C, device='cuda')
                        if crop_tab[t] is not None:
                            p_crop[crop_tab[t][0]] = crop_tab[t][1]
                    out = out + c['b'] * torch.where(cov, p_crop.clamp_min(1e-8).log()[idm].permute(2, 0, 1), 0.0)
                fused.append(out.half())
            if c.get('post'):
                fused = [x.half() for x in aggregate(fused, flows, {**c['post'], 'space': 'prob'})]
            preds = [F.interpolate(o[None].float(), (H, W), mode='bilinear', align_corners=False).argmax(1)[0]
                     .to(torch.uint8).cpu().numpy() for o in fused]
            for p, g in zip(preds, gts):
                if g is not None:
                    mi[i].update(p, g)
            if a.save_root and c.get('name'):
                pd = os.path.join(a.save_root, c['name'], 'preds', seq)
                os.makedirs(pd, exist_ok=True)
                for p, f in zip(preds, frames):
                    Image.fromarray(p).save(os.path.join(pd, f['name'] + '.png'))
            for s0 in range(0, len(preds) - 7):
                if all(g is not None for g in gts[s0:s0 + 8]):
                    vc, vcc = video_consistency(preds[s0:s0 + 8], gts[s0:s0 + 8], 8, ds.ignore_index)
                    if not np.isnan(vc):
                        vcs[i].append((vc, vcc))
        print(f'seq {seq} done, track coverage {cover[-1]:.3f}', flush=True)
        del d, logits, flows, ids, dense_cache
        torch.cuda.empty_cache()
    rows = []
    for c, m, v in zip(cfgs, mi, vcs):
        r = m.result()
        rows.append({'pf': pf, 'cfg': c, 'mIoU': r['mIoU'], 'per_class': dict(zip(ds.classes, r['per_class'])),
                     'VC8': float(np.mean([x[0] for x in v]) * 100), 'VCc8': float(np.mean([x[1] for x in v]) * 100),
                     'coverage': cover})
        print(json.dumps({'cfg': c, 'mIoU': round(r['mIoU'], 2), 'VC8': round(rows[-1]['VC8'], 2),
                          'VCc8': round(rows[-1]['VCc8'], 2)}), flush=True)
        if a.save_root and c.get('name'):
            import subprocess
            git = subprocess.run(['git', 'rev-parse', 'HEAD'], capture_output=True, text=True).stdout.strip()
            dirty = bool(subprocess.run(['git', 'status', '--porcelain', '--', 'pipeline', 'tovss', 'tools'],
                                        capture_output=True, text=True).stdout.strip())
            json.dump({'exp': c['name'], 'dataset': a.dataset, 'split': a.split, 'stride': 1,
                       'seqs': [s for s, _ in ds.sequences], 'cfg': c, 'pf': pf, 'tracks': a.tracks,
                       'mIoU': r['mIoU'], 'per_class': rows[-1]['per_class'],
                       'VC': {'8': rows[-1]['VC8']}, 'VCc': {'8': rows[-1]['VCc8']}, 'coverage': cover,
                       'git': git, 'git_dirty': dirty, 'frozen': a.final},
                      open(os.path.join(a.save_root, c['name'], 'metrics.json'), 'w'), indent=2)
    old = json.load(open(a.out)) if os.path.exists(a.out) else []
    json.dump(old + rows, open(a.out, 'w'), indent=1)


if __name__ == '__main__':
    main()
