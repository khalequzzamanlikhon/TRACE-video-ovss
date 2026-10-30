"""Motion-compensated temporal consistency (TC, TA) for several experiments at once.

RAFT flow is computed once per consecutive frame pair (at native resolution) and shared by all experiments.
Also reports reference rows: 'GT' (labels warped, where both frames are labelled) and 'const' (single class).
Writes tc.json into each experiment directory and prints a summary.

  python tools/eval_tc.py results/kitti_val__proxy__none results/kitti_val__proxy__tpa
"""
import json
import os
import sys

import numpy as np
import torch
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from tovss.data import build_dataset, load_rgb          # noqa: E402
from tovss.flow import Raft, backward_warp, fb_valid    # noqa: E402
from tovss.metrics import pair_consistency              # noqa: E402


def main():
    exps = [e.rstrip('/') for e in sys.argv[1:]]
    metas = [json.load(open(os.path.join(e, 'metrics.json'))) for e in exps]
    m0 = metas[0]
    for m in metas:
        assert (m['dataset'], m['split'], m['stride']) == (m0['dataset'], m0['split'], m0['stride']), 'mismatch'
    kw = {'seqs': m0['seqs']} if m0['dataset'] == 'kitti_step' else {}
    ds = build_dataset(m0['dataset'], m0['split'], **kw)
    ds.sequences = [s for s in ds.sequences if s[0] in set(m0['seqs'])]
    C = len(ds.classes)
    raft = Raft('cuda')
    acc = {e: [] for e in exps}
    acc['GT'], acc['const'] = [], []
    valid_frac = []
    for seq, frames in ds.sequences:
        frames = frames[::m0['stride']]
        prev = None
        for f in frames:
            if not all(os.path.exists(os.path.join(e, 'preds', seq, f['name'] + '.png')) for e in exps):
                prev = None
                continue
            img = torch.from_numpy(load_rgb(f['img_path'])).cuda().permute(2, 0, 1)[None].float() / 255
            if prev is not None:
                pf, pimg = prev
                f_tp, f_pt = raft(img, pimg), raft(pimg, img)
                for e in exps:
                    pt = torch.from_numpy(np.array(Image.open(os.path.join(e, 'preds', seq, f['name'] + '.png')))).cuda()
                    pp = torch.from_numpy(np.array(Image.open(os.path.join(e, 'preds', seq, pf['name'] + '.png')))).cuda()
                    w, inb, grid = backward_warp(pp[None, None].float(), f_tp, mode='nearest')
                    valid = inb & fb_valid(f_tp, f_pt, grid)
                    acc[e].append(pair_consistency(pt.long(), w[0, 0].long(), valid, C))
                if f['gt_path'] and pf['gt_path']:
                    gt_t = torch.from_numpy(ds.load_gt(f).astype(np.int64)).cuda()
                    gt_p = torch.from_numpy(ds.load_gt(pf).astype(np.int64)).cuda()
                    w, inb, grid = backward_warp(gt_p[None, None].float(), f_tp, mode='nearest')
                    v = inb & fb_valid(f_tp, f_pt, grid) & (gt_t != ds.ignore_index) & (w[0, 0].long() != ds.ignore_index)
                    acc['GT'].append(pair_consistency(gt_t, w[0, 0].long(), v, C))
                    valid_frac.append(v.float().mean().item())
                acc['const'].append((1.0, 1.0))
            prev = (f, img)
        print(f'seq {seq} done', flush=True)
    summary = {}
    for k, v in acc.items():
        arr = np.array(v, dtype=np.float64) if v else np.zeros((0, 2))
        summary[k] = {'TC': float(np.nanmean(arr[:, 0]) * 100) if len(arr) else None,
                      'TA': float(np.nanmean(arr[:, 1]) * 100) if len(arr) else None, 'pairs': len(arr)}
    for e in exps:
        json.dump(summary[e] | {'ref_GT': summary['GT'], 'ref_const': summary['const']},
                  open(os.path.join(e, 'tc.json'), 'w'), indent=2)
    for k, v in summary.items():
        print(f'{os.path.basename(k):<45} TC={v["TC"]}  TA={v["TA"]}  pairs={v["pairs"]}')


if __name__ == '__main__':
    main()
