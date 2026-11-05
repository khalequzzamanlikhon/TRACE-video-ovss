"""Where do the remaining false positives fall: inside SAM 2 tracks or on uncovered pixels? (evaluation only)
Reads saved predictions (results/<exp>/preds) and the track id maps; per over-predicted class, prints the share
of its false-positive pixels that are uncovered and the class precision on covered / uncovered pixels.
With --tex, writes paper_output/tables/coverage.tex.
  python tools/coverage.py kitti_step_val__trk2__full cache/tracks_kitti_step_val --split val --tex
"""
import argparse
import json
import os
import sys

import numpy as np
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from tovss.data import build_dataset      # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument('exp')
ap.add_argument('tracks')
ap.add_argument('--dataset', default='kitti_step')
ap.add_argument('--split', default='dev')
ap.add_argument('--tex', action='store_true')
a = ap.parse_args()
ds = build_dataset(a.dataset, a.split)
C = len(ds.classes)
cm = {k: np.zeros((C, C), np.int64) for k in ('cov', 'unc')}
for seq, frames in ds.sequences:
    for f in frames:
        if not f['gt_path']:
            continue
        p = np.array(Image.open(os.path.join('results', a.exp, 'preds', seq, f['name'] + '.png')))
        ids = np.array(Image.open(os.path.join(a.tracks, seq, f'ids_{f["name"]}.png')))
        g = ds.load_gt(f)
        for k, m in (('cov', ids > 0), ('unc', ids == 0)):
            v = m & (g < C)
            cm[k] += np.bincount(C * g[v].astype(np.int64) + p[v], minlength=C * C).reshape(C, C)
tot = cm['cov'] + cm['unc']
frac_cov = cm['cov'].sum() / tot.sum()
print(f'pixels covered: {frac_cov:.3f}')
rows, out = [], {'covered': float(frac_cov), 'classes': {}}
for i, c in enumerate(ds.classes):
    fp = tot[:, i].sum() - tot[i, i]
    if tot[:, i].sum() == 0 or fp / max(tot[:, i].sum(), 1) < 0.5:
        continue
    fpu = (cm['unc'][:, i].sum() - cm['unc'][i, i]) / max(fp, 1)
    pc = cm['cov'][i, i] / max(cm['cov'][:, i].sum(), 1)
    pu = cm['unc'][i, i] / max(cm['unc'][:, i].sum(), 1)
    out['classes'][c] = {'fp_px': int(fp), 'fp_uncovered': float(fpu), 'prec_cov': float(pc), 'prec_unc': float(pu)}
    print(f'{c:<14}{fp:>10}{fpu:>11.2f}{pc:>10.2f}{pu:>10.2f}')
    rows.append(f'{c} & {100 * fpu:.0f}\\% & {100 * pc:.0f} & {100 * pu:.0f} \\\\')
json.dump(out, open(os.path.join('results', f'coverage_{a.exp}.json'), 'w'), indent=1)
if a.tex:
    os.makedirs(os.path.join('paper_output', 'tables'), exist_ok=True)
    tex = [r'\begin{table}[t]\centering\small',
           rf'\caption{{Remaining false positives of \ours on KITTI-STEP val. SAM\,2 tracks cover {100 * frac_cov:.0f}\% '
           r'of the pixels; for each class whose predictions are mostly wrong, the share of its false positives on '
           r'uncovered pixels and its precision on covered / uncovered pixels.}\label{tab:coverage}',
           r'\begin{adjustbox}{max width=\columnwidth}', r'\begin{tabular}{lccc}', r'\toprule',
           r'Class & FP on uncovered & Prec.\ covered & Prec.\ uncovered \\',
           r'\midrule'] + rows + [r'\bottomrule', r'\end{tabular}', r'\end{adjustbox}', r'\end{table}']
    open(os.path.join('paper_output', 'tables', 'coverage.tex'), 'w').write('\n'.join(tex) + '\n')
