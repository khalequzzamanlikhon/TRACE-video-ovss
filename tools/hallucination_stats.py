"""Per-class recall / precision of saved predictions and where each class's false positives fall
(evaluation only; reads results/<exp>/preds and the GT). Writes paper_output/tables/halluc.tex.

  python tools/hallucination_stats.py kitti_step_val__trk2__perframe kitti_step_val__trk2__full
"""
import json
import os
import sys

import numpy as np
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from tovss.data import build_dataset      # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def confusion(exp):
    m = json.load(open(os.path.join(ROOT, 'results', exp, 'metrics.json')))
    ds = build_dataset(m['dataset'], m['split'])
    C = len(ds.classes)
    cm = np.zeros((C, C), np.int64)
    for seq, frames in ds.sequences:
        for f in frames:
            if not f['gt_path']:
                continue
            p = np.array(Image.open(os.path.join(ROOT, 'results', exp, 'preds', seq, f['name'] + '.png')))
            g = ds.load_gt(f)
            k = g < C
            cm += np.bincount(C * g[k].astype(np.int64) + p[k], minlength=C * C).reshape(C, C)
    return ds.classes, cm


def main():
    exps = sys.argv[1:]
    out = {}
    for e in exps:
        classes, cm = confusion(e)
        tp = np.diag(cm).astype(float)
        rec, prec = tp / np.maximum(cm.sum(1), 1), tp / np.maximum(cm.sum(0), 1)
        fp_to = {}
        for i, c in enumerate(classes):
            col = cm[:, i].astype(float)
            col[i] = 0
            j = int(col.argmax())
            fp_to[c] = (classes[j], float(col[j] / max(cm[:, i].sum(), 1)))
        out[e] = {'classes': classes, 'recall': rec.tolist(), 'precision': prec.tolist(), 'fp_to': fp_to,
                  'gt_px': cm.sum(1).tolist(), 'pred_px': cm.sum(0).tolist()}
    json.dump(out, open(os.path.join(ROOT, 'results', 'hallucination_stats.json'), 'w'), indent=1)
    a = out[exps[0]]
    b = out[exps[1]] if len(exps) > 1 else None
    rows = []
    for i, c in enumerate(a['classes']):
        ratio = a['pred_px'][i] / max(a['gt_px'][i], 1)
        if ratio < 1.5 and a['precision'][i] > 0.5:
            continue               # only classes that are clearly over-predicted
        r = [c, f'{100 * a["recall"][i]:.0f}', f'{100 * a["precision"][i]:.0f}', f'{ratio:.1f}',
             f'{a["fp_to"][c][0]} ({100 * a["fp_to"][c][1]:.0f}\\%)']
        if b:
            r += [f'{100 * b["precision"][i]:.0f}']
        rows.append(' & '.join(r) + r' \\')
    head = ['Class', 'Recall', 'Precision', 'Pred./GT px', 'Main false-positive source'] + (['Prec.\\ (\\ours)'] if b else [])
    tex = [r'\begin{table}[t]\centering\small',
           r'\caption{Hallucination on KITTI-STEP val: over-predicted classes of the per-frame model, the ground-truth '
           r'class most of their false positives belong to, and precision after \ours.}\label{tab:halluc}',
           r'\setlength{\tabcolsep}{3pt}', r'\begin{adjustbox}{max width=\columnwidth}',
           r'\begin{tabular}{l' + 'c' * (len(head) - 1) + '}', r'\toprule',
           ' & '.join(head) + r' \\', r'\midrule'] + rows + [r'\bottomrule', r'\end{tabular}', r'\end{adjustbox}',
                                                             r'\end{table}']
    os.makedirs(os.path.join(ROOT, 'paper_output', 'tables'), exist_ok=True)
    open(os.path.join(ROOT, 'paper_output', 'tables', 'halluc.tex'), 'w').write('\n'.join(tex) + '\n')
    print('\n'.join(rows))


if __name__ == '__main__':
    main()
