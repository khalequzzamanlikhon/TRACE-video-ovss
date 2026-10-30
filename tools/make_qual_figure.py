"""Qualitative figure from saved held-out predictions: rows = frames, columns = image, per-frame, TRACE, GT.
Frames are fixed in advance (middle frame of three KITTI-STEP val sequences), not selected by result.
  python tools/make_qual_figure.py
"""
import os
import sys

import numpy as np
from PIL import Image, ImageDraw, ImageFont

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from tovss.data import build_dataset, load_rgb      # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PALETTE = np.array([(128, 64, 128), (244, 35, 232), (70, 70, 70), (102, 102, 156), (190, 153, 153), (153, 153, 153),
                    (250, 170, 30), (220, 220, 0), (107, 142, 35), (152, 251, 152), (70, 130, 180), (220, 20, 60),
                    (255, 0, 0), (0, 0, 142), (0, 0, 70), (0, 60, 100), (0, 80, 100), (0, 0, 230), (119, 11, 32)],
                   np.uint8)
SEQS = ('0002', '0008', '0016')
EXPS = (('kitti_step_val__trk2__perframe', 'Per-frame'), ('kitti_step_val__trk2__full', 'TRACE'))


def colour(lbl):
    out = np.zeros(lbl.shape + (3,), np.uint8)
    ok = lbl < len(PALETTE)
    out[ok] = PALETTE[lbl[ok]]
    return out


def main():
    ds = build_dataset('kitti_step', 'val')
    seqs = dict(ds.sequences)
    W = 620
    rows = []
    for s in SEQS:
        f = seqs[s][len(seqs[s]) // 2]
        rgb = load_rgb(f['img_path'])
        h = int(round(rgb.shape[0] * W / rgb.shape[1]))
        cells = [rgb] + [colour(np.array(Image.open(os.path.join(ROOT, 'results', e, 'preds', s, f['name'] + '.png'))))
                         for e, _ in EXPS] + [colour(ds.load_gt(f))]
        cells = [np.array(Image.fromarray(c).resize((W, h), Image.NEAREST if i else Image.BILINEAR))
                 for i, c in enumerate(cells)]
        rows.append(np.concatenate([np.concatenate([c, np.full((h, 6, 3), 255, np.uint8)], 1) for c in cells], 1))
    gap = np.full((6, rows[0].shape[1], 3), 255, np.uint8)
    body = np.concatenate(sum([[r, gap] for r in rows], [])[:-1], 0)
    head = Image.new('RGB', (body.shape[1], 44), 'white')
    d = ImageDraw.Draw(head)
    try:
        font = ImageFont.truetype('DejaVuSans.ttf', 30)
    except OSError:
        font = ImageFont.load_default()
    for i, t in enumerate(['Image'] + [n for _, n in EXPS] + ['Ground truth']):
        w = d.textlength(t, font=font)
        d.text((i * (W + 6) + (W - w) / 2, 6), t, fill='black', font=font)
    fig = np.concatenate([np.array(head), body], 0)
    os.makedirs(os.path.join(ROOT, 'paper_output', 'figures'), exist_ok=True)
    Image.fromarray(fig).save(os.path.join(ROOT, 'paper_output', 'figures', 'qual.png'))
    print(fig.shape)


if __name__ == '__main__':
    main()
