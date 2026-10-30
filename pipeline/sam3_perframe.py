"""SAM 3 as a per-frame open-vocabulary semantic segmenter, cached in the same format as pipeline/temporal.py
(per-frame class log-scores at half native resolution + RAFT flows), so pipeline/fusion.py can evaluate it per frame
or wrap it with the video method.

Per frame and class c (max over the class's query names), with presence score pi_c (SAM 3's presence head):
  semantic : s_c(x) = sigmoid(semantic_seg_c(x)) * pi_c
  instance : s_c(x) = max_q  sigmoid(score_q) * pi_c * sigmoid(mask_q(x))
  both     : max(semantic, instance)
Cached value: log(s_c + 1e-6)  (softmax over classes then gives the normalised scores).

Runs in the `sam3` env:
  python pipeline/sam3_perframe.py --mode semantic --split dev --cache_dir cache
"""
import argparse
import hashlib
import json
import os
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from tovss.data import build_dataset, load_rgb          # noqa: E402
from tovss.flow import Raft                              # noqa: E402
from tovss.metrics import ConfusionMIoU                  # noqa: E402

SAM3_ROOT = os.environ.get('SAM3_ROOT', 'third_party/sam3')


MODES = ('semantic', 'instance', 'both')


@torch.inference_mode()
def class_scores(model, proc, state, text_cache, queries, size):
    """-> {mode: (C, h, w) scores in [0, 1] at `size`} for all three modes (one forward pass per query)."""
    out = {m: [] for m in MODES}
    for names in queries:
        best = {m: None for m in MODES}
        for n in names:
            state['backbone_out'].update(text_cache[n])
            o = model.forward_grounding(backbone_out=state['backbone_out'], find_input=proc.find_stage,
                                        geometric_prompt=model._get_dummy_prompt(), find_target=None)
            pres = o['presence_logit_dec'].float().sigmoid().reshape(-1)[0]
            sem = o['semantic_seg'].float()
            s_sem = F.interpolate(sem.reshape(1, 1, *sem.shape[-2:]), size, mode='bilinear',
                                  align_corners=False)[0, 0].sigmoid() * pres
            sc = o['pred_logits'].float().sigmoid().reshape(-1) * pres            # (Q,)
            m = o['pred_masks'].float()
            m = F.interpolate(m.reshape(1, -1, *m.shape[-2:]), size, mode='bilinear', align_corners=False)[0]
            s_ins = (sc[:, None, None] * m.sigmoid()).max(0).values
            for mode, s in zip(MODES, (s_sem, s_ins, torch.maximum(s_sem, s_ins))):
                best[mode] = s if best[mode] is None else torch.maximum(best[mode], s)
        for mode in MODES:
            out[mode].append(best[mode])
    return {mode: torch.stack(v) for mode, v in out.items()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--dataset', default='kitti_step')
    ap.add_argument('--split', default='dev')
    ap.add_argument('--mode', default='semantic', choices=['semantic', 'instance', 'both'])
    ap.add_argument('--scale', type=float, default=0.5)
    ap.add_argument('--cache_dir', default='cache')
    ap.add_argument('--max_frames', type=int, default=None)
    ap.add_argument('--final', default=None, help='frozen config file (committed); required for val/test')
    ap.add_argument('--out', required=True)
    a = ap.parse_args()
    pf = {'model': 'sam3', 'mode': a.mode}
    if a.dataset != 'kitti_step' or a.split not in ('dev', 'train'):
        assert a.final and os.path.exists(a.final), 'val/test only with a committed frozen config (--final)'
        assert pf == json.load(open(a.final))['pf'], 'per-frame config must match the frozen one'
    sys.path.insert(0, SAM3_ROOT)
    from sam3.model_builder import build_sam3_image_model
    from sam3.model.sam3_image_processor import Sam3Processor

    model = build_sam3_image_model()
    proc = Sam3Processor(model)
    ds = build_dataset(a.dataset, a.split)
    names = sorted({n for q in ds.queries for n in q})
    with torch.inference_mode():
        text_cache = {n: model.backbone.forward_text([n], device='cuda') for n in names}
    raft = Raft('cuda')
    tag = hashlib.md5(json.dumps([pf, a.scale, a.max_frames], sort_keys=True).encode()).hexdigest()[:8]
    C = len(ds.classes)
    mious = {m: ConfusionMIoU(C, ds.ignore_index) for m in MODES}
    miou = mious[a.mode]
    t0 = time.time()
    for seq, frames in ds.sequences:
        frames = frames[:a.max_frames]
        logits, flows, prev = [], [None], None
        for f in frames:
            rgb = load_rgb(f['img_path'])
            H, W = rgb.shape[:2]
            h, w = int(round(H * a.scale)), int(round(W * a.scale))
            with torch.autocast('cuda', dtype=torch.bfloat16):
                state = proc.set_image(Image.fromarray(rgb))
                S = class_scores(model, proc, state, text_cache, ds.queries, (h, w))
            lgs = {m: torch.log(s.float() + 1e-6) for m, s in S.items()}
            logits.append(lgs[a.mode].half())
            if f['gt_path']:
                gt = ds.load_gt(f)
                for m, lg in lgs.items():
                    pred = F.interpolate(lg[None], (H, W), mode='bilinear', align_corners=False).argmax(1)[0]
                    mious[m].update(pred.to(torch.uint8).cpu().numpy(), gt)
            img = torch.from_numpy(rgb).cuda().permute(2, 0, 1)[None].float() / 255.0
            small = F.interpolate(img, (h, w), mode='bilinear', align_corners=False, antialias=True)
            if prev is not None:
                flows.append((raft(small, prev).half(), raft(prev, small).half()))
            prev = small
        os.makedirs(a.cache_dir, exist_ok=True)
        torch.save({'logits': logits, 'flows': flows, 'size': (H, W)}, os.path.join(a.cache_dir, f'{seq}_{tag}.pt'))
        print(f'seq {seq} done ({len(frames)} frames, {time.time() - t0:.0f}s), running mIoU {miou.result()["mIoU"]:.2f}',
              flush=True)
    rows = []
    for m, mi in mious.items():
        r = mi.result()
        print(json.dumps({'mode': m, 'cached': m == a.mode, 'mIoU': round(r['mIoU'], 2)}), flush=True)
        rows.append({'pf': {'model': 'sam3', 'mode': m}, 'cached': m == a.mode, 'split': a.split,
                     'dataset': a.dataset, 'mIoU': r['mIoU'], 'per_class': dict(zip(ds.classes, r['per_class']))})
    old = json.load(open(a.out)) if os.path.exists(a.out) else []
    json.dump(old + rows, open(a.out, 'w'), indent=1)


if __name__ == '__main__':
    main()
