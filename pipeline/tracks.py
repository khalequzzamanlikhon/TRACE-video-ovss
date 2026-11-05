"""Class-agnostic space-time tracks with SAM2 + multi-view crop-CLIP embeddings.

Per sequence, in clips of L frames:
  1. SAM2 automatic masks on the clip's middle frame -> non-overlapping segments (small masks painted on top).
  2. Each segment becomes an object of the SAM2 video predictor, propagated forward and backward over the clip.
  3. Per frame: id map = argmax over objects of mask logits (> 0), 0 = uncovered.
  4. Per track: the K largest views -> masked crop and context crop -> CLIP ViT-B/16 image embeddings.
Outputs (cache/tracks/<seq>/): ids_<frame>.png (uint16, native res, globally unique ids) and tracks.pt
  {track_id: {'clip': c, 'views': [(frame_idx, area)], 'emb_mask': (K,E), 'emb_ctx': (K,E)}}.

Called by scripts/run_heldout.sh for the held-out sets. On the KITTI-STEP development sequences:
  python pipeline/tracks.py --seqs 0001,0004,0011
"""
import argparse
import os
import shutil
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from tovss.data import build_dataset, load_rgb          # noqa: E402

SAM2_ROOT = os.environ.get('SAM2_ROOT', 'third_party/sam2')
CLIP_MEAN = torch.tensor((0.48145466, 0.4578275, 0.40821073)).view(3, 1, 1)
CLIP_STD = torch.tensor((0.26862954, 0.26130258, 0.27577711)).view(3, 1, 1)


def segments_from_amg(masks, min_area):
    """AMG mask dicts -> list of non-overlapping bool masks (large first, small painted on top)."""
    masks = sorted(masks, key=lambda m: -m['area'])
    H, W = masks[0]['segmentation'].shape if masks else (0, 0)
    idmap = np.zeros((H, W), np.int32)
    for i, m in enumerate(masks, start=1):
        idmap[m['segmentation']] = i
    segs = []
    for i in range(1, len(masks) + 1):
        s = idmap == i
        if s.sum() >= min_area:
            segs.append(s)
    return segs


def crops_for_view(rgb, mask, size=224, margin=0.15):
    """rgb (H,W,3) uint8, mask (H,W) bool -> (masked crop, context crop) as (3,size,size) float in [0,1]."""
    ys, xs = np.nonzero(mask)
    y0, y1, x0, x1 = ys.min(), ys.max() + 1, xs.min(), xs.max() + 1
    h, w = y1 - y0, x1 - x0
    side = max(h, w) * (1 + 2 * margin)
    side = max(side, 32)
    cy, cx = (y0 + y1) / 2, (x0 + x1) / 2
    H, W = mask.shape
    yy0, xx0 = int(round(cy - side / 2)), int(round(cx - side / 2))
    yy1, xx1 = int(round(yy0 + side)), int(round(xx0 + side))
    pad = [max(0, -yy0), max(0, yy1 - H), max(0, -xx0), max(0, xx1 - W)]
    img = torch.from_numpy(rgb).permute(2, 0, 1).float() / 255.0
    m = torch.from_numpy(mask)[None].float()
    img = F.pad(img, (pad[2], pad[3], pad[0], pad[1]), value=0.5)
    m = F.pad(m, (pad[2], pad[3], pad[0], pad[1]), value=0.0)
    sl = (slice(None), slice(yy0 + pad[0], yy1 + pad[0]), slice(xx0 + pad[2], xx1 + pad[2]))
    ci, cm = img[sl], m[sl]
    masked = ci * cm + 0.5 * (1 - cm)
    rs = lambda x: F.interpolate(x[None], (size, size), mode='bilinear', align_corners=False, antialias=True)[0]
    return rs(masked), rs(ci)


@torch.no_grad()
def embed(clip, crops, bs=64):
    out = []
    for i in range(0, len(crops), bs):
        x = torch.stack(crops[i:i + bs]).cuda()
        x = (x - CLIP_MEAN.cuda()) / CLIP_STD.cuda()
        out.append(F.normalize(clip.encode_image(x).float(), dim=-1).cpu())
    return torch.cat(out) if out else torch.zeros(0, 512)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--split', default='dev')
    ap.add_argument('--seqs', default=None)
    ap.add_argument('--clip_len', type=int, default=20)
    ap.add_argument('--max_frames', type=int, default=None)
    ap.add_argument('--K', type=int, default=5, help='views per track')
    ap.add_argument('--pps', type=int, default=24, help='AMG points per side')
    ap.add_argument('--min_area', type=int, default=200)
    ap.add_argument('--causal', action='store_true',
                    help='online: keyframe = first frame of the clip, forward tracking only, embeddings for every view')
    ap.add_argument('--iou_thr', type=float, default=0.8)
    ap.add_argument('--stab_thr', type=float, default=0.9)
    ap.add_argument('--model', default='l')
    ap.add_argument('--out', default='cache/tracks')
    ap.add_argument('--final', default=None, help='frozen config file (committed); required for val/test')
    ap.add_argument('--dataset', default='kitti_step')
    a = ap.parse_args()
    if a.dataset != 'kitti_step' or a.split not in ('dev', 'train'):
        assert a.final and os.path.exists(a.final), 'val/test only with a committed frozen config (--final)'
    sys.path.insert(0, SAM2_ROOT)
    from sam2.build_sam import build_sam2, build_sam2_video_predictor
    from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator
    import open_clip

    cfg = f'configs/sam2.1/sam2.1_hiera_{a.model}.yaml'
    ckpt = os.path.join(SAM2_ROOT, 'checkpoints', f'sam2.1_hiera_{"large" if a.model == "l" else a.model}.pt')
    amg = SAM2AutomaticMaskGenerator(build_sam2(cfg, ckpt, device='cuda'), points_per_side=a.pps,
                                     pred_iou_thresh=a.iou_thr, stability_score_thresh=a.stab_thr,
                                     min_mask_region_area=a.min_area)
    vp = build_sam2_video_predictor(cfg, ckpt, device='cuda')
    clip, _, _ = open_clip.create_model_and_transforms('ViT-B-16', pretrained='openai')
    clip = clip.cuda().eval()

    kw = {'seqs': a.seqs.split(',')} if a.seqs else {}
    ds = build_dataset(a.dataset, a.split, **kw)
    tmp = os.path.join(a.out, '_jpg')
    t0 = time.time()
    for seq, frames in ds.sequences:
        frames = frames[:a.max_frames]
        od = os.path.join(a.out, seq)
        os.makedirs(od, exist_ok=True)
        tracks, next_id = {}, 1
        for c0 in range(0, len(frames), a.clip_len):
            clipf = frames[c0:c0 + a.clip_len]
            shutil.rmtree(tmp, ignore_errors=True)
            os.makedirs(tmp)
            rgbs = [load_rgb(f['img_path']) for f in clipf]
            for i, r in enumerate(rgbs):
                Image.fromarray(r).save(os.path.join(tmp, f'{i:05d}.jpg'), quality=95)
            key = 0 if a.causal else len(clipf) // 2
            with torch.autocast('cuda', dtype=torch.bfloat16):
                segs = segments_from_amg(amg.generate(rgbs[key]), a.min_area)
                H, W = rgbs[0].shape[:2]
                logits = {}
                if segs:
                    st = vp.init_state(video_path=tmp, offload_video_to_cpu=True)
                    for k, s in enumerate(segs):
                        vp.add_new_mask(st, frame_idx=key, obj_id=k + 1, mask=s)
                    for rev in ((False,) if a.causal else (False, True)):
                        for fi, oids, ml in vp.propagate_in_video(st, start_frame_idx=key, reverse=rev):
                            logits[fi] = (list(oids), ml[:, 0].float())
                    vp.reset_state(st)
            n_obj = len(segs)
            areas = np.zeros((len(clipf), n_obj + 1), np.int64)
            idmaps = []
            for fi in range(len(clipf)):
                if fi in logits:
                    oids, ml = logits[fi]
                    best, arg = ml.max(0)
                    idm = torch.tensor(oids, device=ml.device)[arg]
                    idm[best <= 0] = 0
                    idm = idm.cpu().numpy().astype(np.int32)
                else:
                    idm = np.zeros((H, W), np.int32)
                idmaps.append(idm)
                areas[fi] = np.bincount(idm.ravel(), minlength=n_obj + 1)[:n_obj + 1]
            # multi-view crops: K largest views of each local object
            crops_m, crops_c, owners = [], [], []
            for k in range(1, n_obj + 1):
                # online: every view in frame order (the fusion picks the largest views seen so far)
                order = np.arange(len(clipf)) if a.causal else np.argsort(-areas[:, k])[:a.K]
                views = [(int(fi), int(areas[fi, k])) for fi in order if areas[fi, k] >= a.min_area]
                if not views:
                    continue
                gid = next_id
                next_id += 1
                tracks[gid] = {'clip': c0, 'local': k, 'views': views,
                               'total_area': int(areas[:, k].sum()), 'n_frames': int((areas[:, k] > 0).sum())}
                for fi, _ in views:
                    cm, cc = crops_for_view(rgbs[fi], idmaps[fi] == k)
                    crops_m.append(cm)
                    crops_c.append(cc)
                    owners.append(gid)
            em, ec = embed(clip, crops_m), embed(clip, crops_c)
            for gid in set(owners):
                idx = [i for i, o in enumerate(owners) if o == gid]
                tracks[gid]['emb_mask'] = em[idx].half()
                tracks[gid]['emb_ctx'] = ec[idx].half()
            local2global = np.zeros(n_obj + 1, np.int32)
            for gid, t in tracks.items():
                if t['clip'] == c0:
                    local2global[t['local']] = gid
            for fi, f in enumerate(clipf):
                Image.fromarray(local2global[idmaps[fi]].astype(np.uint16)).save(os.path.join(od, f'ids_{f["name"]}.png'))
            print(f'{seq} clip {c0}: {n_obj} segments, {time.time() - t0:.0f}s', flush=True)
        torch.save(tracks, os.path.join(od, 'tracks.pt'))
    shutil.rmtree(tmp, ignore_errors=True)


if __name__ == '__main__':
    main()
