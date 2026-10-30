"""Video datasets. Every dataset is a list of sequences; each sequence is an ordered list of frames.

A frame is a dict: {'seq', 'name', 'img_path', 'gt_path' (or None)}. Images are returned as uint8 HxWx3 (RGB),
labels as uint8 HxW at NATIVE resolution (never resized). Void = dataset.ignore_index.
"""
import os
import re
import numpy as np
from PIL import Image

from tovss.text import CAMVID_CLASSES, CAMVID_QUERIES, CITYSCAPES_CLASSES, CITYSCAPES_QUERIES

DATA_ROOT = os.environ.get('TOVSS_DATA', 'data')

KITTI_STEP_SPLITS = {
    'val': ['0002', '0006', '0007', '0008', '0010', '0013', '0014', '0016', '0018'],
    'train': ['0000', '0001', '0003', '0004', '0005', '0009', '0011', '0012', '0015', '0017', '0019', '0020'],
    'dev': ['0001', '0004', '0011'],  # tuning subset of train (never used for reporting)
}


def load_rgb(path):
    return np.array(Image.open(path).convert('RGB'))


class VideoDataset:
    name = ''
    classes = []
    queries = []
    ignore_index = 255
    short_side = 448      # inference resolution (short side), aspect ratio kept

    def __init__(self):
        self.sequences = []   # list of (seq_id, [frame dicts])

    def load_gt(self, frame):
        raise NotImplementedError

    def num_frames(self, labelled_only=True):
        return sum(sum(1 for f in fr if (f['gt_path'] or not labelled_only)) for _, fr in self.sequences)


class CamVid(VideoDataset):
    """SegNet CamVid (480x360, 11 classes, void=11).

    val  = 0016E5 frames 07959..08159, step 2 (15 Hz).
    test = 0001TP (62) + Seq05VD (171), step 30 (1 Hz).
    train part of 0016E5 (204 frames, 15 Hz) is also exposed as split 'train0016E5'.
    """
    name = 'camvid'
    classes = CAMVID_CLASSES
    queries = CAMVID_QUERIES
    ignore_index = 11
    short_side = 448

    def __init__(self, split='val', root=None):
        super().__init__()
        root = root or os.path.join(DATA_ROOT, 'CamVid')
        base_split = 'train' if split.startswith('train') else split
        img_dir, ann_dir = os.path.join(root, base_split), os.path.join(root, base_split + 'annot')
        names = sorted(f for f in os.listdir(img_dir) if f.endswith('.png'))
        seqs = {}
        for n in names:
            seq, fr = n[:-4].split('_')
            if split == 'train0016E5' and seq != '0016E5':
                continue
            num = int(re.sub(r'\D', '', fr))
            seqs.setdefault(seq, []).append((num, {'seq': seq, 'name': n[:-4], 'img_path': os.path.join(img_dir, n),
                                                   'gt_path': os.path.join(ann_dir, n)}))
        for seq in sorted(seqs):
            self.sequences.append((seq, [f for _, f in sorted(seqs[seq], key=lambda x: x[0])]))

    def load_gt(self, frame):
        return np.array(Image.open(frame['gt_path']))


class KittiStep(VideoDataset):
    """KITTI-STEP: KITTI tracking sequences (10 fps) with dense per-frame Cityscapes-19 labels.
    Panoptic PNG: R = semantic trainId (0..18, 255 void), G*256+B = instance id."""
    name = 'kitti_step'
    classes = CITYSCAPES_CLASSES
    queries = CITYSCAPES_QUERIES
    ignore_index = 255
    short_side = 384

    def __init__(self, split='val', root=None, seqs=None):
        super().__init__()
        root = root or os.path.join(DATA_ROOT, 'kitti_step')
        label_split = 'val' if split == 'val' else 'train'
        seq_list = seqs or KITTI_STEP_SPLITS[split]
        for seq in seq_list:
            img_dir = os.path.join(root, 'training', 'image_02', seq)
            lab_dir = os.path.join(root, 'kitti-step', 'panoptic_maps', label_split, seq)
            if not os.path.isdir(img_dir):
                raise FileNotFoundError(img_dir)
            labelled = set(os.listdir(lab_dir)) if os.path.isdir(lab_dir) else set()
            frames = []
            for n in sorted(f for f in os.listdir(img_dir) if f.endswith('.png')):
                frames.append({'seq': seq, 'name': n[:-4], 'img_path': os.path.join(img_dir, n),
                               'gt_path': os.path.join(lab_dir, n) if n in labelled else None})
            self.sequences.append((seq, frames))

    def load_gt(self, frame):
        return np.array(Image.open(frame['gt_path']).convert('RGB'))[:, :, 0]


def build_dataset(name, split, **kw):
    if name == 'camvid':
        return CamVid(split, **kw)
    if name == 'kitti_step':
        return KittiStep(split, **kw)
    raise ValueError(name)


def inference_size(h, w, short_side, mult=16):
    """Aspect-preserving resize so the short side ~= short_side; both sides multiples of `mult`."""
    s = short_side / min(h, w)
    return max(mult, int(round(h * s / mult)) * mult), max(mult, int(round(w * s / mult)) * mult)
