# TRACE: Track-level Region Aggregation with Crop Evidence

Code for the paper **"Static Training-Free Open-Vocabulary Segmenters on Video: Flicker, Hallucination, and a Space-Time Remedy"**.

TRACE is a training-free, model-agnostic wrapper that turns any per-frame open-vocabulary segmentation model into a video segmenter. It pools evidence over space-time SAM 2 tracks, classifies each track from its largest views with a video-level class debiasing, and aligns the remaining dense evidence with optical flow. It has an offline and an online (causal) variant.

## Results

mIoU, per-frame → with TRACE (offline). Hyperparameters were selected on KITTI-STEP training sequences only; the sets below were used once, with a frozen configuration.

| Per-frame model | KITTI-STEP val | CamVid val | CamVid test (1 Hz) |
|---|---|---|---|
| Proxy (DINO B/16) | 34.7 → **41.5** | 45.4 → **57.6** | 39.2 → **46.6** |
| MaskCLIP | 25.4 → **39.5** | 35.4 → **55.1** | 30.7 → **45.0** |
| SCLIP-style | 32.3 → **39.7** | 35.8 → **47.2** | 32.6 → **42.1** |
| NACLIP-style | 33.0 → **40.9** | 38.7 → **50.9** | 36.2 → **47.0** |
| SAM 3 per frame (reference) | 64.6 | 68.7 | 69.4 |

SAM 3 is trained with concept-level mask supervision, so it is reported as a reference, not as a training-free method. All numbers, including per-class IoU and consistency metrics, are in [`results/summary.json`](results/summary.json).

## Installation

```bash
conda create -n trace python=3.10 -y
conda activate trace
pip install -r requirements.txt

git clone https://github.com/facebookresearch/sam2.git third_party/sam2
git -C third_party/sam2 checkout 2b90b9f
SAM2_BUILD_CUDA=0 pip install -e third_party/sam2
mkdir -p third_party/sam2/checkpoints
wget -P third_party/sam2/checkpoints https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_large.pt
```

The results were produced with PyTorch 2.5.1. DINO and CLIP weights download automatically on first use.

The SAM 3 reference needs its own environment (Python 3.12, `requirements-sam3.txt`, [facebookresearch/sam3](https://github.com/facebookresearch/sam3) at commit `7f473cf`) and access to the gated `facebook/sam3` weights.

## Data

Put the datasets under `data/`, or set `TOVSS_DATA` to their location:

```
data/CamVid/{train,trainannot,val,valannot,test,testannot}/        SegNet version, 480x360
data/kitti_step/training/image_02/<seq>/*.png                      KITTI tracking images
data/kitti_step/kitti-step/panoptic_maps/{train,val}/<seq>/*.png   KITTI-STEP labels
```

Set `SAM2_ROOT` and `SAM3_ROOT` if the repositories are not in `third_party/`.

## Reproducing the results

```bash
GPU=0 bash scripts/run_heldout.sh kitti_step val
GPU=0 bash scripts/run_heldout.sh camvid val
GPU=0 bash scripts/run_heldout.sh camvid test
```

Each run computes per-frame scores and RAFT flow, builds offline and online tracks, fuses them, and writes `results/<dataset>_<split>__trk2*/{metrics.json,tc.json,preds/}`. `trk2` is offline and `trk2on` is online. KITTI-STEP val takes several hours and about 16 GB of GPU memory.

Then, for the other per-frame models and the SAM 3 reference:

```bash
GPU=0 bash scripts/run_plug.sh kitti_step val
GPU=0 PY_SAM3=/path/to/sam3/env/bin/python bash scripts/run_sam3.sh kitti_step val
```

The frozen configurations are in `configs/`. Evaluating a validation or test split requires a configuration listed in one of those files. `tools/` holds the consistency evaluation and the scripts that generate the paper's tables and figure.

## Layout

```
tovss/      data loading, CLIP and DINO encoders, RAFT wrapper, metrics, class names
pipeline/   perframe.py, temporal.py (flow smoothing), tracks.py (SAM 2), fusion.py, sam3_perframe.py
configs/    frozen configurations
scripts/    end-to-end runs
tools/      evaluation, tables, figure
results/    summary of all reported numbers
```

## License

The code is released under the MIT License. SAM 2, SAM 3, CLIP, DINO, RAFT, KITTI-STEP and CamVid have their own licenses and terms of use.
