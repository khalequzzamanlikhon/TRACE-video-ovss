#!/usr/bin/env bash
# Runs the frozen TRACE configuration on one held-out set: dense per-frame scores and RAFT flow, SAM 2 tracks
# (offline and online), fusion (mIoU, VC, VCc) and motion-compensated consistency (TC, TA).
#   GPU=0 bash scripts/run_heldout.sh kitti_step val
#   GPU=0 bash scripts/run_heldout.sh camvid val
#   GPU=0 bash scripts/run_heldout.sh camvid test
set -e
cd "$(dirname "$0")/.."
DS=$1; S=$2; T=${DS}_${S}
export CUDA_VISIBLE_DEVICES=${GPU:-0} PYTHONPATH=. PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
PY=${PY:-python}
FROZEN=configs/frozen_track_v2.json
mkdir -p results
PF=$($PY -c "import json; print(json.dumps(json.load(open('$FROZEN'))['pf']))")
args() { $PY -c "import json; a=json.load(open('$FROZEN'))['$1']; print(' '.join(('--'+k) if v is True else f'--{k} {v}' for k,v in a.items()))"; }
sel() { $PY -c "import json; print(json.dumps([c for c in json.load(open('$FROZEN'))['cfgs'] if c['name'].startswith('${T}__$1__')]))"; }
OFF=cache/tracks_$T; ONT=cache/tracks_${T}_online
$PY pipeline/temporal.py --dataset $DS --split $S --final $FROZEN --pf "$PF" --cache_dir cache \
    --aggs '[{"type":"none"}]' --out results/${T}_dense.json
[ -f $OFF/done ] || { $PY pipeline/tracks.py --dataset $DS --split $S --final $FROZEN $(args tracks_args) --out $OFF && touch $OFF/done; }
[ -f $ONT/done ] || { $PY pipeline/tracks.py --dataset $DS --split $S --final $FROZEN $(args tracks_args_online) --out $ONT && touch $ONT/done; }
$PY pipeline/fusion.py --dataset $DS --split $S --final $FROZEN --pf "$PF" --tracks $OFF --clip_len 10 \
    --save_root results --cfgs "$(sel trk2)" --out results/${T}_offline.json
$PY pipeline/fusion.py --dataset $DS --split $S --final $FROZEN --pf "$PF" --tracks $ONT --clip_len 10 \
    --save_root results --cfgs "$(sel trk2on)" --out results/${T}_online.json
$PY tools/eval_tc.py $(ls -d results/${T}__trk2__* results/${T}__trk2on__*)
