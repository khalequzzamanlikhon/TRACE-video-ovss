#!/usr/bin/env bash
# Plug-and-play: the same fusion parameters on other per-frame models. Run scripts/run_heldout.sh first (tracks).
#   GPU=0 HEADS="maskclip sclip naclip" bash scripts/run_plug.sh kitti_step val
set -e
cd "$(dirname "$0")/.."
DS=$1; S=$2; T=${DS}_${S}
export CUDA_VISIBLE_DEVICES=${GPU:-0} PYTHONPATH=. PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
PY=${PY:-python}
for HEAD in ${HEADS:-maskclip sclip naclip}; do
  FROZEN=configs/frozen_track_v2_plug_$HEAD.json
  PF=$($PY -c "import json; print(json.dumps(json.load(open('$FROZEN'))['pf']))")
  sel() { $PY -c "import json; print(json.dumps([c for c in json.load(open('$FROZEN'))['cfgs'] if c['name'].startswith('${T}__trk2$1_${HEAD}__')]))"; }
  $PY pipeline/temporal.py --dataset $DS --split $S --final $FROZEN --pf "$PF" --cache_dir cache \
      --aggs '[{"type":"none"}]' --out results/${T}_${HEAD}_dense.json
  $PY pipeline/fusion.py --dataset $DS --split $S --final $FROZEN --pf "$PF" --tracks cache/tracks_$T --clip_len 10 \
      --save_root results --cfgs "$(sel '')" --out results/${T}_${HEAD}_offline.json
  $PY pipeline/fusion.py --dataset $DS --split $S --final $FROZEN --pf "$PF" --tracks cache/tracks_${T}_online --clip_len 10 \
      --save_root results --cfgs "$(sel on)" --out results/${T}_${HEAD}_online.json
  $PY tools/eval_tc.py $(ls -d results/${T}__trk2_${HEAD}__* results/${T}__trk2on_${HEAD}__*)
done
