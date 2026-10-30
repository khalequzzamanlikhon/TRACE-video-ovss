#!/usr/bin/env bash
# SAM 3 reference: per frame, with the flow smoother, and wrapped by TRACE. Run scripts/run_heldout.sh first (tracks).
# PY_SAM3 is the python of the SAM 3 environment (see README); PY is the python of the main environment.
#   GPU=0 PY_SAM3=/path/to/sam3/env/bin/python bash scripts/run_sam3.sh kitti_step val
set -e
cd "$(dirname "$0")/.."
DS=$1; S=$2; T=${DS}_${S}
export CUDA_VISIBLE_DEVICES=${GPU:-0} PYTHONPATH=. PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
PY=${PY:-python}
FROZEN=configs/frozen_sam3.json
PF=$($PY -c "import json; print(json.dumps(json.load(open('$FROZEN'))['pf']))")
CFGS=$($PY -c "import json; print(json.dumps([c for c in json.load(open('$FROZEN'))['cfgs'] if c['name'].startswith('${T}__')]))")
$PY_SAM3 pipeline/sam3_perframe.py --dataset $DS --split $S --final $FROZEN --mode instance --cache_dir cache --out results/${T}_sam3.json
$PY pipeline/fusion.py --dataset $DS --split $S --final $FROZEN --pf "$PF" --tracks cache/tracks_$T --clip_len 10 \
    --save_root results --cfgs "$CFGS" --out results/${T}_sam3_fusion.json
$PY tools/eval_tc.py $(ls -d results/${T}__trk2_sam3__*)
