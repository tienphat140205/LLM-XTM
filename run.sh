#!/bin/bash

echo "Usage: bash run.sh <MODEL> <DATASET> [SEED] [DEVICE]"
echo "Example: bash run.sh XTRA Amazon_Review 0 0"

MODEL=${1:-XTRA}
DATASET=${2:-Amazon_Review}
SEED=${3:-0}
DEVICE=${4:-0}

python main.py --model ${MODEL} --dataset ${DATASET} --seed ${SEED} --device ${DEVICE} --gemini_api_key $GEMINI_API_KEY

