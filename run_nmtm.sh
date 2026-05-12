#!/bin/bash

echo "Running NMTM model on all datasets..."

# NMTM on Amazon_Review
echo "=== Running NMTM on Amazon_Review ==="
python main.py --model NMTM --seed 0 --dataset Amazon_Review --device 0 --gemini_api_key $GEMINI_API_KEY

# NMTM on ECNews
echo "=== Running NMTM on ECNews ==="
python main.py --model NMTM --seed 0 --dataset ECNews --device 0 --gemini_api_key $GEMINI_API_KEY

# NMTM on Rakuten_Amazon
echo "=== Running NMTM on Rakuten_Amazon ==="
python main.py --model NMTM --seed 7 --dataset Rakuten_Amazon --device 0 --gemini_api_key $GEMINI_API_KEY 

echo "NMTM experiments completed!"
