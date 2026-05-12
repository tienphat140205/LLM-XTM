#!/bin/bash

echo "Running XTRA model on all datasets..."

# XTRA on Amazon_Review
echo "=== Running XTRA on Amazon_Review ==="
python main.py --model XTRA --seed 0 --dataset Amazon_Review --device 0 --gemini_api_key $GEMINI_API_KEY

# XTRA on ECNews
echo "=== Running XTRA on ECNews ==="
python main.py --model XTRA --seed 0 --dataset ECNews --device 0 --gemini_api_key $GEMINI_API_KEY

# XTRA on Rakuten_Amazon
echo "=== Running XTRA on Rakuten_Amazon ==="
python main.py --model XTRA --seed 7 --dataset Rakuten_Amazon --device 0 --gemini_api_key $GEMINI_API_KEY

echo "XTRA experiments completed!"
