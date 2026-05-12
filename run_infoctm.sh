#!/bin/bash

echo "Running InfoCTM model on all datasets..."

# InfoCTM on Amazon_Review (weight_MI: 50.0)
echo "=== Running InfoCTM on Amazon_Review ==="
python main.py --model InfoCTM --seed 0 --dataset Amazon_Review --device 0 --gemini_api_key $GEMINI_API_KEY

# InfoCTM on ECNews (weight_MI: 30.0)  
echo "=== Running InfoCTM on ECNews ==="
python main.py --model InfoCTM --seed 0 --dataset ECNews --device 0 --gemini_api_key $GEMINI_API_KEY

# InfoCTM on Rakuten_Amazon (weight_MI: 50.0)
echo "=== Running InfoCTM on Rakuten_Amazon ==="
python main.py --model InfoCTM --seed 7 --dataset Rakuten_Amazon --device 0 --gemini_api_key $GEMINI_API_KEY

echo "InfoCTM experiments completed!"
