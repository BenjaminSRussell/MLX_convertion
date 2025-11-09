#!/bin/bash
# Quick test script for encoder model conversion
#
# Usage: ./test_encoder_conversion.sh [model_name]
# Example: ./test_encoder_conversion.sh distilbert-base-uncased-mnli

set -e

MODEL_NAME="${1:-distilbert-base-uncased-mnli}"
OUTPUT_DIR="models/mlx_converted"
MODEL_PATH="${OUTPUT_DIR}/${MODEL_NAME}-mlx-q8"

echo "========================================"
echo "MLX Encoder Model Conversion Test"
echo "========================================"
echo ""
echo "Model: $MODEL_NAME"
echo "Output: $MODEL_PATH"
echo ""

# Check if virtual environment is activated
if [ -z "$VIRTUAL_ENV" ]; then
    echo "⚠️  No virtual environment detected!"
    echo "   Please activate your virtual environment first:"
    echo "   source .venv/bin/activate"
    echo ""
    exit 1
fi

echo "✓ Virtual environment active: $VIRTUAL_ENV"
echo ""

# Step 1: Convert the model
echo "Step 1: Converting model..."
echo "----------------------------------------"
python scripts/convert_encoder.py --model "$MODEL_NAME"

if [ $? -ne 0 ]; then
    echo "❌ Conversion failed!"
    exit 1
fi

echo ""
echo "✓ Conversion completed!"
echo ""

# Step 2: Test the converted model
echo "Step 2: Testing converted model..."
echo "----------------------------------------"
python scripts/test_encoder.py "$MODEL_PATH" --full-test

if [ $? -ne 0 ]; then
    echo "❌ Testing failed!"
    exit 1
fi

echo ""
echo "========================================"
echo "✓ All tests passed successfully!"
echo "========================================"
echo ""
echo "Model saved at: $MODEL_PATH"
echo ""
echo "Next steps:"
echo "  1. Review the model at: $MODEL_PATH"
echo "  2. Check metadata: cat $MODEL_PATH/conversion_metadata.json"
echo "  3. Use the model in your MLX application"
echo ""
