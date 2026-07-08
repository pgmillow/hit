#!/bin/bash
# Setup debug environment for encoder_norm.bias check
# Creates symlinks for offline dataset access, then runs the debug training.
# After completion, removes the symlink.

set -e

DS_DIR="$HOME/openpi/assets/gxd_pi05_v3src_pad/local/openpi_V5_mcap0625_v3src_pad"
CACHE_DIR="$HOME/.cache/huggingface/lerobot/local"

echo "=== Setting up dataset symlink ==="
mkdir -p "$CACHE_DIR"
if [ ! -e "$CACHE_DIR/openpi_V5_mcap0625_v3src_pad" ]; then
    ln -s "$DS_DIR" "$CACHE_DIR/openpi_V5_mcap0625_v3src_pad"
    echo "  Created symlink: $CACHE_DIR/openpi_V5_mcap0625_v3src_pad -> $DS_DIR"
else
    echo "  Symlink already exists"
fi

echo ""
echo "=== Running debug_encoder_norm (10 steps) ==="
cd "$HOME/openpi"
CUDA_VISIBLE_DEVICES=2,3 .venv/bin/python scripts/train.py debug_encoder_norm

echo ""
echo "=== Cleanup ==="
if [ -L "$CACHE_DIR/openpi_V5_mcap0625_v3src_pad" ]; then
    rm "$CACHE_DIR/openpi_V5_mcap0625_v3src_pad"
    echo "  Removed symlink"
fi
echo "Done"
