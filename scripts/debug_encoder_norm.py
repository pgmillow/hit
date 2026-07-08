#!/usr/bin/env python3
"""Run debug_encoder_norm config: 10 steps, check encoder_norm.bias before/after save."""
import os
import sys

# Setup offline dataset access via symlink
DS_SRC = os.path.expanduser("~/openpi/assets/gxd_pi05_v3src_pad/local/openpi_V5_mcap0625_v3src_pad")
CACHE_DIR = os.path.expanduser("~/.cache/huggingface/lerobot/local")
DS_LINK = os.path.join(CACHE_DIR, "openpi_V5_mcap0625_v3src_pad")

os.makedirs(CACHE_DIR, exist_ok=True)
if not os.path.exists(DS_LINK):
    os.symlink(DS_SRC, DS_LINK)
    print(f"[setup] Created symlink: {DS_LINK} -> {DS_SRC}")

# Run training (import after symlink is set up)
os.chdir(os.path.expanduser("~/openpi"))
sys.argv = ["train.py", "debug_encoder_norm"]

from scripts.train import main as train_main
import openpi.training.config as _config

train_main(_config.cli())

# Cleanup
if os.path.islink(DS_LINK):
    os.unlink(DS_LINK)
    print(f"[cleanup] Removed symlink: {DS_LINK}")
