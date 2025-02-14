#!/bin/bash

# Install PyTorch with CPU support
python -m pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu

# Install other dependencies
python -m pip install transforms3d scipy pyyaml psutil

echo "All dependencies installed successfully!"