#!/bin/bash

# Handle env constants

export TRDLM_BASE_DIR="$HOME/.cache/trdlm"
mkdir -p $TRDLM_BASE_DIR

# Handle uv environment

uv sync
source .venv/bin/activate
