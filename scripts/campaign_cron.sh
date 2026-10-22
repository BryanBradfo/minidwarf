#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# Daily campaign step for cron. Usage: campaign_cron.sh <runs_dir> <config:samples>...
# Reads GEMINI_API_KEY from ~/.zshrc (never printed); one step at a time (flock).
set -u
cd "$(dirname "$0")/.."
runs="$1"; shift
eval "$(grep -m1 '^export GEMINI_API_KEY=' "$HOME/.zshrc")"
export PATH="/usr/local/cuda/bin:$HOME/miniconda3/bin:$PATH" LD_LIBRARY_PATH=/usr/local/cuda/lib64
export PYTHONPATH="$PWD"  # import this checkout, not whichever branch the editable install points at
mkdir -p "$runs"
exec flock -n "$runs/.campaign.lock" python scripts/run_campaign.py --runs-dir "$runs" "$@" >> "$runs/campaign.log" 2>&1
