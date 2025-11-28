#!/usr/bin/env bash
set -euo pipefail

COMMON_ARGS=(
  --dataset karel_dataset_experiments.jls
  --prompts-dir ../prompts
  --runs-dir ../runs
  --model deepseek-chat
  --nrows 8
)

julia main.jl "${COMMON_ARGS[@]}" \
  --llm_enabled true --traces_enabled false --updating_enabled false --depth_aware_enabled true

julia main.jl "${COMMON_ARGS[@]}" \
  --llm_enabled true --traces_enabled false --updating_enabled false --depth_aware_enabled false

julia main.jl "${COMMON_ARGS[@]}" \
  --llm_enabled true --traces_enabled false --updating_enabled false --depth_aware_enabled true --smoothing-enabled true







