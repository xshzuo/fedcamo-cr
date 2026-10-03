#!/usr/bin/env bash
# FedCAMO-CR matrix runner — simple version using xargs -P
# Usage: bash experiments/run_matrix_simple.sh [rounds] [seeds] [nproc]
set -e

ROUNDS="${1:-200}"
SEEDS="${2:-3}"
NPROC="${3:-24}"

cd "$(dirname "$0")/.."
ROOT=$(pwd)
export PATH=$HOME/.local/bin:$PATH

mkdir -p logs

ALGOS=(fedavg fedcamo_pr fedcamo_cb fedcamo_cr_pr fedcamo_cr_cb)
AGGS=(fedavg krum trimmed_mean foolsgold trust_weighted)
ATTACKS_RATIOS=("none 0.0" "lf 0.3" "mr 0.3" "aa 0.3")

run_one() {
  local algo=$1 agg=$2 attack=$3 ratio=$4 seed=$5
  python3 experiments/run_robust.py \
    --scenario baseline \
    --algo "$algo" --agg "$agg" \
    --attack "$attack" --attack-ratio "$ratio" \
    --rounds "$ROUNDS" --seeds 1 \
    --out-root "$ROOT/results" \
    > "$ROOT/logs/${algo}_${agg}_${attack}_r${ratio}_s${seed}.log" 2>&1
}

export -f run_one
export ROUNDS ROOT

> /tmp/fedcamo_jobs.txt
for algo in "${ALGOS[@]}"; do
  for agg in "${AGGS[@]}"; do
    for ar in "${ATTACKS_RATIOS[@]}"; do
      attack=$(echo $ar | cut -d' ' -f1)
      ratio=$(echo $ar | cut -d' ' -f2)
      for ((seed=0; seed<SEEDS; seed++)); do
        echo "$algo $agg $attack $ratio $seed" >> /tmp/fedcamo_jobs.txt
      done
    done
  done
done

TOTAL=$(wc -l < /tmp/fedcamo_jobs.txt)
echo "Total runs: $TOTAL  rounds=$ROUNDS seeds=$SEEDS nproc=$NPROC"
date

cat /tmp/fedcamo_jobs.txt | xargs -n 5 -P "$NPROC" bash -c 'run_one "$@"' _

echo "=== done at $(date) ==="
ls results/baseline | wc -l
