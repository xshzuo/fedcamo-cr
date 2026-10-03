#!/usr/bin/env bash
# FedCAMO-CR full matrix runner
# 在 192.168.0.105 上跑全实验矩阵
# 用法：bash experiments/run_matrix.sh [scenario] [rounds] [seeds]
# 默认：baseline scenario, 200 rounds, 3 seeds
set -e

SCENARIO="${1:-baseline}"
ROUNDS="${2:-200}"
SEEDS="${3:-3}"
NPROC="${4:-48}"   # HP Z840 有 48 核

cd "$(dirname "$0")/.."
ROOT=$(pwd)
export PATH=$HOME/.local/bin:$PATH

mkdir -p logs results

echo "=== FedCAMO-CR full matrix ==="
echo "scenario=$SCENARIO rounds=$ROUNDS seeds=$SEEDS nproc=$NPROC"
echo "ROOT=$ROOT"
date

ALGOS=(
  fedavg fedcamo_pr fedcamo_cb fedcamo_cr_pr fedcamo_cr_cb
)
AGGS=(
  fedavg krum trimmed_mean foolsgold trust_weighted
)
ATTACKS_RATIOS=(
  "none 0.0"
  "lf 0.3"
  "mr 0.3"
  "aa 0.3"
)

run_one() {
  local algo=$1 agg=$2 attack=$3 ratio=$4 seed=$5
  python3 experiments/run_robust.py \
    --scenario "$SCENARIO" \
    --algo "$algo" --agg "$agg" \
    --attack "$attack" --attack-ratio "$ratio" \
    --rounds "$ROUNDS" --seeds 1 \
    --out-root "$ROOT/results" \
    > "$ROOT/logs/${algo}_${agg}_${attack}_r${ratio}_s${seed}.log" 2>&1
}

export -f run_one
export SCENARIO ROUNDS ROOT

# 构造所有 (algo, agg, attack, ratio, seed) 组合
JOBS=()
for algo in "${ALGOS[@]}"; do
  for agg in "${AGGS[@]}"; do
    # fedcamo_cr_* 不需要单独接 agg：trust_weighted 主要路径
    # 但我们要保留所有组合用于 ablation
    for ar in "${ATTACKS_RATIOS[@]}"; do
      attack=$(echo $ar | cut -d' ' -f1)
      ratio=$(echo $ar | cut -d' ' -f2)
      for ((seed=0; seed<SEEDS; seed++)); do
        JOBS+=("$algo $agg $attack $ratio $seed")
      done
    done
  done
done

TOTAL=${#JOBS[@]}
echo "Total runs: $TOTAL"

# GNU parallel 如果有就用，没有就用 xargs -P
if command -v parallel >/dev/null 2>&1; then
  printf "%s\n" "${JOBS[@]}" | parallel -j "$NPROC" --colsep ' ' run_one {1} {2} {3} {4} {5}
else
  printf "%s\n" "${JOBS[@]}" | xargs -n 5 -P "$NPROC" bash -c 'run_one "$@"' _
fi

echo "=== done at $(date) ==="
