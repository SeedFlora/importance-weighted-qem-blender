#!/usr/bin/env bash
# Evaluation runs as pre-registered in PREREGISTRATION.md (Amandemen 1). Run from the project root:
#   bash results/evaluation/run_all.sh
set -u
B="${BLENDER:-/c/Program Files/Blender Foundation/Blender 5.1/blender.exe}"
OUT=results/evaluation
COMMON=(--split evaluation --no-preserve-seams --resolution 512 --ratios 0.5,0.2,0.05 --seed 20260921 --samples 4000)
FINAL=(--strength 12 --weights "saliency=0,silhouette=1,sharp=0")
run() {  # name, then run_experiment arguments
  local name=$1; shift
  "$B" --background --factory-startup --python-exit-code 1 --python scripts/run_experiment.py -- "$@" \
    > "$OUT/$name.log" 2>&1
  echo "$name exit=$?"
}

# 1) Quality (deterministic): three asset groups in parallel, LOD meshes saved for figures and the user study.
run quality_A --out "$OUT/quality_A" --assets max_planck,bunny,fandisk "${COMMON[@]}" "${FINAL[@]}" \
  --experiments main,ablation,sweep --lambdas 0,1,2,3,5,8,12,20 --save-lods &
run quality_B --out "$OUT/quality_B" --assets igea,horse,rocker_arm "${COMMON[@]}" "${FINAL[@]}" \
  --experiments main,ablation,sweep --lambdas 0,1,2,3,5,8,12,20 --save-lods &
run quality_C --out "$OUT/quality_C" --assets ogre,armadillo,beetle "${COMMON[@]}" "${FINAL[@]}" \
  --experiments main,ablation,sweep --lambdas 0,1,2,3,5,8,12,20 --save-lods &
wait

# 2) Timing: sequential, nothing else running; 1 warm-up + 3 repetitions, method order shuffled per repetition.
run timing --out "$OUT/timing" "${COMMON[@]}" "${FINAL[@]}" --experiments timing --reps 3 --warmup 1

# 3) Exploratory only (never used for selection): the old default channel weights.
run explore_W1 --out "$OUT/explore_W1" "${COMMON[@]}" --strength 12 \
  --weights "saliency=0.35,silhouette=0.4,sharp=0.25" --experiments main,sweep --lambdas 0,2,5,12
echo ALLDONE
