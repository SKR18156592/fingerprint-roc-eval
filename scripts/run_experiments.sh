#!/usr/bin/env bash
# Reproduce every experiment reported in docs/report.md.
# Assumes SOCOFing is extracted under data/raw/socofing (see README).
set -euo pipefail
cd "$(dirname "$0")/.."
PY=${PYTHON:-python}
OUT=results/experiments

run() {  # run <name> <gallery dir> <collect_scores args...>
  local name=$1 data=$2; shift 2
  mkdir -p "$OUT/$name"
  echo "=== $name ==="
  $PY collect_scores.py --data "$data" --out "$OUT/$name/scores.json" "$@"
  $PY roc_analysis.py --scores "$OUT/$name/scores.json" --out-dir "$OUT/$name" > "$OUT/$name/report.txt"
  grep -E "^(EER|TAR|Recommended)" "$OUT/$name/report.txt"
}

# 1) Main setting: 100 identities x 5 captures with acquisition variation.
$PY scripts/prepare_socofing.py --people 100 --captures 5 --out data/subjects
run sift_100x5_cross data/subjects --matcher sift
run orb_100x5_cross  data/subjects --matcher orb

# 2) Raw SOCOFing (altered copies, no simulated variation) — shows how easy
#    the unmodified dataset is.
$PY scripts/prepare_socofing.py --people 100 --captures 5 --no-variation --out data/subjects_raw
run sift_100x5_raw data/subjects_raw --matcher sift

# 3) Minimal protocol: 10 identities x 3 captures, one reference per person
#    for impostors (45 pairs) vs. all cross-captures (405 pairs).
$PY scripts/prepare_socofing.py --people 10 --captures 3 --out data/subjects_10x3
run sift_10x3_reference data/subjects_10x3 --matcher sift --impostor-mode reference
run sift_10x3_cross     data/subjects_10x3 --matcher sift --impostor-mode cross
