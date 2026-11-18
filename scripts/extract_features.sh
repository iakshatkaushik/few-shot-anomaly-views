#!/bin/bash
# Cache test/shot tokens, masks, views and synthetic anomalies for a list of datasets.
# Re-running skips shards that already exist.
# usage: scripts/extract_features.sh <encoder> <sets|all> <dataset> [dataset...]
set -e
cd "$(dirname "$0")/.."
PY=${PY:-python}
ENC=$1; SETS=$2; shift 2
SA=""; [ "$SETS" != "all" ] && SA="--sets $SETS"
$PY -m features.extract --encoder $ENC --datasets "$@" --kinds test shots $SA
$PY -m features.extract --masks --datasets "$@"
$PY -m features.views --encoder $ENC --datasets "$@" $SA
$PY -m features.synth --encoder $ENC --datasets "$@" $SA
