#!/usr/bin/env bash
# Copy LongBench results and logs from DSMLP back to this Mac. Run ON THE MAC
# from the repo root:
#     bash scripts/fetch_results.sh
#
# Uses the `dsmlp` host from ~/.ssh/config. If the shared connection has
# expired, open it first in a terminal (Duo push):  ssh -fN dsmlp
#
# Paths: inside a pod, ~/private is your persistent DSMLP home; on the login
# jumpbox (`dsmlp`) that same home is simply ~. So the pod's
# ~/private/dsc180-quantization is ~/dsc180-quantization on the jumpbox.
# Override with REMOTE_REPO=<path relative to the jumpbox home> if needed.

set -euo pipefail

HOST="${HOST:-dsmlp}"
REMOTE_REPO="${REMOTE_REPO:-}"
cd "$(dirname "$0")/.."

ssh -o BatchMode=yes "$HOST" true 2>/dev/null \
    || { echo "ERROR: cannot reach $HOST without a prompt. Run: ssh -fN $HOST  (approve Duo), then retry."; exit 1; }

if [ -z "$REMOTE_REPO" ]; then
    for candidate in dsc180-quantization private/dsc180-quantization; do
        if ssh -o BatchMode=yes "$HOST" "test -d $candidate/results" 2>/dev/null; then
            REMOTE_REPO="$candidate"; break
        fi
    done
    [ -n "$REMOTE_REPO" ] || { echo "ERROR: no results found on $HOST (tried ~/dsc180-quantization and ~/private/dsc180-quantization). Set REMOTE_REPO."; exit 1; }
fi
echo "Fetching from $HOST:~/$REMOTE_REPO"

mkdir -p results/longbench logs
# rsync only copies new/changed files, so it is cheap to run repeatedly.
rsync -av "$HOST:$REMOTE_REPO/results/longbench/" results/longbench/
rsync -av "$HOST:$REMOTE_REPO/logs/" logs/ 2>/dev/null || echo "(no logs/ on remote yet)"

echo
echo "Runs now on this Mac:"
ls results/longbench/
echo "Re-score a run locally:  uv run python src/evaluation/run_longbench.py --run-name <run> --score-only"
