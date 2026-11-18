#!/usr/bin/env bash
# Convenience launcher for a full stage1+stage2 run: fills in paths, guards against
# double-launch and half-downloaded weights, then hands off to run_card.sh.
#
#   bash run_train.sh <card> <tag> [extra train_lora.py args...]
#   bash run_train.sh 3 q14 --model .../Qwen3-14B-Base --layers 27 --bs 8
#
# Detached launch (the only supported way for a multi-hour run):
#   cd ${NTL_WORK:?set NTL_WORK to your workspace root}/jobs
#   nohup bash run_train.sh 3 q14 --model ... --layers 27 --bs 8 \
#       > ../logs/q14.log 2>&1 &
#   echo $! > ../logs/q14.pid
#
# Layout assumed under ${NTL_WORK:?set NTL_WORK to your workspace root}: data/ models/ out/ logs/ jobs/
set -uo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$HERE/.." && pwd)"

CARD="${1:?usage: bash run_train.sh <card> <tag> [train args...]}"; shift
TAG="${1:?usage: bash run_train.sh <card> <tag> [train args...]}"; shift

DATA="${DATA:-$ROOT/data}"
OUT_ROOT="${OUT_ROOT:-$ROOT/out}"
LOG_DIR="${LOG_DIR:-$ROOT/logs}"
MIN_FREE_GB="${MIN_FREE_GB:-30}"
PY="${PY:-${NTL_WORK:?set NTL_WORK to your workspace root}/miniconda3/envs/lht/bin/python}"
mkdir -p "$LOG_DIR" "$OUT_ROOT"

OUT_DIR="$OUT_ROOT/$TAG"
MARKER="$LOG_DIR/$TAG.running"

# --- single-instance guard -------------------------------------------------- #
# No flock on this NFS mount, so use an O_EXCL claim file holding the owner pid.
# The claim is deliberately NOT removed by an EXIT trap: the final `exec` below
# replaces this process, so no trap could ever fire, and deleting on exit would
# open a race between two overlapping starts. Instead a stale claim (owner pid
# gone) is reclaimed on the next start -- which also covers a node that died
# without running any cleanup. Verified by test_guard.sh (3 cases).
if ( set -o noclobber; printf '%s %s\n' "$$" "$(date -Is)" > "$MARKER" ) 2>/dev/null; then
  :
else
  read -r old_pid _ < "$MARKER" 2>/dev/null || old_pid=""
  if [[ -n "$old_pid" ]] && kill -0 "$old_pid" 2>/dev/null; then
    printf '[run_train] "%s" is already running (pid %s, %s).\n' \
      "$TAG" "$old_pid" "$(cut -d' ' -f2- "$MARKER")" >&2
    printf '[run_train] stop it first: kill %s   (or rm -f %s if that pid is wrong)\n' \
      "$old_pid" "$MARKER" >&2
    exit 88
  fi
  printf '[run_train] reclaiming stale claim on "%s" (pid %s is gone)\n' \
    "$TAG" "${old_pid:-?}" >&2
  printf '%s %s\n' "$$" "$(date -Is)" > "$MARKER"
fi

# --- inputs ------------------------------------------------------------------ #
STAGE1="${STAGE1:-$DATA/union_capped.parquet}"
STAGE2="${STAGE2:-$DATA/official.parquet}"
for f in "$STAGE1" "$STAGE2"; do
  [[ -f "$f" ]] || { printf '[run_train] missing data file: %s\n' "$f" >&2; exit 2; }
done
[[ -x "$PY" ]] || { printf '[run_train] python not executable: %s\n' "$PY" >&2; exit 3; }

# Model dir: must be complete. A live HF download leaves *.part and an incomplete
# config, and loading it would fail 20 minutes into the run instead of instantly.
MODEL_DIR=""
prev=""
for a in "$@"; do
  case "$a" in --model=*) MODEL_DIR="${a#--model=}" ;; esac
  [[ "$prev" == "--model" ]] && MODEL_DIR="$a"
  prev="$a"
done
if [[ -n "$MODEL_DIR" ]]; then
  if compgen -G "$MODEL_DIR/*.part" >/dev/null 2>&1; then
    printf '[run_train] %s still contains .part files -- download in progress:\n' "$MODEL_DIR" >&2
    ls -la "$MODEL_DIR" >&2
    exit 3
  fi
  [[ -f "$MODEL_DIR/config.json" ]] \
    || { printf '[run_train] %s has no config.json\n' "$MODEL_DIR" >&2; exit 3; }
  printf '[run_train] model: %s\n' "$MODEL_DIR"
fi

printf '[run_train] tag=%s card=%s out=%s\n' "$TAG" "$CARD" "$OUT_DIR"
printf '[run_train] started %s on %s (pid %s)\n' "$(date -Is)" "$(hostname)" "$$"

# exec chain: nohup's $! is this pid, which becomes the python pid after two execs,
# so `kill <pid>` reaches the trainer directly with no orphaned children.
exec bash "$HERE/run_card.sh" "$CARD" --min_free_gb "$MIN_FREE_GB" -- \
  "$PY" -u "$HERE/train_lora.py" \
  --stage1 "$STAGE1" --stage2 "$STAGE2" --out "$OUT_DIR" "$@"
