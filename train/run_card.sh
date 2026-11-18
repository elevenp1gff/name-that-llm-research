#!/usr/bin/env bash
# Launch one training/inference process on one Ascend card.
#
#   bash run_card.sh <card> [--min_free_gb N] [--stop_at_hbm_mb N] -- <python args...>
#   bash run_card.sh 3 --min_free_gb 40 -- \
#       train_lora.py --model .../Qwen3-14B-Base --layers 27 ...
#
# Behaviour
#   1. source the Ascend toolkit (without it, importing libhccl.so fails)
#   2. re-verify the card is actually free via `npu-smi info`  -> exit 90 if not
#   3. exec the python process (no subshell: signals reach python, so `kill <pid>`
#      works and `nohup ... &` leaves a single, findable pid)
#
# Exit codes
#    0   success
#   90   card busy / too little free HBM (pick another card or retry later)
#   91   npu-smi unavailable or its output could not be parsed
#   other: passed through from python
#
# Safe under nohup: nothing here reads stdin, and all messages go to stderr so
# the stdout log stays pure python.
set -uo pipefail

PY="${PY:-${NTL_WORK:?set NTL_WORK to your workspace root}/miniconda3/envs/lht/bin/python}"
ASCEND_ENV="${ASCEND_ENV:-/usr/local/Ascend/ascend-toolkit/set_env.sh}"
MIN_FREE_GB="${MIN_FREE_GB:-30}"       # free HBM required (measured peak 21.8 GB + margin)
HWM_TOTAL_MB="${HWM_TOTAL_MB:-65536}"  # 910B3 = 64 GB

die() {
  local msg="$1" code="${2:-1}"
  printf '[run_card] ERROR: %s\n' "$msg" >&2
  exit "$code"
}
log() { printf '[run_card] %s\n' "$*" >&2; }

# ---------------------------------------------------------------- arg parsing
CARD=""
while [[ $# -gt 0 ]]; do
  case "$1" in
    --min_free_gb)   MIN_FREE_GB="${2:?--min_free_gb needs a value}"; shift 2 ;;
    --min_free_gb=*) MIN_FREE_GB="${1#*=}"; shift ;;
    --stop_at_hbm_mb) HWM_TOTAL_MB="${2:?}"; shift 2 ;;
    --) shift; break ;;
    -h|--help) sed -n '2,25p' "$0"; exit 0 ;;
    *)
      if [[ -z "$CARD" ]]; then CARD="$1"; shift; else break; fi
      ;;
  esac
done

[[ -n "$CARD" ]] || die "usage: bash run_card.sh <card-id> [opts] -- <python args...>" 2
case "$CARD" in
  ''|*[!0-9]*) die "card id must be an integer 0-7, got '$CARD'" 2 ;;
esac
(( CARD >= 0 && CARD <= 7 )) || die "card id out of range 0-7: $CARD" 2
[[ $# -ge 1 ]] || die "no python command given (use -- before it)" 2
[[ -n "$1" ]] || die "empty python command" 2

# ------------------------------------------------------------ Ascend toolkit
[[ -f "$ASCEND_ENV" ]] || die "Ascend env not found: $ASCEND_ENV" 3
# CRITICAL: set_env.sh *prepends* /usr/local/Ascend/cann-*/python/site-packages and
# opp/built-in/op_impl/ai_core/tbe to PYTHONPATH. Overwriting PYTHONPATH after this
# point makes aclInit fail with "No module named 'tbe'" -> GEInitialize failed,
# which looks like a broken driver rather than a missing env var. Append only.
#
# set -u must be suspended while sourcing it: set_env.sh reads LD_LIBRARY_PATH,
# PYTHONPATH and CMAKE_PREFIX_PATH unguarded and would abort mid-source.
set +u
# shellcheck disable=SC1090
source "$ASCEND_ENV"
set -u
log "sourced $ASCEND_ENV"

VENDOR="${VENDOR:-${NTL_WORK:?set NTL_WORK to your workspace root}/vendor}"
if [[ -d "$VENDOR" ]]; then
  # append, never prepend-with-overwrite: set_env.sh's tbe path must survive
  export PYTHONPATH=".:${PYTHONPATH:-}:${VENDOR}"
else
  export PYTHONPATH=".:${PYTHONPATH:-}"
  log "WARNING: $VENDOR missing -- transformers cannot import Qwen3 (needs einops)."
  log "         Run jobs/bootstrap_deps.sh once."
fi
[[ -x "$PY" ]] || die "python not executable: $PY" 3

export PYTHONUNBUFFERED=1
export PYTHONIOENCODING=utf-8
export ASCEND_RT_VISIBLE_DEVICES="$CARD"
# 1 makes every op run eagerly (slow, but points at the exact failing kernel).
export ASCEND_LAUNCH_BLOCKING="${ASCEND_LAUNCH_BLOCKING:-0}"
# Explicit BLAS threads: multi-process CPU fallbacks otherwise oversubscribe 192 cores.
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-12}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-12}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-1}"

# ------------------------------------------------------------------ HBM check
# `npu-smi info` prints one row per card whose memory column holds three "a/b"
# pairs -- hugepages (0/0), memory-usage (0/0) and HBM (used/total, in MB).
# `npu-smi info -t memory -i N` reports capacity only, no usage, so parse the table.
check_hbm() {
  command -v npu-smi >/dev/null 2>&1 || return 91
  CARD_ID="$CARD" MIN_FREE_GB="$MIN_FREE_GB" HWM_TOTAL="$HWM_TOTAL_MB" \
    timeout 60 "$PY" - <<'PYEOF'
import os, re, subprocess, sys

card = int(os.environ["CARD_ID"])
min_free = float(os.environ["MIN_FREE_GB"]) * 1024.0
expect_total = int(os.environ["HWM_TOTAL"])

try:
    out = subprocess.run(["npu-smi", "info"], capture_output=True, text=True, timeout=60).stdout
except Exception as exc:
    sys.stderr.write(f"[run_card] npu-smi failed: {exc}\n")
    sys.exit(91)
if not out.strip():
    sys.stderr.write("[run_card] npu-smi returned nothing\n")
    sys.exit(91)

rows = []
for line in out.splitlines():
    if "|" not in line:
        continue
    # HBM is the only pair whose denominator is a real capacity in MB.
    pairs = [(int(a), int(b)) for a, b in re.findall(r"(\d+)\s*/\s*(\d+)", line)]
    hbm = [p for p in pairs if p[1] > 1000 and abs(p[1] - expect_total) <= 1024]
    if hbm:
        rows.append(hbm[-1])

if len(rows) < card + 1:
    sys.stderr.write(f"[run_card] could not parse HBM for card {card}: "
                     f"found {len(rows)} card row(s) in npu-smi output\n")
    sys.exit(91)

used, total = rows[card]
free = total - used
sys.stderr.write(f"[run_card] card {card}: HBM {used}/{total} MB used, {free} MB free, "
                 f"need {min_free:.0f} MB\n")
sys.exit(0 if free >= min_free else 90)
PYEOF
}

check_hbm
rc=$?
case "$rc" in
  0)  : ;;
  90) die "card $CARD has less than ${MIN_FREE_GB} GB free HBM (exit 90). Pick another card." 90 ;;
  91) die "npu-smi unavailable or unparsable (exit 91); cannot verify the card is free." 91 ;;
  *)  die "HBM check failed with code $rc" "$rc" ;;
esac

# --------------------------------------------------------------------- launch
log "card $CARD -> $*"
if [[ -n "${RUN_CARD_DRY_RUN:-}" ]]; then
  log "RUN_CARD_DRY_RUN set; not launching"
  exit 0
fi
exec "$@"
