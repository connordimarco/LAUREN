#!/usr/bin/env bash
# AU/AL corrector per-minute tick wrapper.
#
# NOT YET IN CRONTAB — the service is switched on together with Connor
# once scoped. To enable:
#   * * * * * bash /data/tuija/cdimarco/LAUREN/aual_tick.sh >> /data/tuija/cdimarco/LAUREN/logs/cron_tick.log 2>&1
#
# flock -n : if the previous tick still runs, skip (skip beats queue)
# sleep 15 : this minute's IMF_14Re lands ~:10-:14; grid batches land at
#            arbitrary seconds — anything missed is caught next tick
# timeout  : a wedged tick dies at 150 s (+10 s SIGKILL grace)
set -u
ROOT=/data/tuija/cdimarco/LAUREN
export TMPDIR=/data/tuija/cdimarco/tmp
# Single-core budget: LightGBM/BLAS otherwise spawn a thread per core.
# Measured cost is unchanged (~5 s, 32% of one core — the tick is I/O-bound).
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
mkdir -p "$ROOT/state" "$ROOT/logs" "$ROOT/realtime-aual" "$ROOT/realtime-f107"

exec 9>"$ROOT/state/tick.lock"
flock -n 9 || exit 0
sleep 15
cd "$ROOT"
nice timeout -k 10 150 venv/bin/python run_aual_tick.py >/dev/null

# ---- push the SME product to SWORD:/data/Gitm/cdimarco/LAUREN ---------------
# realtime-aual/ only: realtime-midl/ and realtime-f107/ ship via the MIDL
# relay (realtime_sword_push.sh), which runs per minute with or without
# this tick — one transport owner per product.
# DORMANT until state/sword_push_enabled exists AND the dedicated key is
# installed. The SWORD side needs a write-only rrsync jail chrooted to
# /data/Gitm/cdimarco/LAUREN (mirror image of this box's realtime-geodgp
# jail), because a key carries exactly one forced command — the MIDL relay
# key is jailed to the MIDL mirror and cannot be reused for this path.
# Non-fatal: a dead push leaves a stale SWORD copy and one log line.
KEY="$HOME/.ssh/id_ed25519_sword_aual"
FLAG="$ROOT/state/sword_push_enabled"
if [ -f "$FLAG" ] && [ -f "$KEY" ] && [ -f "$ROOT/realtime-aual/aual.dat" ]; then
    RSH="ssh -i $KEY -o BatchMode=yes -o ConnectTimeout=10 -o StrictHostKeyChecking=accept-new"
    # --exclude protects SWORD's adapter output (gitm_aual.dat, written
    # next to ours for inspection) from this push's --delete
    err=$(nice timeout -k 5 45 rsync -a --delete --exclude='gitm_aual.dat' \
            --timeout=30 -e "$RSH" \
            "$ROOT/realtime-aual" \
            "cdimarco@csrsvr2024-0246.engin.umich.edu:/" 2>&1 >/dev/null) \
        || echo "$(date -u +%FT%TZ) PUSH-FAIL $(printf '%s\n' "$err" | grep -v '^\*' | grep -v '^$' | tail -1)" \
            >> "$ROOT/logs/push_$(date -u +%Y%m).log"
fi
exit 0
