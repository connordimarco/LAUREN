#!/usr/bin/env bash
# LAUREN AU/AL watchdog wrapper (5-minute cron) — see run_aual_watchdog.py.
#
# crontab:
#   3-59/5 * * * * bash /data/tuija/cdimarco/LAUREN/aual_watchdog.sh >> /data/tuija/cdimarco/LAUREN/logs/cron_watchdog.log 2>&1
# (offset from :00/:05 so it reads post-tick state, and clear of the MIDL
# watchdog's :00/:05 and the GITM watchdog's :02/:07 slots)
#
# Sources the workspace .env for RESEND_API_KEY (shared Resend notifier).
set -u
ROOT=/data/tuija/cdimarco/LAUREN
export TMPDIR=/data/tuija/cdimarco/tmp
set -a
[ -f /data/tuija/cdimarco/.env ] && . /data/tuija/cdimarco/.env
set +a
cd "$ROOT"
exec venv/bin/python run_aual_watchdog.py
