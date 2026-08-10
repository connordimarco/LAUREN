#!/usr/bin/env python3
"""LAUREN watchdog — health of the AU/AL corrector service (5-minute cron).

Follows the GITM-RT watchdog pattern (MIDL-Pipeline/dev/run_gitm_rt_watchdog.py):
independent conditions, one alert per condition per RATE_LIMIT_H hours plus a
recovery email when it clears, via the shared Resend notifier in ../notify/.

Conditions:

  tick-dead        no tick log line at all in TICK_STALE_MIN (cron off/wedged)
  tick-failing     ticks running but FAILing for >= FAIL_RUN_MIN
  no-model         latest tick reports 'no-model' (model file vanished)
  grids-stale      newest realtime-geodgp file > GRIDS_STALE_MIN old
                   (SWORD-side GeoDGP push died; their cadence is ~3 min)
  frontier-stale   aual.dat's last row < FRONTIER_MIN_LEAD ahead of now —
                   the MIDL 14 Re frontier stopped advancing (normally the
                   file ends ~+55 min in the future)
  push-failing     PUSH-FAIL lines in logs/push_*.log spanning >= PUSH_FAIL_MIN
  herot-aual-failing  the relay's aual views publish retry marker is set and
                   no successful publish for HEROT_STALE_MIN

Boot-grace: everything is suppressed for the first BOOT_GRACE_MIN of local
uptime (this box patch-reboots Wednesday mornings); grids-stale/push-failing
are additionally suppressed for SWORD_REBOOT_GRACE_MIN when the outage starts
in a Wednesday 07:00-08:00 ET window (SWORD's own patch reboot).

Testing: thresholds are env-overridable (AUAL_WD_*), and AUAL_WD_DRY_RUN=1
makes the notifier print instead of send — e.g. a live alert+recovery drill:
  AUAL_WD_FRONTIER_MIN_LEAD=999 bash aual_watchdog.sh   # fires
  bash aual_watchdog.sh                                 # recovers
"""
import json
import os
import subprocess
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent
LOGS = ROOT / 'logs'
STATE = ROOT / 'state'
GRIDS = ROOT / 'realtime-geodgp'
AUAL = ROOT / 'realtime-aual' / 'aual.dat'
MODEL_FLAGLINE = 'no-model'
STATE_FILE = STATE / 'watchdog_state.json'
DETAIL_FILE = STATE / 'watchdog_last_detail.txt'
WD_LOG = LOGS / 'watchdog.log'
NOTIFY = Path('/data/tuija/cdimarco/notify/send_refresh_email.py')
PY = str(ROOT / 'venv' / 'bin' / 'python')
MAIL_FROM = 'Realtime <realtime@connordimarco.com>'
# the relay publishes our website views; its retry marker lives in MIDL state
MIDL_STATE = Path('/data/tuija/cdimarco/MIDL/realtime/state')
MIDL_LOGS = Path('/data/tuija/cdimarco/MIDL/realtime/logs')
AUAL_PUB_RETRY = MIDL_STATE / 'aual_pub_retry'

TS_FMT = '%Y-%m-%dT%H:%M:%S'


def _envf(name, default):
    try:
        return float(os.environ.get(name, default))
    except ValueError:
        return default


TICK_STALE_MIN = _envf('AUAL_WD_TICK_STALE_MIN', 7)
FAIL_RUN_MIN = _envf('AUAL_WD_FAIL_RUN_MIN', 10)
GRIDS_STALE_MIN = _envf('AUAL_WD_GRIDS_STALE_MIN', 15)
FRONTIER_MIN_LEAD = _envf('AUAL_WD_FRONTIER_MIN_LEAD', 10)
PUSH_FAIL_MIN = _envf('AUAL_WD_PUSH_FAIL_MIN', 10)
HEROT_STALE_MIN = _envf('AUAL_WD_HEROT_STALE_MIN', 20)
RATE_LIMIT_H = _envf('AUAL_WD_RATE_LIMIT_H', 6)
BOOT_GRACE_MIN = 15
SWORD_REBOOT_GRACE_MIN = 40


def _entries(logdir, prefix, limit=120):
    """[(utc_dt, ok, line)] from the newest two monthly logs, oldest first.
    Accepts both 'TS OK ...' (tick) and 'TSZ PUSH-FAIL/OK/FAIL ...' lines."""
    out = []
    for path in sorted(logdir.glob(prefix + '_*.log'))[-2:]:
        try:
            lines = path.read_text().splitlines()[-limit:]
        except OSError:
            continue
        for line in lines:
            parts = line.split()
            if len(parts) < 2:
                continue
            try:
                ts = datetime.strptime(parts[0].rstrip('Z'), TS_FMT)
            except ValueError:
                continue
            out.append((ts, parts[1] == 'OK', line))
    out.sort(key=lambda e: e[0])
    return out


def _in_sword_reboot_window(now, start_utc):
    try:
        from zoneinfo import ZoneInfo
        start_et = start_utc.replace(tzinfo=timezone.utc).astimezone(
            ZoneInfo('America/New_York'))
    except Exception:
        start_et = start_utc - timedelta(hours=4)
    return (start_et.weekday() == 2 and start_et.hour == 7
            and now - start_utc < timedelta(minutes=SWORD_REBOOT_GRACE_MIN))


def _newest_grid_age_min(now):
    newest = None
    try:
        for p in GRIDS.glob('deltaB_*_real4.out'):
            m = p.stat().st_mtime
            if newest is None or m > newest:
                newest = m
    except OSError:
        return None
    if newest is None:
        return None
    return (now - datetime.utcfromtimestamp(newest)).total_seconds() / 60


def _aual_last_row(now):
    """Minutes the last aual.dat row leads (+) or trails (-) the clock."""
    try:
        line = AUAL.read_text().splitlines()[-1].split()
        t = datetime(*map(int, line[:6]))
    except (OSError, ValueError, IndexError, TypeError):
        return None
    return (t - now).total_seconds() / 60


def evaluate(now):
    ticks = _entries(LOGS, 'tick')
    conditions = {}

    # 1/2/3 — the tick itself.
    recent = [e for e in ticks if now - e[0] <= timedelta(minutes=TICK_STALE_MIN)]
    if not recent:
        conditions['tick-dead'] = (
            f'no AU/AL tick log line in {TICK_STALE_MIN:.0f} min — '
            'cron off or tick wedged')
    else:
        oks = [ts for ts, ok, _ in ticks if ok]
        last_ok = max(oks) if oks else None
        if last_ok is None or now - last_ok > timedelta(minutes=FAIL_RUN_MIN):
            fails = [ts for ts, ok, _ in ticks
                     if not ok and (last_ok is None or ts > last_ok)]
            if fails:
                mins = (now - min(fails)).total_seconds() / 60
                conditions['tick-failing'] = (
                    f'AU/AL tick FAILing for {mins:.0f} min '
                    f'(threshold {FAIL_RUN_MIN:.0f}); see logs/tick log tail')
        elif MODEL_FLAGLINE in recent[-1][2]:
            conditions['no-model'] = (
                'tick reports no-model: the corrector .joblib is missing')

    # 4 — inbound GeoDGP feed.
    grid_age = _newest_grid_age_min(now)
    if grid_age is None or grid_age > GRIDS_STALE_MIN:
        start = (now - timedelta(minutes=grid_age)) if grid_age else now
        if not _in_sword_reboot_window(now, start):
            conditions['grids-stale'] = (
                f'no fresh GeoDGP grids: newest is '
                f'{"absent" if grid_age is None else f"{grid_age:.0f} min old"} '
                f'(threshold {GRIDS_STALE_MIN:.0f}) — SWORD-side push or '
                'GeoDGP realtime is down')

    # 5 — MIDL frontier (only meaningful while the tick is healthy).
    if 'tick-dead' not in conditions and 'tick-failing' not in conditions:
        lead = _aual_last_row(now)
        if lead is not None and lead < FRONTIER_MIN_LEAD:
            conditions['frontier-stale'] = (
                f'aual.dat forecast frontier is only {lead:.0f} min ahead '
                f'(normally ~+55, threshold {FRONTIER_MIN_LEAD:.0f}) — the '
                'MIDL 14 Re feed has likely stopped advancing')

    # 6 — our SWORD push.
    pushes = _entries(LOGS, 'push')
    pfails = [ts for ts, ok, line in pushes if 'PUSH-FAIL' in line]
    if pfails and now - max(pfails) <= timedelta(minutes=3):
        run = [ts for ts in pfails
               if (max(pfails) - ts) <= timedelta(minutes=PUSH_FAIL_MIN + 5)]
        span = (now - min(run)).total_seconds() / 60
        if span >= PUSH_FAIL_MIN and not _in_sword_reboot_window(now, min(run)):
            conditions['push-failing'] = (
                f'aual push to SWORD failing for {span:.0f} min '
                f'(threshold {PUSH_FAIL_MIN:.0f})')

    # 7 — herot views publish (runs inside the MIDL relay).
    if AUAL_PUB_RETRY.exists():
        pubs = _entries(MIDL_LOGS, 'aual_pub')
        pub_oks = [ts for ts, ok, _ in pubs if ok]
        last_pub = max(pub_oks) if pub_oks else None
        if last_pub is None or now - last_pub > timedelta(
                minutes=HEROT_STALE_MIN):
            since = ('never' if last_pub is None
                     else f'{(now - last_pub).total_seconds() / 60:.0f} min ago')
            conditions['herot-aual-failing'] = (
                f'herot publish of AU/AL website views failing: retry marker '
                f'set, last success {since} (threshold {HEROT_STALE_MIN:.0f})')

    return conditions


def _write_detail(conditions, now):
    try:
        parts = ['LAUREN AU/AL watchdog (solsticedisk)', '']
        parts += [f'{name}: {detail}' for name, detail in conditions.items()]
        parts += ['', 'tick log tail:']
        parts += [line for _, _, line in _entries(LOGS, 'tick', limit=8)][-8:]
        tail = [line for _, _, line in _entries(LOGS, 'push', limit=5)][-5:]
        if tail:
            parts += ['', 'push log tail:'] + tail
        DETAIL_FILE.parent.mkdir(parents=True, exist_ok=True)
        DETAIL_FILE.write_text('\n'.join(parts) + '\n')
        return DETAIL_FILE
    except OSError:
        return None


def _send_email(status, label, log_file=None):
    cmd = [PY, str(NOTIFY),
           '--label', f'LAUREN: {label[:110]}',
           '--from', MAIL_FROM,
           '--status', status,
           '--summary-kind', 'none',
           '--started', str(int(time.time()))]
    if status != 'ok' and log_file:
        cmd += ['--log-file', str(log_file)]
    if os.environ.get('AUAL_WD_DRY_RUN'):
        cmd += ['--dry-run']
    try:
        subprocess.run(cmd, capture_output=True, timeout=60)
    except Exception:
        pass


def _read_json(path):
    try:
        return json.loads(Path(path).read_text())
    except (OSError, ValueError):
        return None


def _uptime_min():
    try:
        return float(Path('/proc/uptime').read_text().split()[0]) / 60
    except (OSError, ValueError, IndexError):
        return BOOT_GRACE_MIN + 1


def check(now=None, notify=None):
    now = now or datetime.utcnow()
    notify = notify or _send_email
    if _uptime_min() < BOOT_GRACE_MIN:
        return 'suppressed-local-boot'

    conditions = evaluate(now)
    state = _read_json(STATE_FILE) or {}
    conds = state.get('conditions', {})
    outcomes = []

    detail_file = _write_detail(conditions, now) if conditions else None
    for name, detail in conditions.items():
        cs = conds.get(name, {})
        last = cs.get('last_alert')
        throttled = False
        if last:
            try:
                hours = (now - datetime.strptime(last, TS_FMT)
                         ).total_seconds() / 3600
                throttled = hours < RATE_LIMIT_H
            except ValueError:
                pass
        if throttled:
            outcomes.append(f'{name}:rate-limited')
        else:
            notify('fail', detail, detail_file)
            cs['last_alert'] = now.strftime(TS_FMT)
            outcomes.append(f'{name}:alerted')
        cs['alerting'] = True
        conds[name] = cs

    for name in sorted(conds):
        if conds[name].get('alerting') and name not in conditions:
            notify('ok', f'recovered: {name}')
            conds[name]['alerting'] = False
            outcomes.append(f'{name}:recovered')

    STATE.mkdir(parents=True, exist_ok=True)
    tmp = STATE_FILE.with_suffix('.json.tmp')
    tmp.write_text(json.dumps({'conditions': conds}))
    tmp.rename(STATE_FILE)
    return ' '.join(outcomes) if outcomes else 'ok'


def main():
    outcome = check()
    LOGS.mkdir(parents=True, exist_ok=True)
    with open(WD_LOG, 'a') as fh:
        fh.write(f'{datetime.utcnow().strftime(TS_FMT)} {outcome}\n')
    return 0


if __name__ == '__main__':
    import sys
    sys.exit(main())
