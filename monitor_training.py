"""
Live Training Monitor
Displays real-time progress of the background YOLO training run.

Usage:  python monitor_training.py            (live, refreshes every 3s)
        python monitor_training.py --once     (single snapshot)
Exit:   Ctrl+C (does NOT stop training)
"""

import os
import re
import sys
import time
from datetime import datetime

import glob
LOG_CANDIDATES = ['train_log.txt'] + sorted(glob.glob('train_log_resume*.txt'))

def active_log():
    """Most recently written training log (handles resume sessions)"""
    logs = [(os.path.getmtime(p), p) for p in LOG_CANDIDATES if os.path.exists(p)]
    return max(logs)[1] if logs else LOG_CANDIDATES[0]

RESULTS_CSV = os.path.join('runs', 'detect', 'runs', 'detect', 'currency_detection', 'results.csv')
TOTAL_EPOCHS = 100

# Latest progress line for the current epoch, e.g.
# "      3/100         0G      1.42      2.10      1.61         40        640: 45% ... 78/174 3.9s/it 5:12<6:22"
TRAIN_RE = re.compile(
    r'^\s*(\d+)/(\d+)\s+\S+\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)\s+(\d+)\s+(\d+):\s+(\d+)%.*?(\d+)/(\d+)\s+([\d.]+)s/it\s+([\d:.]+)<([\d:.]+)'
)
VAL_RE = re.compile(r'^\s*val:\s+.*?(\d+)%.*?(\d+)/(\d+)')


def read_last_lines(path, n=40):
    """Read the final n lines of a file efficiently"""
    if not os.path.exists(path):
        return []
    with open(path, 'r', encoding='utf-8', errors='replace') as f:
        return f.readlines()[-n:]


def parse_progress(log_file=None):
    """Extract current epoch/batch/losses from the tail of the log"""
    log_file = log_file or active_log()
    state = {'epoch': None, 'total': TOTAL_EPOCHS, 'batch': None, 'nbatches': None,
             'pct': None, 'box': None, 'cls': None, 'dfl': None, 'eta': None, 'phase': 'unknown'}
    for line in read_last_lines(log_file, 60):
        m = TRAIN_RE.search(line.strip())
        if m:
            state.update(
                phase='TRAINING', epoch=int(m.group(1)), total=int(m.group(2)),
                batch=int(m.group(9)), nbatches=int(m.group(10)), pct=int(m.group(8)),
                box=float(m.group(3)), cls=float(m.group(4)), dfl=float(m.group(5)),
                eta=m.group(12),
            )
            continue
        m = VAL_RE.search(line.strip())
        if m:
            state.update(phase='VALIDATING', batch=int(m.group(2)), nbatches=int(m.group(3)), pct=int(m.group(1)))
    return state


def bar(pct, width=40):
    """Render an ASCII progress bar"""
    if pct is None:
        return '[' + '?' * width + ']'
    filled = int(width * pct / 100)
    return '[' + '#' * filled + '-' * (width - filled) + f'] {pct:3d}%'


def load_results():
    """Parse results.csv into a list of per-epoch metric dicts"""
    if not os.path.exists(RESULTS_CSV):
        return []
    with open(RESULTS_CSV) as f:
        header = f.readline().strip().split(',')
        rows = []
        for line in f:
            vals = line.strip().split(',')
            if len(vals) == len(header):
                rows.append(dict(zip(header, vals)))
    return rows


def fmt(row, key, digits=4):
    try:
        return f"{float(row[key]):.{digits}f}"
    except (KeyError, ValueError):
        return '   n/a'


def fitness_of(row):
    """Ultralytics fitness = 0.1*mAP50 + 0.9*mAP50-95 (not written to results.csv)"""
    try:
        return 0.1 * float(row['metrics/mAP50(B)']) + 0.9 * float(row['metrics/mAP50-95(B)'])
    except (KeyError, ValueError, TypeError):
        return None


def render():
    os.system('cls' if os.name == 'nt' else 'clear')
    now = datetime.now().strftime('%H:%M:%S')
    log_file = active_log()
    p = parse_progress(log_file)

    print('=' * 64)
    print('   CURRENCY DETECTION - LIVE TRAINING MONITOR')
    print('=' * 64)
    print(f'   Time: {now}    Status: {p["phase"]}    Log: {log_file}')
    print('-' * 64)

    # Stale-log detection
    if os.path.exists(log_file):
        age = time.time() - os.path.getmtime(log_file)
        flag = 'HEALTHY' if age < 120 else ('STALLED? ' + '(no log write in %ds)' % age)
        print(f'   Log last updated: {age:4.0f}s ago   -> {flag}')
    else:
        print('   No training log found (train_log.txt / train_log_resume.txt)')
    print('-' * 64)

    ep = p['epoch'] or (len(load_results()) + 1)
    print(f'   EPOCH {ep} / {p["total"]}')
    if p['batch']:
        print(f'   {p["phase"]:12s} {bar(p["pct"])}  batch {p["batch"]}/{p["nbatches"]}')
    else:
        print(f'   {bar(None)}  waiting for progress output...')
    print()

    if p['box'] is not None:
        print(f'   box_loss {p["box"]:.3f}   cls_loss {p["cls"]:.3f}   dfl_loss {p["dfl"]:.3f}', end='')
        print(f'   epoch ETA {p["eta"]}' if p['eta'] else '')
    print('-' * 64)

    rows = load_results()
    if not rows:
        print('   Metrics appear after the first epoch completes validation.')
        print('=' * 64)
        return

    print(f'   COMPLETED EPOCHS: {len(rows)}   (mAP50 is now a HONEST number)')
    print()
    print(f'   {"ep":>4} {"P":>7} {"R":>7} {"mAP50":>7} {"mAP50-95":>9} {"fitness":>8} {"min/ep":>7}')
    prev_t = None
    for r in rows[-10:]:
        try:
            t = float(r.get('time', 0))
            mins = (t - prev_t) / 60 if prev_t is not None else t / 60
            prev_t = t
            pace = f'{mins:7.1f}'
        except (ValueError, TypeError):
            pace = f'{"n/a":>7}'
        f = fitness_of(r)
        print(f'   {r.get("epoch", "?"):>4} '
              f'{fmt(r, "metrics/precision(B)", 3)} {fmt(r, "metrics/recall(B)", 3)} '
              f'{fmt(r, "metrics/mAP50(B)"):>7} {fmt(r, "metrics/mAP50-95(B)"):>9} '
              f'{f if f is None else format(f, ".4f"):>8} {pace}')

    scored = [(fitness_of(r), r) for r in rows if fitness_of(r) is not None]
    print()
    if scored:
        best_f, best = max(scored, key=lambda x: x[0])
        print(f'   BEST fitness so far: epoch {best.get("epoch")}  fitness={best_f:.4f}  mAP50={fmt(best, "metrics/mAP50(B)")}')
    try:
        last = rows[-1]
        lr = float(last.get('lr/pg0', 0))
        print(f'   Learning rate: {lr:.5f}   (decays toward lr0*0.1=0.001)')
    except ValueError:
        pass

    # Rough total-time estimate from the cumulative time column
    try:
        el = float(rows[-1].get('time', 0))
        n = len(rows)
        if el > 0 and n:
            print(f'   Est. full run: ~{el / n * p["total"] / 3600:.1f}h  ({el / n / 60:.1f} min/epoch, {n} done)')
    except (ValueError, ZeroDivisionError):
        pass
    print('=' * 64)


if __name__ == '__main__':
    if '--once' in sys.argv:
        render()
        raise SystemExit(0)

    print('Starting monitor... (press Ctrl+C to close - training keeps running)')
    time.sleep(2)
    try:
        while True:
            render()
            time.sleep(3)
    except KeyboardInterrupt:
        print('\nMonitor closed. Training is STILL RUNNING in the background.')
