"""Poll the running app's /api/detections and log every reported detection
with confidence and box geometry. Used to see what the model fires on when
NO currency is shown (body-movement false positives).

Keep the webcam page open and do NOT show any currency while this runs.
"""
import time
import urllib.request
import json

URL = 'http://127.0.0.1:5000/api/detections'
SECONDS = 30

print(f'Polling {URL} for {SECONDS}s (show NO currency)...\n')
samples = {}
t_end = time.time() + SECONDS
errors = 0
while time.time() < t_end:
    try:
        with urllib.request.urlopen(URL, timeout=3) as r:
            data = json.loads(r.read().decode())
        for d in data.get('detections', []):
            key = d['currency']
            x1, y1, x2, y2 = d['bbox']
            w, h = x2 - x1, y2 - y1
            samples.setdefault(key, []).append(
                (d['confidence'], w, h))
    except Exception as e:
        errors += 1
    time.sleep(0.2)

if not samples:
    print('No detections reported during the window. (Is the webcam page open?)')
else:
    print('False-positive classes seen with NO currency shown:')
    print(f"{'class':<20}{'count':>6}{'min conf':>10}{'avg conf':>10}{'max conf':>10}  box sizes")
    for key, vals in sorted(samples.items(), key=lambda kv: -len(kv[1])):
        confs = [v[0] for v in vals]
        ws = [v[1] for v in vals]
        hs = [v[2] for v in vals]
        print(f"{key:<20}{len(vals):>6}{min(confs):>10.2f}{sum(confs)/len(confs):>10.2f}"
              f"{max(confs):>10.2f}  ~{int(sum(ws)/len(ws))}x{int(sum(hs)/len(hs))}")
if errors:
    print(f'\n({errors} poll errors)')
