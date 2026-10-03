import urllib.request, json

try:
    r = urllib.request.urlopen('http://127.0.0.1:5000/debug_raw', timeout=30)
    data = json.loads(r.read().decode())
except Exception as e:
    print('ERROR:', e)
    raise SystemExit

if isinstance(data, dict):
    print('SERVER:', data)
    raise SystemExit

for fr in data:
    if fr.get('read_failed'):
        print('frame', fr['frame'], 'READ FAILED')
        continue
    print('frame', fr['frame'], 'shape', fr.get('shape'))
    raw = fr.get('raw', [])
    if not raw:
        print('   (no raw detections at conf>=0.15)')
    for d in sorted(raw, key=lambda x: -x['conf'])[:8]:
        print('   {:<18} conf={:.3f}  box={}'.format(d['name'], d['conf'], d['box']))
