"""Measure average object size per class (in 640px terms) in the dataset.
YOLOv8n struggles when an object is < ~24-32 px on a side. If coins are tiny
in training images, they'll vanish in a live webcam frame too.
"""
import collections
import glob
import statistics

NAMES = ['10_note', '20_note', '50_note', '100_note', '200_note', '500_note',
         '1_coin', '2_coin', '5_coin', '10_coin', '20_coin']

# collect normalized width/height per class across train+val
wh = collections.defaultdict(list)
for split in ('train', 'val'):
    for f in glob.glob(f'final_dataset/labels/{split}/*.txt'):
        for line in open(f).read().splitlines():
            if line.strip():
                p = line.split()
                c = int(p[0]); bw = float(p[3]); bh = float(p[4])
                wh[c].append((bw, bh))

print(f'{"class":>10} {"n":>5} {"medW@640":>9} {"medH@640":>9} {"med_area%":>10}')
for i, name in enumerate(NAMES):
    boxes = wh[i]
    if not boxes:
        continue
    # side length in px if image were 640 wide (approx; square-ish assumption)
    ws = [w * 640 for w, h in boxes]
    hs = [h * 640 for w, h in boxes]
    areas = [w * h * 100 for w, h in boxes]  # % of a normalized 1x1 frame
    print(f'{name:>10} {len(boxes):>5} {statistics.median(ws):>9.0f} '
          f'{statistics.median(hs):>9.0f} {statistics.median(areas):>10.1f}')
