"""Simulate the app's fixed conf=0.3 on the val set and report per-class
outcome: correctly detected / missed (below 0.3 or no box) / misclassified.

This answers "why do coins fail at conf=0.3 even though peak recall is ~0.99".
"""
import glob
import os
import cv2
from ultralytics import YOLO

CONF = 0.3          # app's threshold
IOU_MATCH = 0.5     # a detection counts on the right object at IoU>=0.5
NAMES = None


def to_xyxy(box, w, h):
    xc, yc, bw, bh = box
    x1, y1 = (xc - bw / 2) * w, (yc - bh / 2) * h
    x2, y2 = (xc + bw / 2) * w, (yc + bh / 2) * h
    return [x1, y1, x2, y2]


def iou(a, b):
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0, ix2 - ix1), max(0, iy2 - iy1)
    inter = iw * ih
    area_a = (ax2 - ax1) * (ay2 - ay1)
    area_b = (bx2 - bx1) * (by2 - by1)
    return inter / (area_a + area_b - inter + 1e-9)


def main():
    global NAMES
    model = YOLO('runs/detect/runs/detect/currency_detection/weights/best.pt')
    NAMES = model.names
    stats = {i: {'gt': 0, 'ok': 0, 'miss': 0, 'wrong': 0, 'as': {}} for i in range(len(NAMES))}

    only = os.environ.get('ONLY_CLASSES')  # e.g. "6,7,8,9,10" to filter coin images
    subset = set(int(x) for x in only.split(',')) if only else None

    files = sorted(glob.glob('final_dataset/images/val/*.jpg'))
    for img_path in files:
        lbl = img_path.replace('images', 'labels').rsplit('.', 1)[0] + '.txt'
        if not os.path.exists(lbl):
            continue
        img = cv2.imread(img_path)
        h, w = img.shape[:2]
        gts = []
        for line in open(lbl).read().splitlines():
            if line.strip():
                p = [float(x) for x in line.split()[1:]]
                cls = int(line.split()[0])
                gts.append((cls, to_xyxy(p, w, h)))
        if subset and not any(c in subset for c, _ in gts):
            continue

        r = model.predict(img_path, conf=CONF, verbose=False)[0]
        preds = []
        for b in r.boxes:
            preds.append((int(b.cls[0]), [float(v) for v in b.xyxy[0]], float(b.conf[0])))

        used = set()
        for cls, gb in gts:
            if subset and cls not in subset:
                continue
            stats[cls]['gt'] += 1
            best, bi = 0, -1
            for i, (pc, pb, pc_conf) in enumerate(preds):
                if i in used:
                    continue
                v = iou(gb, pb)
                if v > best:
                    best, bi = v, i
            if bi >= 0 and best >= IOU_MATCH:
                used.add(bi)
                if preds[bi][0] == cls:
                    stats[cls]['ok'] += 1
                else:
                    stats[cls]['wrong'] += 1
                    as_c = preds[bi][0]
                    stats[cls]['as'][as_c] = stats[cls]['as'].get(as_c, 0) + 1
            else:
                stats[cls]['miss'] += 1

    print(f'=== App simulation at conf={CONF} (val set) ===')
    print(f'{"class":>18} {"gt":>5} {"ok%":>6} {"miss%":>6} {"wrong%":>7}   main confusion')
    for c in sorted(stats):
        s = stats[c]
        if s['gt'] == 0:
            continue
        okp = 100 * s['ok'] / s['gt']
        missp = 100 * s['miss'] / s['gt']
        wrp = 100 * s['wrong'] / s['gt']
        top = max(s['as'].items(), key=lambda x: x[1])[0] if s['as'] else None
        conf_txt = f'-> {NAMES[top]}' if top is not None else ''
        print(f'{NAMES[c]:>18} {s["gt"]:>5} {okp:>5.0f}% {missp:>5.0f}% {wrp:>6.0f}%   {conf_txt}')


if __name__ == '__main__':
    main()
