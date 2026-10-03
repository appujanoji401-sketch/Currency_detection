from ultralytics import YOLO
import glob, os, cv2

m = YOLO('runs/detect/runs/detect/currency_detection/weights/best.pt')
NAMES = m.names

img = cv2.imread('debug_frame_4.jpg')
print('=== live coin frame, conf=0.01 ===')
for sz in (1024, 640):
    r = m(img, verbose=False, conf=0.01, imgsz=sz)[0]
    dets = sorted(
        [(float(b.conf[0]), int(b.cls[0])) for b in r.boxes], reverse=True)
    print(f'imgsz={sz}: top detections:')
    for conf, cls in dets[:6]:
        print(f'   {NAMES.get(cls, cls):<20} conf={conf:.3f}')
    if not dets:
        print('   (none)')

# Inspect training images of the 10-rupee coin to compare appearance
print('\n=== training coin images (class 9 = 10 Rupee Coin) ===')
# find label files whose first token is class 9
coin_imgs = []
for lbl in glob.glob('final_dataset/labels/train/*.txt'):
    try:
        txt = open(lbl).read().split()
    except Exception:
        continue
    if txt and txt[0] == '9':
        img_path = lbl.replace('labels', 'images').replace('.txt', '.jpg')
        if os.path.exists(img_path):
            coin_imgs.append(img_path)
print('num training images containing class 9:', len(coin_imgs))
for p in coin_imgs[:5]:
    im = cv2.imread(p)
    h, w = im.shape[:2]
    # bbox of the class-9 box
    lbl = p.replace('images', 'labels').replace('.jpg', '.txt')
    for line in open(lbl):
        parts = line.split()
        if parts[0] == '9':
            bw, bh = float(parts[3]) * w, float(parts[4]) * h
            print(f'   {os.path.basename(p)}  img={w}x{h}  coin_box~{int(bw)}x{int(bh)}')
            break
