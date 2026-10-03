"""
Final Dataset Preparation Script
Builds final_dataset/ from combined_dataset/ with:
- Only the 11 supported denominations (drops Rs 1/2/5 notes and Rs 2000 note)
- Class IDs remapped to 0-10
- Duplicate images removed (md5-based)
- Clean 80/20 train/val split with zero overlap (fixes data leakage)
"""

import os
import glob
import random
import hashlib
import shutil
import yaml

# Old class ID (combined_dataset) -> New class ID (final_dataset)
# Dropped: 0 (1_rupee_note), 1 (2_rupee_note), 2 (5_rupee_note), 9 (2000_rupee_note)
CLASS_REMAP = {
    3: 0,    # 10_rupee_note
    4: 1,    # 20_rupee_note
    5: 2,    # 50_rupee_note
    6: 3,    # 100_rupee_note
    7: 4,    # 200_rupee_note
    8: 5,    # 500_rupee_note
    10: 6,   # 1_rupee_coin
    11: 7,   # 2_rupee_coin
    12: 8,   # 5_rupee_coin
    13: 9,   # 10_rupee_coin
    14: 10,  # 20_rupee_coin
}

CLASS_NAMES = [
    '10_rupee_note', '20_rupee_note', '50_rupee_note', '100_rupee_note',
    '200_rupee_note', '500_rupee_note',
    '1_rupee_coin', '2_rupee_coin', '5_rupee_coin', '10_rupee_coin', '20_rupee_coin'
]

SRC_DIR = 'combined_dataset'
DST_DIR = 'final_dataset'
VAL_RATIO = 0.2
SEED = 42


def find_image(images_dir, stem):
    """Locate image file matching a label stem"""
    for ext in ('.jpg', '.jpeg', '.png', '.JPG', '.JPEG', '.PNG'):
        path = os.path.join(images_dir, stem + ext)
        if os.path.exists(path):
            return path
    return None


def md5(path):
    """Compute md5 hash of a file"""
    with open(path, 'rb') as f:
        return hashlib.md5(f.read()).hexdigest()


def remap_label_file(src_label, dst_label):
    """Remap class IDs in a YOLO label file. Returns number of kept boxes."""
    kept = 0
    lines_out = []
    with open(src_label, 'r') as f:
        for line in f:
            parts = line.split()
            if not parts:
                continue
            old_cls = int(float(parts[0]))
            if old_cls in CLASS_REMAP:
                lines_out.append(' '.join([str(CLASS_REMAP[old_cls])] + parts[1:]))
                kept += 1
    if kept > 0:
        with open(dst_label, 'w') as f:
            f.write('\n'.join(lines_out) + '\n')
    return kept


def main():
    random.seed(SEED)

    # Fresh start
    if os.path.exists(DST_DIR):
        print(f"Removing existing {DST_DIR}/ for rebuild...")
        shutil.rmtree(DST_DIR)
    for sub in ('images/train', 'images/val', 'labels/train', 'labels/val'):
        os.makedirs(os.path.join(DST_DIR, sub), exist_ok=True)

    # Collect (image_path, label_path) pairs from both train and val of source
    samples = []
    skipped_no_img, skipped_no_box = 0, 0
    for split in ('train', 'val'):
        images_dir = os.path.join(SRC_DIR, 'images', split)
        labels_dir = os.path.join(SRC_DIR, 'labels', split)
        for lbl in sorted(glob.glob(os.path.join(labels_dir, '*.txt'))):
            stem = os.path.splitext(os.path.basename(lbl))[0]
            img = find_image(images_dir, stem)
            if img is None:
                skipped_no_img += 1
                continue
            samples.append((img, lbl))

    print(f"Collected {len(samples)} image/label pairs (skipped {skipped_no_img} labels without image)")

    # Deduplicate by image content hash (fixes leakage: same image in train & val)
    seen = {}
    unique_samples = []
    for img, lbl in samples:
        h = md5(img)
        if h not in seen:
            seen[h] = True
            unique_samples.append((img, lbl))
    print(f"After md5 deduplication: {unique_samples and len(unique_samples)} unique images")

    # Shuffle and split 80/20
    random.shuffle(unique_samples)
    n_val = int(len(unique_samples) * VAL_RATIO)
    val_samples = unique_samples[:n_val]
    train_samples = unique_samples[n_val:]

    # Write out with remapped labels
    stats = {'train': 0, 'val': 0}
    boxes = {'train': 0, 'val': 0}
    for split, items in (('train', train_samples), ('val', val_samples)):
        img_dir = os.path.join(DST_DIR, 'images', split)
        lbl_dir = os.path.join(DST_DIR, 'labels', split)
        for i, (img, lbl) in enumerate(items):
            stem = os.path.splitext(os.path.basename(img))[0]
            # Prefix index to avoid filename collisions across source splits
            dst_base = f"{i:05d}_{stem}"
            dst_img = os.path.join(img_dir, dst_base + '.jpg')
            dst_lbl = os.path.join(lbl_dir, dst_base + '.txt')
            n_kept = remap_label_file(lbl, dst_lbl)
            if n_kept == 0:
                # Image only contained dropped classes - remove label, skip image
                if os.path.exists(dst_lbl):
                    os.remove(dst_lbl)
                skipped_no_box += 1
                continue
            shutil.copy2(img, dst_img)
            stats[split] += 1
            boxes[split] += n_kept

    # Dataset config
    data_config = {
        'path': os.path.abspath(DST_DIR).replace('\\', '/'),
        'train': 'images/train',
        'val': 'images/val',
        'nc': len(CLASS_NAMES),
        'names': CLASS_NAMES,
    }
    with open(os.path.join(DST_DIR, 'dataset.yaml'), 'w') as f:
        yaml.dump(data_config, f, default_flow_style=False, sort_keys=False)

    # Per-class counts for final dataset
    from collections import Counter
    cnt = Counter()
    for split in ('train', 'val'):
        for lbl in glob.glob(os.path.join(DST_DIR, 'labels', split, '*.txt')):
            for line in open(lbl):
                p = line.split()
                if p:
                    cnt[int(p[0])] += 1

    print("\n" + "=" * 55)
    print("final_dataset built successfully")
    print("=" * 55)
    print(f"Train images : {stats['train']} ({boxes['train']} boxes)")
    print(f"Val images   : {stats['val']} ({boxes['val']} boxes)")
    print(f"Skipped (only dropped-class content): {skipped_no_box}")
    print("\nPer-class instance counts:")
    for cid in sorted(cnt):
        print(f"  {cid:2d} {CLASS_NAMES[cid]:<18} {cnt[cid]}")
    print(f"\nConfig written: {DST_DIR}/dataset.yaml")


if __name__ == '__main__':
    main()
