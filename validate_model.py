"""Final per-class validation for the trained currency model.

Usage: python validate_model.py [weights.pt]
Default weights: runs/detect/runs/detect/currency_detection/weights/best.pt
"""
import sys

from ultralytics import YOLO

WEIGHTS = sys.argv[1] if len(sys.argv) > 1 else 'runs/detect/runs/detect/currency_detection/weights/best.pt'
DATA = 'final_dataset/dataset.yaml'

def main():
    model = YOLO(WEIGHTS)
    print(f'Validating {WEIGHTS} on {model.names}')
    result = model.val(data=DATA, split='val', imgsz=640, plots=True,
                       project='runs/detect', name='final_val')

    box = result.box
    names = model.names
    print()
    print('=== PER-CLASS RESULTS ================================')
    print(f'{"class":>18}  {"P":>6}  {"R":>6}  {"mAP50":>7}  {"mAP50-95":>8}')
    for i, c in enumerate(box.ap_class_index):
        p, r, ap50, ap = box.class_result(i)
        print(f'{names[c]:>18}  {p:6.3f}  {r:6.3f}  {ap50:7.3f}  {ap:8.3f}')
    print('------------------------------------------------------------')
    print(f'{"OVERALL":>18}  {box.mp:6.3f}  {box.mr:6.3f}  {box.map50:7.4f}  {box.map:8.4f}')
    print(f'\nPlots saved to runs/detect/final_val/')

if __name__ == '__main__':
    main()
