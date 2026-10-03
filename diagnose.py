"""Diagnose why notes get misclassified (e.g. fake 10/20/200 read as 500).

1. Per-class instance & image balance in final_dataset
2. Confidence behavior: what a note looks like at various conf thresholds
"""
import collections
import glob
import os

NAMES = ['10_note', '20_note', '50_note', '100_note', '200_note', '500_note',
         '1_coin', '2_coin', '5_coin', '10_coin', '20_coin']


def balance():
    for split in ('train', 'val'):
        inst, imgs = collections.Counter(), collections.Counter()
        for f in glob.glob(f'final_dataset/labels/{split}/*.txt'):
            seen = set()
            for line in open(f).read().splitlines():
                if line.strip():
                    c = int(line.split()[0])
                    inst[c] += 1
                    seen.add(c)
            for c in seen:
                imgs[c] += 1
        print(f'--- {split} ---')
        print(f'{"class":>10} {"instances":>10} {"images":>8}')
        for i, n in enumerate(NAMES):
            print(f'{n:>10} {inst[i]:>10} {imgs[i]:>8}')
        print(f'{"TOTAL":>10} {sum(inst.values()):>10}\n')


if __name__ == '__main__':
    balance()
