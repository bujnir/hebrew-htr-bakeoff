"""Original HebHTR (2021; TensorFlow 1.12, Python 3.6, shipped weights, word-beam-search decoder).

    git clone https://github.com/Lotemn102/HebHTR third_party/HebHTR      # needs its py3.6 / TF 1.12 env
    LANG=C.UTF-8 python runners/hebhtr.py word_beam data/ivrit/manifest.json results/predictions/ivrit/local__hebhtr.jsonl

HebHTR reads single words. Word crops go straight in; multi-word crops are split into words
(dilate + contours, right to left), as HebHTR's own imgToWords does.
"""
import json
import os
import sys
import time

import cv2
import numpy as np

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, HERE)
from common import load_manifest

dec, manifest, out = sys.argv[1], os.path.abspath(sys.argv[2]), os.path.abspath(sys.argv[3])
rows = load_manifest(manifest)
REPO = os.environ.get('HEBHTR_DIR', os.path.join(HERE, 'third_party', 'HebHTR'))
os.chdir(REPO)
sys.path.insert(0, REPO)
from Model import DecoderType, Model
from predictWord import Batch
from processFunctions import preprocessImageForPrediction

model = Model(open('model/charList.txt').read(),
              DecoderType.WordBeamSearch if dec == 'word_beam' else DecoderType.BestPath, mustRestore=True)


def binarize(p):
    g = cv2.imread(p, cv2.IMREAD_GRAYSCALE)
    _, b = cv2.threshold(g, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return b


def crop_ink(b, m=2):
    ys, xs = np.where(b < 128)
    if not len(xs):
        return b
    return b[max(ys.min() - m, 0):ys.max() + m + 1, max(xs.min() - m, 0):xs.max() + m + 1]


def split_words(b):
    ink = (b < 128).astype(np.uint8) * 255
    h = b.shape[0]
    sm = cv2.dilate(ink, np.ones((3, max(3, int(h * 0.35))), np.uint8), iterations=1)  # smear ~1/3 line height
    cs, _ = cv2.findContours(sm, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[-2:]
    boxes = [cv2.boundingRect(c) for c in cs]
    boxes = [bx for bx in boxes if bx[2] * bx[3] > 0.02 * h * h]
    boxes.sort(key=lambda bx: -bx[0])               # Hebrew: right to left
    return [crop_ink(b[:, x:x + w]) for x, y, w, hh in boxes]


def rec(img):
    t, p = model.inferBatch(Batch(None, [preprocessImageForPrediction(img, Model.imgSize)]), True)
    return t[0], float(p[0])


with open(out, 'w', encoding='utf-8') as f:
    for r in rows:
        t0 = time.time()
        b = binarize(r['path'])
        words = [crop_ink(b)] if r.get('tier') == 'word' else (split_words(b) or [crop_ink(b)])
        res = [rec(w) for w in words]
        f.write(json.dumps({'id': r['id'], 'pred': ' '.join(t for t, _ in res), 'conf': float(np.prod([p for _, p in res])),
                            'n_segments': len(words), 'sec': round(time.time() - t0, 2)}, ensure_ascii=False) + '\n')
print('done', len(rows))
