"""Mishkefet-v1 (CPU) with the heb-ocr repo's default line-mode settings: char 6-gram LM (weight 0.4), beam 12, TTA x3.

    git clone https://github.com/itayinbarr/heb-ocr third_party/heb-ocr   # then fetch weights per its README
    python runners/mishkefet.py data/ivrit/manifest.json results/predictions/ivrit/local__mishkefet-v1.jsonl
    CLEAN=1 python runners/mishkefet.py data/id100/manifest.json results/predictions/id100/local__mishkefet-v1.jsonl

CLEAN=1 removes printed form lines first (used for the ID crops). Weights are CC-BY-NC-SA.
"""
import json
import os
import sys
import time

import numpy as np
from PIL import Image

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, HERE)
D = os.environ.get('HEBOCR_DIR', os.path.join(HERE, 'third_party', 'heb-ocr'))
sys.path.insert(0, D)
from hebocr.lm import CharNGramLM
from hebocr.recognize import Recognizer

from common import load_manifest

manifest, out = sys.argv[1], sys.argv[2]
rec = Recognizer(os.path.join(D, 'mishkefet-v1.pt'), device='cpu',
                 lm=CharNGramLM.load(os.path.join(D, 'hebrew_char6.pkl')), lm_weight=0.4, beam_width=12, tta=3)
clean = os.environ.get('CLEAN') == '1'
if clean:
    from digits.readers import remove_form_lines
rows = load_manifest(manifest)
with open(out, 'w', encoding='utf-8') as f:
    for r in rows:
        t = time.time()
        im = Image.open(r['path']).convert('RGB')
        if clean:
            im = Image.fromarray(remove_form_lines(np.asarray(im)))
        f.write(json.dumps({'id': r['id'], 'pred': rec.read_tta([im])[0], 'sec': round(time.time() - t, 2)},
                           ensure_ascii=False) + '\n')
print('done', len(rows))
