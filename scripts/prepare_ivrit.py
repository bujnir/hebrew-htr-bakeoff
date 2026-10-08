"""Download the gated ivrit.ai benchmark (lines config) and write data/ivrit/manifest.json + images.

    huggingface-cli login     # after accepting the dataset terms on the Hub
    python scripts/prepare_ivrit.py

The benchmark is test-only. data/ivrit/ is git-ignored: do not commit its images or transcriptions.
"""
import io
import json
import os
import sys

from datasets import load_dataset
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from common import DATA

OUT = os.path.join(DATA, 'ivrit')
os.makedirs(os.path.join(OUT, 'img'), exist_ok=True)


def tier(text):
    n = len(text.split())
    return 'word' if n <= 1 else 'field' if n <= 3 else 'line'


ds = load_dataset('ivrit-ai/hebrew-handwriting-ocr-benchmark', 'lines', split='test')
rows = []
for r in ds:
    im = r['image'] if isinstance(r['image'], Image.Image) else Image.open(io.BytesIO(r['image']['bytes']))
    im.convert('RGB').save(os.path.join(OUT, 'img', f"{r['line_id']}.png"))
    rows.append({'id': r['line_id'], 'tier': tier(r['text']), 'path': f"img/{r['line_id']}.png", 'gt': r['text'],
                 'legibility': r['legibility_score'], 'page': r['submission_id'], 'source': r['source']})
json.dump(rows, open(os.path.join(OUT, 'manifest.json'), 'w', encoding='utf-8'), ensure_ascii=False, indent=0)
print(len(rows), 'lines ->', OUT)
