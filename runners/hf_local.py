"""Small Hugging Face models on CPU: TrOCR-Hebrew and GLM-OCR.

    python runners/hf_local.py trocr  data/ivrit/manifest.json results/predictions/ivrit/local__trocr-hebrew.jsonl
    python runners/hf_local.py glmocr data/ivrit/manifest.json results/predictions/ivrit/local__glm-ocr.jsonl
"""
import json
import os
import sys
import time

import torch
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from common import done_ids, load_manifest

torch.set_num_threads(int(os.environ.get('THREADS', 2)))
which, manifest, out = sys.argv[1], sys.argv[2], sys.argv[3]
rows = load_manifest(manifest)
done = done_ids(out)

if which == 'trocr':
    from transformers import AutoTokenizer, TrOCRProcessor, ViTImageProcessor, VisionEncoderDecoderModel
    mid = 'cyttic/exp15-trocr-hebrew-synth1m'
    model = VisionEncoderDecoderModel.from_pretrained(mid).eval()
    try:
        proc = TrOCRProcessor.from_pretrained(mid)
    except Exception:  # repo ships no image processor: build one matching the encoder config
        size = model.config.encoder.image_size
        ip = ViTImageProcessor(size={'height': size, 'width': size}, image_mean=[0.5] * 3, image_std=[0.5] * 3)
        proc = TrOCRProcessor(image_processor=ip, tokenizer=AutoTokenizer.from_pretrained(mid))

    def read(path):
        px = proc(images=Image.open(path).convert('RGB'), return_tensors='pt').pixel_values
        return proc.batch_decode(model.generate(px, max_new_tokens=128, num_beams=4), skip_special_tokens=True)[0].strip()

elif which == 'glmocr':
    from transformers import AutoModelForImageTextToText, AutoProcessor
    mid = 'zai-org/GLM-OCR'
    proc = AutoProcessor.from_pretrained(mid)
    model = AutoModelForImageTextToText.from_pretrained(mid, torch_dtype=torch.float32).eval()

    def read(path):
        msgs = [{'role': 'user', 'content': [{'type': 'image', 'url': path}, {'type': 'text', 'text': 'Text Recognition:'}]}]
        inp = proc.apply_chat_template(msgs, tokenize=True, add_generation_prompt=True, return_dict=True, return_tensors='pt')
        inp.pop('token_type_ids', None)
        ids = model.generate(**inp, max_new_tokens=128, do_sample=False)
        return proc.decode(ids[0][inp['input_ids'].shape[1]:], skip_special_tokens=True).strip()
else:
    raise SystemExit('which = trocr | glmocr')

with open(out, 'a', encoding='utf-8') as f, torch.no_grad():
    for r in rows:
        if r['id'] in done:
            continue
        t = time.time()
        f.write(json.dumps({'id': r['id'], 'pred': read(r['path']), 'sec': round(time.time() - t, 2)}, ensure_ascii=False) + '\n')
        f.flush()
print(which, 'done')
