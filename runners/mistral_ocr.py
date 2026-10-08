"""Mistral OCR through OpenRouter's file-parser plugin: each crop is wrapped as a one-page PDF.

    python runners/mistral_ocr.py data/ivrit/manifest.json results/predictions/ivrit/or__mistral-ocr.jsonl

The chat model is only a carrier (asked to reply "OK"); the OCR text comes back in message.annotations.
"""
import base64
import io
import json
import os
import sys
import threading
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor

from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from common import done_ids, env, guard, load_manifest

manifest, out = sys.argv[1], sys.argv[2]
KEY = env('OPENROUTER_API_KEY')
lock = threading.Lock()


def pdf_b64(p):
    buf = io.BytesIO()
    Image.open(p).convert('RGB').save(buf, format='PDF', resolution=200)
    return base64.b64encode(buf.getvalue()).decode()


def call(r):
    p = guard(r['path'])
    body = {'model': 'mistralai/mistral-small-3.2-24b-instruct', 'max_tokens': 3,
            'messages': [{'role': 'user', 'content': [
                {'type': 'text', 'text': 'Reply with OK.'},
                {'type': 'file', 'file': {'filename': 'page.pdf', 'file_data': 'data:application/pdf;base64,' + pdf_b64(p)}}]}],
            'plugins': [{'id': 'file-parser', 'pdf': {'engine': 'mistral-ocr'}}]}
    err = ''
    for a in range(6):
        try:
            req = urllib.request.Request('https://openrouter.ai/api/v1/chat/completions', data=json.dumps(body).encode(),
                                         headers={'Authorization': 'Bearer ' + KEY, 'Content-Type': 'application/json'})
            d = json.load(urllib.request.urlopen(req, timeout=180))
            ann = d['choices'][0]['message'].get('annotations') or []
            parts = [c['text'] for x in ann if x.get('type') == 'file' for c in x['file'].get('content', [])
                     if c.get('type') == 'text' and not c['text'].startswith(('<file', '</file'))]
            return {'id': r['id'], 'pred': ' '.join(' '.join(parts).replace('#', ' ').split()),
                    'cost': d.get('usage', {}).get('cost')}
        except urllib.error.HTTPError as e:
            err = f'{e.code} {e.read()[:200]}'
        except Exception as e:
            err = str(e)[:200]
        time.sleep(8 * (a + 1))
    return {'id': r['id'], 'pred': '', 'error': err}


def work(r):
    res = call(r)
    with lock, open(out, 'a', encoding='utf-8') as f:
        f.write(json.dumps(res, ensure_ascii=False) + '\n')


rows = load_manifest(manifest)
done = done_ids(out)
with ThreadPoolExecutor(int(os.environ.get('WORKERS', 4))) as ex:
    list(ex.map(work, [r for r in rows if r['id'] not in done]))
print('done')
