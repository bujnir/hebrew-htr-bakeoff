"""Any vision model behind an OpenAI-compatible chat API: OpenRouter (default) or our own vLLM server.

    python runners/openai_compat.py google/gemini-3.8-flash data/ivrit/manifest.json results/predictions/ivrit/or__google_gemini-3.8-flash.jsonl
    TASK=id python runners/openai_compat.py google/gemini-3.8-flash data/id100/manifest.json results/predictions/id100/or__google_gemini-3.8-flash.jsonl

Environment:
    TASK      text (default) | id           picks the default prompt
    PROMPT    overrides the prompt
    EFFORT    low (default) | off | omit    reasoning setting sent to OpenRouter (off: reasoning disabled; omit: no field)
    MAXTOK    max output tokens (default 8000; reasoning models need room)
    BASE_URL  default https://openrouter.ai/api/v1; set to a vLLM server's /v1 to use it
    API_KEY   key for BASE_URL (default: OPENROUTER_API_KEY)
    WORKERS   parallel requests (default 6)

Resumable: rows already in the output file are skipped; failed rows are retried on the next run.
temperature 0; the answer is taken from <answer>...</answer> when the model uses the tags.
"""
import base64
import json
import os
import re
import sys
import threading
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from common import done_ids, env, guard, load_manifest

PROMPTS = {
    'text': ('Transcribe the handwritten Hebrew text in this image exactly as written, including punctuation. '
             'Reply with the transcription inside <answer></answer> tags and nothing else.'),
    'id': ('This image is a handwritten field from an Israeli form containing a 9-digit ID number. '
           'Transcribe the digits exactly as written. '
           'Reply with the digits inside <answer></answer> tags and nothing else.'),
}

model, manifest, out = sys.argv[1], sys.argv[2], sys.argv[3]
BASE = os.environ.get('BASE_URL', 'https://openrouter.ai/api/v1')
KEY = os.environ.get('API_KEY') or env('OPENROUTER_API_KEY')
PROMPT = os.environ.get('PROMPT') or PROMPTS[os.environ.get('TASK', 'text')]
EFFORT = os.environ.get('EFFORT', 'low')
lock = threading.Lock()


def extract(txt):
    m = re.findall(r'<answer>(.*?)</answer>', txt, re.S) or re.findall(r'<answer>(.*)', txt, re.S)
    if m:
        return m[-1].strip()
    lines = [l for l in txt.strip().splitlines() if l.strip()]
    return lines[-1].strip() if lines else ''


def call(row):
    p = guard(row['path'])
    mime = 'image/jpeg' if p.lower().endswith(('.jpg', '.jpeg')) else 'image/png'
    url = f"data:{mime};base64,{base64.b64encode(open(p, 'rb').read()).decode()}"
    body = {'model': model, 'temperature': 0, 'max_tokens': int(os.environ.get('MAXTOK', 8000)),
            'messages': [{'role': 'user', 'content': [{'type': 'text', 'text': PROMPT},
                                                      {'type': 'image_url', 'image_url': {'url': url}}]}]}
    if EFFORT == 'off':
        body['reasoning'] = {'enabled': False}
    elif EFFORT != 'omit':
        body['reasoning'] = {'effort': EFFORT, 'exclude': True}
    for attempt in range(6):
        t = time.time()
        try:
            req = urllib.request.Request(BASE + '/chat/completions', data=json.dumps(body).encode(),
                                         headers={'Authorization': 'Bearer ' + KEY, 'Content-Type': 'application/json',
                                                  'User-Agent': 'hebrew-htr-bakeoff'})
            d = json.load(urllib.request.urlopen(req, timeout=240))
            if 'choices' not in d:
                raise RuntimeError(str(d)[:200])
            raw = (d['choices'][0]['message'].get('content') or '').strip()
            return {'id': row['id'], 'pred': extract(raw), 'raw': raw[:500], 'sec': round(time.time() - t, 2),
                    'cost': d.get('usage', {}).get('cost')}
        except urllib.error.HTTPError as e:
            msg = e.read().decode(errors='ignore')[:300]
            if e.code in (408, 429, 500, 502, 503, 504):
                time.sleep(8 * (attempt + 1))
                continue
            return {'id': row['id'], 'pred': '', 'error': f'{e.code} {msg}'}
        except Exception:
            time.sleep(8 * (attempt + 1))
    return {'id': row['id'], 'pred': '', 'error': 'retries'}


def work(r):
    res = call(r)
    with lock, open(out, 'a', encoding='utf-8') as f:
        f.write(json.dumps(res, ensure_ascii=False) + '\n')
    return res


rows = load_manifest(manifest)
os.makedirs(os.path.dirname(os.path.abspath(out)), exist_ok=True)
done = done_ids(out)
with ThreadPoolExecutor(int(os.environ.get('WORKERS', 6))) as ex:
    res = list(ex.map(work, [r for r in rows if r['id'] not in done]))
errs = [r for r in res if 'error' in r]
print(model, 'done', len(done) + len(res) - len(errs), '/', len(rows), 'errors', len(errs),
      errs[0]['error'][:150] if errs else '')
