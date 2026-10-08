"""Shared helpers: repo paths, API keys, manifests, and the send-guard.

GUARD: runners that call a remote API refuse any image outside data/ (the public
ivrit.ai benchmark and our synthetic ID crops). Never point them at real documents.
"""
import json
import os

ROOT = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(ROOT, 'data')


def env(name):
    """Read a key from the environment, else from .env in the repo root."""
    if os.environ.get(name):
        return os.environ[name]
    path = os.path.join(ROOT, '.env')
    if os.path.exists(path):
        for line in open(path, encoding='utf-8-sig'):
            line = line.strip()
            if line and not line.startswith('#') and '=' in line:
                k, v = line.split('=', 1)
                if k.strip() == name:
                    return v.strip().strip('"\'')
    raise SystemExit(f'missing {name}: set it in the environment or in .env (see .env.example)')


def load_manifest(path):
    """List of {id, path, gt, ...}; relative image paths are resolved against the manifest's folder."""
    base = os.path.dirname(os.path.abspath(path))
    rows = json.load(open(path, encoding='utf-8'))
    for r in rows:
        r['path'] = os.path.normpath(os.path.join(base, r['path']))
    return rows


def guard(path):
    p = os.path.abspath(path)
    if not p.startswith(DATA + os.sep):
        raise SystemExit(f'refusing to send an image outside data/: {p}')
    return p


def done_ids(out):
    """Ids already written to a results file (lines with an error are dropped, so they are retried)."""
    if not os.path.exists(out):
        return set()
    keep = [j for j in map(json.loads, open(out, encoding='utf-8')) if 'error' not in j]
    with open(out, 'w', encoding='utf-8') as f:
        for j in keep:
            f.write(json.dumps(j, ensure_ascii=False) + '\n')
    return {j['id'] for j in keep}
