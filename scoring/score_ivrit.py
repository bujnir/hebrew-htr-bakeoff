"""Score every prediction file in results/predictions/ivrit/ against the 225 ivrit.ai lines.

    python scoring/score_ivrit.py            # needs data/ivrit/manifest.json (scripts/prepare_ivrit.py)

Per engine:
  cer_median_lb      leaderboard number: median line CER, blank outputs dropped (the "Median CER" we report)
  cer_median_nodrop  same, blanks charged as CER 1.0
  cer / cer_capped   per-line CER mean, SD, 95% bootstrap CI (2000 resamples, seed 0); capped = each line min(CER, 1)
  wer / wer_capped   per-line word error rate, same statistics
  word_count_ratio   predicted words / true words per line
  exact              share of lines identical after normalisation
Failed API calls (at most 10 per engine) count as blank outputs. Writes results/scores_ivrit.json and .md.
"""
import glob
import json
import os
import sys

import numpy as np
from rapidfuzz.distance import Levenshtein

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)
sys.path.insert(0, ROOT)
from common import load_manifest
from engines import name
from hebocr.metrics import cer, line_report
from hebocr.normalize import normalize

man = load_manifest(os.path.join(ROOT, 'data', 'ivrit', 'manifest.json'))
rng = np.random.default_rng(0)


def stats(x):
    x = np.array(x, float)
    b = rng.choice(x, (2000, len(x))).mean(1)
    return dict(mean=round(float(x.mean()), 3), sd=round(float(x.std(ddof=1)), 3),
                ci95=[round(float(np.percentile(b, 2.5)), 3), round(float(np.percentile(b, 97.5)), 3)],
                median=round(float(np.median(x)), 3))


rows = []
for f in sorted(glob.glob(os.path.join(ROOT, 'results', 'predictions', 'ivrit', '*.jsonl'))):
    key = os.path.basename(f)[:-6]
    P = {j['id']: j for j in map(json.loads, open(f, encoding='utf-8')) if 'error' not in j}
    missing = len(man) - len(P)
    if missing > 10:
        rows.append(dict(key=key, engine=name(key)[0], n=len(P), status='incomplete'))
        continue
    refs = [m['gt'] for m in man]
    hyps = [P[m['id']]['pred'] if m['id'] in P else '' for m in man]
    rep = line_report(refs, hyps)
    c = [cer(r, h) for r, h in zip(refs, hyps)]
    w = [Levenshtein.distance(normalize(r).split(), normalize(h).split()) / max(len(normalize(r).split()), 1) for r, h in zip(refs, hyps)]
    wr = [len(normalize(h).split()) / max(len(normalize(r).split()), 1) for r, h in zip(refs, hyps)]
    rows.append(dict(key=key, engine=name(key)[0], runs=name(key)[1], failed_calls=missing,
                     cer_median_lb=round(rep.cer_median, 3), cer_median_nodrop=round(rep.cer_median_nodrop, 3), blank=rep.n_blank,
                     cer=stats(c), cer_capped=stats([min(x, 1.0) for x in c]), wer=stats(w), wer_capped=stats([min(x, 1.0) for x in w]),
                     word_count_ratio=stats(wr), exact=round(sum(normalize(r) == normalize(h) for r, h in zip(refs, hyps)) / len(refs), 3),
                     cost_usd=round(sum((P[m['id']].get('cost') or 0) for m in man if m['id'] in P), 3)))
rows.sort(key=lambda r: r.get('cer_median_nodrop', 9))
json.dump(rows, open(os.path.join(ROOT, 'results', 'scores_ivrit.json'), 'w', encoding='utf-8'), ensure_ascii=False, indent=1)

md = ['| Engine | Runs on | Median CER | Median CER, blanks = 1 | Blank | CER mean ± SD [95% CI] (capped) | WER mean ± SD (capped) | Exact lines |', '|---|---|---|---|---|---|---|---|']
for r in rows:
    if 'cer' not in r:
        md.append(f"| {r['engine']} | | incomplete ({r['n']}) | | | | | |")
        continue
    cc, wc = r['cer_capped'], r['wer_capped']
    md.append(f"| {r['engine']} | {r['runs']} | {r['cer_median_lb']:.3f} | {r['cer_median_nodrop']:.3f} | {r['blank']} | {cc['mean']:.2f} ± {cc['sd']:.2f} [{cc['ci95'][0]:.2f}, {cc['ci95'][1]:.2f}] "
              f"| {wc['mean']:.2f} ± {wc['sd']:.2f} | {r['exact'] * 100:.1f}% |")
open(os.path.join(ROOT, 'results', 'scores_ivrit.md'), 'w', encoding='utf-8').write('\n'.join(md) + '\n')
print('\n'.join(md))
