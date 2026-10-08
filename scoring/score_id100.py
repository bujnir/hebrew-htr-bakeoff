"""Score every prediction file in results/predictions/id100/ against the 100 synthetic handwritten ID numbers.

    python scoring/score_id100.py

Non-digits are stripped from each prediction first. Per engine and per field style (all / comb boxes / free table cell):
  exact           all 9 digits right
  digit_acc       matching digits / 9 (0 when the prediction is not 9 digits long)
  checksum_pass   share of predictions that pass the Israeli ID check digit
  wrong_but_pass  wrong reads that still pass the check digit (would be accepted unreviewed)
Writes results/scores_id100.json and .md.
"""
import glob
import json
import os
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)
sys.path.insert(0, ROOT)
from common import load_manifest
from engines import name


def id_checksum_ok(s):
    if not (s.isdigit() and len(s) == 9):
        return False
    t = 0
    for i, ch in enumerate(s):
        v = int(ch) * (1 + i % 2)
        t += v - 9 if v > 9 else v
    return t % 10 == 0


S = load_manifest(os.path.join(ROOT, 'data', 'id100', 'manifest.json'))
out = []
for f in sorted(glob.glob(os.path.join(ROOT, 'results', 'predictions', 'id100', '*.jsonl'))):
    key = os.path.basename(f)[:-6]
    P = {j['id']: j['pred'] for j in map(json.loads, open(f, encoding='utf-8')) if 'error' not in j}
    if len(P) < len(S):
        out.append(dict(key=key, engine=name(key)[0], n=len(P), status='incomplete'))
        continue
    res = {}
    for grp in ('all', 'comb', 'table'):
        rs = [r for r in S if grp == 'all' or (r['style'] == 'comb') == (grp == 'comb')]
        d = [re.sub(r'\D', '', P[r['id']]) for r in rs]
        ex = [p == r['gt'] for p, r in zip(d, rs)]
        dig = [sum(a == b for a, b in zip(p, r['gt'])) / 9 if len(p) == 9 else 0 for p, r in zip(d, rs)]
        ok = [id_checksum_ok(p) for p in d]
        res[grp] = dict(n=len(rs), exact=round(sum(ex) / len(rs), 2), digit_acc=round(sum(dig) / len(rs), 3),
                        checksum_pass=round(sum(ok) / len(rs), 2), wrong_but_pass=sum(o and not e for o, e in zip(ok, ex)))
    out.append(dict(key=key, engine=name(key)[0], runs=name(key)[1], **res))
out.sort(key=lambda r: -r.get('all', {}).get('exact', -1))
json.dump(out, open(os.path.join(ROOT, 'results', 'scores_id100.json'), 'w', encoding='utf-8'), ensure_ascii=False, indent=1)

md = ['| Engine | Runs on | Exact | Exact, boxes / free | Digit accuracy | Wrong but passes check digit |', '|---|---|---|---|---|---|']
for r in out:
    if 'all' not in r:
        md.append(f"| {r['engine']} | | incomplete ({r['n']}) | | | |")
        continue
    a = r['all']
    md.append(f"| {r['engine']} | {r['runs']} | {a['exact'] * 100:.0f}% | {r['comb']['exact'] * 100:.0f}% / {r['table']['exact'] * 100:.0f}% "
              f"| {a['digit_acc'] * 100:.1f}% | {a['wrong_but_pass']} |")
open(os.path.join(ROOT, 'results', 'scores_id100.md'), 'w', encoding='utf-8').write('\n'.join(md) + '\n')
print('\n'.join(md))
