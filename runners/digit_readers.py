"""Our digit readers on the 100 ID crops: comb fields -> CellNet per box, free-written -> NumCRNN.

    python runners/digit_readers.py data/id100/manifest.json results/predictions/id100/local__our-digit-readers.jsonl
"""
import json
import os
import sys

import cv2
import numpy as np
import torch
from PIL import Image

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, HERE)
from common import load_manifest
from digits.readers import CellNet, NumCRNN, cell_tensor, comb_cells, ctc_greedy, remove_form_lines, strip_tensor

manifest, out = sys.argv[1], sys.argv[2]
M = os.path.join(HERE, 'digits', 'models')
cell = CellNet()
cell.load_state_dict(torch.load(os.path.join(M, 'cellnet.pt'), map_location='cpu'))
cell.eval()
crnn = NumCRNN()
crnn.load_state_dict(torch.load(os.path.join(M, 'numcrnn.pt'), map_location='cpu'))
crnn.eval()

with open(out, 'w', encoding='utf-8') as f, torch.no_grad():
    for r in load_manifest(manifest):
        rgb = np.asarray(Image.open(r['path']).convert('RGB'))
        if r['style'] == 'comb':
            cells = comb_cells(cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY), r['field_bbox'], 9)
            pr = cell(torch.stack([cell_tensor(c) for c in cells])).softmax(-1)
            pred = ''.join(str(int(k)) for k in pr.argmax(-1) if int(k) < 10)
        else:
            g = cv2.cvtColor(remove_form_lines(rgb), cv2.COLOR_RGB2GRAY)
            pred = ctc_greedy(crnn(strip_tensor(g)[None])[0])
        f.write(json.dumps({'id': r['id'], 'pred': pred}) + '\n')
print('done')
