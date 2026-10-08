"""Our two small digit readers for handwritten numbers on forms (CPU, PyTorch).

  CellNet  - one digit per comb box (ID, zip, branch). 11 classes: 0-9 + empty.
  NumCRNN  - free-written digit strings. CTC over '0123456789/.-,' so it can never output a letter.

Both were trained only on synthetic renders (HHD-style glyphs and MNIST digits from a split the
test crops never use, plus handwriting fonts); see the main project for the generator.
"""
import cv2
import numpy as np
import torch
import torch.nn as nn

NUM_CHARS = '0123456789/.-,'


def remove_form_lines(rgb):
    """Erase printed underlines, box borders and comb separators from a field crop (RGB uint8 array)."""
    g = cv2.cvtColor(np.asarray(rgb), cv2.COLOR_RGB2GRAY)
    h, w = g.shape
    bw = cv2.adaptiveThreshold(g, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 31, 15)
    horiz = cv2.morphologyEx(bw, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, (max(15, w // 6), 1)))
    vert = cv2.morphologyEx(bw, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, (1, max(10, int(h * 0.55)))))
    lines = cv2.dilate(cv2.bitwise_or(horiz, vert), np.ones((3, 3), np.uint8))
    out = g.copy()
    out[lines > 0] = int(np.median(g))
    return cv2.cvtColor(out, cv2.COLOR_GRAY2RGB)


# ---------------- comb cells ----------------

def cell_tensor(g):
    g = cv2.resize(g, (32, 40), interpolation=cv2.INTER_AREA).astype(np.float32) / 255.0
    return torch.from_numpy(1.0 - g)[None]   # ink = high


class CellNet(nn.Module):
    def __init__(self, n=11):
        super().__init__()

        def blk(i, o):
            return nn.Sequential(nn.Conv2d(i, o, 3, padding=1), nn.BatchNorm2d(o), nn.ReLU(),
                                 nn.Conv2d(o, o, 3, padding=1), nn.BatchNorm2d(o), nn.ReLU(), nn.MaxPool2d(2))
        self.f = nn.Sequential(blk(1, 32), blk(32, 64), blk(64, 128))
        self.h = nn.Sequential(nn.Flatten(), nn.Dropout(0.3), nn.Linear(128 * 5 * 4, 256), nn.ReLU(), nn.Linear(256, n))

    def forward(self, x):
        return self.h(self.f(x))


def comb_cells(gray, field_bbox, n):
    """Slice a rectified comb crop into its n cells, using the field's box on the blank form.
    The crop was cut with padding max(4, 2% w) left/right and 15% h top/bottom around the box."""
    x, y, w, h = field_bbox
    px, py = max(4, 0.02 * w), 0.15 * h
    sx, sy = gray.shape[1] / (w + 2 * px), gray.shape[0] / (h + 2 * py)
    cw = w / n
    return [gray[int(py * sy):int((py + h) * sy), int((px + i * cw) * sx):int((px + (i + 1) * cw) * sx)] for i in range(n)]


# ---------------- free-written numbers ----------------

def ink_crop(g, margin=6):
    """Crop a cleaned strip to its handwriting (the field box is mostly empty paper)."""
    b = cv2.GaussianBlur(g, (3, 3), 0)
    ink = b < (np.median(b) - 50)
    ink = cv2.morphologyEx(ink.astype(np.uint8), cv2.MORPH_OPEN, np.ones((2, 2), np.uint8)) > 0
    cols = np.where(ink.sum(0) >= 2)[0]
    rows = np.where(ink.sum(1) >= 2)[0]
    if len(cols) < 3 or len(rows) < 3:
        return g
    return g[max(rows[0] - margin, 0):rows[-1] + margin + 1, max(cols[0] - margin, 0):cols[-1] + margin + 1]


def strip_tensor(g, H=40, maxW=320):
    g = ink_crop(g)
    w = max(8, min(maxW, int(g.shape[1] * H / g.shape[0])))
    g = cv2.resize(g, (w, H), interpolation=cv2.INTER_AREA).astype(np.float32) / 255.0
    return torch.from_numpy(1.0 - g)[None]


class NumCRNN(nn.Module):
    def __init__(self, n=len(NUM_CHARS) + 1):
        super().__init__()

        def c(i, o, p):
            return [nn.Conv2d(i, o, 3, padding=1), nn.BatchNorm2d(o), nn.ReLU()] + ([nn.MaxPool2d(p)] if p else [])
        self.cnn = nn.Sequential(*c(1, 32, 2), *c(32, 64, 2), *c(64, 96, (2, 1)), *c(96, 96, None), *c(96, 128, (2, 1)))
        self.rnn = nn.LSTM(128 * 2, 128, num_layers=2, bidirectional=True, batch_first=True, dropout=0.2)
        self.out = nn.Linear(256, n)

    def forward(self, x):                        # x: B,1,40,W
        f = self.cnn(x)                          # B,128,2,W/4
        f = f.flatten(1, 2).transpose(1, 2)      # B,W/4,256
        return self.out(self.rnn(f)[0]).log_softmax(-1)


def ctc_greedy(lp):
    best = lp.argmax(-1).tolist()
    out, prev = [], 0
    for k in best:
        if k != prev and k != 0:
            out.append(NUM_CHARS[k - 1])
        prev = k
    return ''.join(out)
