# Hebrew handwriting OCR and VLM bake-off

Code, raw outputs and scores for our comparison of 19 OCR engines and vision-language models (VLMs) on
handwritten Hebrew, run in September–October 2026 for a final project in the AI engineering program at the
Hebrew University of Jerusalem (Nir Bujanover, nirbuj@gmail.com).

Two test sets:

1. **ivrit.ai Hebrew Handwriting OCR Benchmark**, lines config: 225 real handwritten line crops
   ([ivrit-ai/hebrew-handwriting-ocr-benchmark](https://huggingface.co/datasets/ivrit-ai/hebrew-handwriting-ocr-benchmark), gated).
   Used **for testing only**; nothing here was trained or tuned on it.
2. **id100**: 100 handwritten 9-digit Israeli ID numbers cut from our synthetic forms (50 written one digit per
   box, 50 written freely in a table cell; scan-like and phone-photo-like pages). Fully synthetic, included in `data/id100/`.

## Results

### ivrit.ai benchmark, 225 lines

| Engine | Runs on | Median CER | Median CER, blanks = 1 | Blank | CER mean ± SD [95% CI] (capped) | WER mean ± SD (capped) | Exact lines |
|---|---|---|---|---|---|---|---|
| Gemini 3.8 Flash | cloud | 0.149 | 0.160 | 6 | 0.35 ± 0.39 [0.30, 0.40] | 0.48 ± 0.41 | 27.6% |
| Mishkefet-v1 | local CPU | 0.189 | 0.194 | 2 | 0.24 ± 0.20 [0.22, 0.27] | 0.64 ± 0.29 | 4.9% |
| Gemini 3.1 Pro | cloud | 0.194 | 0.194 | 0 | 0.26 ± 0.27 [0.22, 0.29] | 0.45 ± 0.36 | 24.0% |
| Claude Opus 5.5 | cloud | 0.214 | 0.214 | 0 | 0.29 ± 0.26 [0.26, 0.32] | 0.55 ± 0.34 | 11.6% |
| GPT-5.6-Sol | cloud | 0.450 | 0.450 | 0 | 0.43 ± 0.27 [0.40, 0.47] | 0.72 ± 0.32 | 6.7% |
| HebHTR (2021) | local CPU | 0.714 | 0.714 | 0 | 0.66 ± 0.24 [0.63, 0.69] | 0.92 ± 0.18 | 2.2% |
| Gemma 4 31B | open weights (run via OpenRouter) | 0.737 | 0.737 | 0 | 0.68 ± 0.26 [0.64, 0.71] | 0.91 ± 0.21 | 1.8% |
| Qwen3-VL-8B Hebrew-manuscript fine-tune (isaacmg) | own GPU (vLLM) | 0.750 | 0.750 | 0 | 0.75 ± 0.20 [0.72, 0.78] | 0.98 ± 0.06 | 0.0% |
| Llama 4 Maverick | open weights (run via OpenRouter) | 0.750 | 0.750 | 1 | 0.72 ± 0.21 [0.70, 0.75] | 0.95 ± 0.13 | 0.0% |
| Qwen3.8-27B | open weights (run via OpenRouter) | 0.773 | 0.773 | 0 | 0.78 ± 0.13 [0.77, 0.80] | 0.98 ± 0.05 | 0.0% |
| Qwen3-VL-32B | open weights (run via OpenRouter) | 0.833 | 0.833 | 0 | 0.84 ± 0.13 [0.82, 0.86] | 0.98 ± 0.08 | 0.4% |
| GLM-4.6V | open weights (run via OpenRouter) | 0.846 | 0.850 | 5 | 0.86 ± 0.12 [0.84, 0.88] | 0.99 ± 0.07 | 0.4% |
| Mistral Small 3.2 | open weights (run via OpenRouter) | 0.905 | 0.905 | 0 | 0.90 ± 0.11 [0.88, 0.91] | 1.00 ± 0.02 | 0.0% |
| PaddleOCR-VL 1.6 | own GPU (vLLM) | 1.000 | 1.000 | 0 | 0.97 ± 0.05 [0.97, 0.98] | 1.00 ± 0.01 | 0.0% |
| GLM-OCR 0.9B | local CPU | 1.000 | 1.000 | 0 | 0.94 ± 0.08 [0.93, 0.95] | 0.99 ± 0.03 | 0.0% |
| Qwen3.6-35B-A3B | open weights (run via OpenRouter) | 1.000 | 1.000 | 1 | 0.92 ± 0.12 [0.90, 0.93] | 0.99 ± 0.03 | 0.0% |
| Qwen3-VL-8B | open weights (run via OpenRouter) | 1.308 | 1.308 | 0 | 0.97 ± 0.08 [0.96, 0.98] | 0.99 ± 0.04 | 0.0% |
| Mistral OCR | cloud (OpenRouter file parser) | 1.716 | 1.716 | 0 | 0.97 ± 0.06 [0.96, 0.98] | 1.00 ± 0.02 | 0.0% |
| TrOCR-Hebrew (cyttic exp15) | local CPU | 1.927 | 1.927 | 0 | 0.99 ± 0.04 [0.98, 0.99] | 1.00 ± 0.00 | 0.0% |

- **Median CER** is the leaderboard metric: per-line character error rate, blank outputs dropped, median over lines.
  The next column charges blanks as 1.0.
- **Mean ± SD [95% CI]** uses per-line CER capped at 1.0, so a few runaway outputs cannot dominate;
  CI from 2,000 bootstrap resamples of the 225 lines (seed 0). Uncapped values are in `results/scores_ivrit.json`.
- **Exact lines**: prediction identical to the transcription after normalisation.
- Failed API calls (Gemini 3.8 Flash: 5 of 225) count as blank outputs.

### id100, handwritten 9-digit IDs

| Engine | Runs on | Exact | Exact, boxes / free | Digit accuracy | Wrong but passes check digit |
|---|---|---|---|---|---|
| Gemini 3.1 Pro | cloud | 91% | 92% / 90% | 98.3% | 0 |
| Gemini 3.8 Flash | cloud | 91% | 92% / 90% | 97.9% | 3 |
| Claude Opus 5.5 | cloud | 90% | 88% / 92% | 98.3% | 0 |
| Gemma 4 31B | open weights (run via OpenRouter) | 85% | 84% / 86% | 96.4% | 1 |
| Qwen3-VL-8B | open weights (run via OpenRouter) | 85% | 84% / 86% | 95.9% | 0 |
| Qwen3-VL-32B | open weights (run via OpenRouter) | 83% | 82% / 84% | 94.9% | 1 |
| Qwen3.8-27B | open weights (run via OpenRouter) | 82% | 84% / 80% | 96.9% | 0 |
| Our digit readers (CellNet + NumCRNN) | local CPU | 81% | 78% / 84% | 94.7% | 0 |
| PaddleOCR-VL 1.6 | own GPU (vLLM) | 76% | 70% / 82% | 87.8% | 0 |
| Qwen3.6-35B-A3B | open weights (run via OpenRouter) | 76% | 78% / 74% | 83.9% | 2 |
| GPT-5.6-Sol | cloud | 68% | 74% / 62% | 93.1% | 4 |
| Llama 4 Maverick | open weights (run via OpenRouter) | 66% | 58% / 74% | 87.3% | 1 |
| Mistral OCR | cloud (OpenRouter file parser) | 62% | 44% / 80% | 72.1% | 1 |
| GLM-4.6V | open weights (run via OpenRouter) | 62% | 58% / 66% | 81.1% | 2 |
| Mistral Small 3.2 | open weights (run via OpenRouter) | 55% | 54% / 56% | 74.2% | 1 |
| Qwen3-VL-8B Hebrew-manuscript fine-tune (isaacmg) | own GPU (vLLM) | 32% | 26% / 38% | 38.2% | 6 |
| Mishkefet-v1 | local CPU | 0% | 0% / 0% | 0.0% | 0 |

- Non-digits are stripped before scoring. **Digit accuracy** = matching digits / 9 (0 if the read is not 9 digits).
- **Wrong but passes check digit**: misreads that the Israeli ID check digit cannot catch.
- Caveat: these are MNIST-style synthetic digits; real forms will be harder.

## What is in the repo

```
common.py                   keys (.env), manifests, and the send-guard (remote calls only for images under data/)
scripts/prepare_ivrit.py    downloads the gated benchmark into data/ivrit/ (git-ignored)
data/id100/                 the 100 synthetic ID crops + manifest (field box geometry for the boxed ones)
runners/
  openai_compat.py          any vision model via OpenRouter, or our own vLLM server (same API)
  mistral_ocr.py            Mistral OCR via OpenRouter's file-parser plugin (each crop as a 1-page PDF)
  mishkefet.py              Mishkefet-v1 on CPU, heb-ocr default line settings (char LM + beam 12 + TTA x3)
  hebhtr.py                 original HebHTR (TF 1.12) with word segmentation
  hf_local.py               TrOCR-Hebrew and GLM-OCR on CPU
  digit_readers.py          our two small digit readers (digits/readers.py, weights in digits/models/)
  vllm_runpod/pod.py        start / wait / stop a one-GPU vLLM server on RunPod
scoring/
  score_ivrit.py            all ivrit.ai metrics -> results/scores_ivrit.json / .md
  score_id100.py            all ID metrics -> results/scores_id100.json / .md
  hebocr/                   leaderboard metric + text normalisation, vendored unchanged from heb-ocr (MIT)
results/predictions/        every engine's raw output, one JSON line per image (id, pred, and for API runs the raw reply, time, cost)
run_all.sh                  re-runs everything and re-scores
```

## Reproduce

```bash
pip install -r requirements.txt
cp .env.example .env                       # add OPENROUTER_API_KEY (and HF_TOKEN)
huggingface-cli login                      # after accepting the benchmark's terms on the Hub
python scripts/prepare_ivrit.py            # -> data/ivrit/
python scoring/score_ivrit.py              # re-score the stored predictions
python scoring/score_id100.py
```

Re-running the engines: `run_all.sh`. One engine, for example:

```bash
python runners/openai_compat.py google/gemini-3.1-pro-preview data/ivrit/manifest.json results/predictions/ivrit/or__google_gemini-3.1-pro-preview.jsonl
TASK=id python runners/openai_compat.py google/gemini-3.1-pro-preview data/id100/manifest.json results/predictions/id100/or__google_gemini-3.1-pro-preview.jsonl
```

### Settings

- All API models: temperature 0, one image per request, answer requested inside `<answer>` tags (the runner keeps the
  text inside them). Reasoning models ran at low effort with up to 8,000 output tokens; Gemma 4 and Qwen3.6 were
  run with reasoning off because their reasoning mode returned empty answers.
- Prompts are in `runners/openai_compat.py` (`text` for the benchmark, `id` for the IDs).
- Open-weight models in the "run via OpenRouter" rows are the same weights anyone can run locally; we used hosted
  endpoints for convenience.

### Own-GPU models (RunPod + vLLM)

```bash
python runners/vllm_runpod/pod.py start hebvl isaacmg/qwen3-vl-8b-hebrew-v20a-merged hebvl
python runners/vllm_runpod/pod.py wait <pod_id>
export BASE_URL=https://<pod_id>-8000.proxy.runpod.net/v1 API_KEY=$VLLM_API_KEY EFFORT=omit MAXTOK=300 WORKERS=12
PROMPT=$'This image is a line of modern handwritten Hebrew text (a note or a form).\n\nTranscribe the text exactly as written, in reading order.\nDo NOT correct, restore, or complete from memory.\n\nReturn ONLY the transcription.' \
  python runners/openai_compat.py hebvl data/ivrit/manifest.json results/predictions/ivrit/gpu__qwen3-vl-8b-hebrew-ft.jsonl
PROMPT=$'This image is a handwritten field from an Israeli form containing a 9-digit ID number.\n\nTranscribe the digits exactly as written.\n\nReturn ONLY the digits.' \
  python runners/openai_compat.py hebvl data/id100/manifest.json results/predictions/id100/gpu__qwen3-vl-8b-hebrew-ft.jsonl
python runners/vllm_runpod/pod.py stop <pod_id>        # pods bill by the hour
```

PaddleOCR-VL 1.6 ran the same way on its own pod. Both GPU runs together cost under $1.

### Notes on reproducibility

- Hosted models change behind the same name; expect small differences on a re-run.
- The prompts for the OpenRouter ID runs and for PaddleOCR-VL are reconstructed from our notes; the ivrit.ai
  text prompt and the HebVL prompts are exactly as run.
- The ivrit.ai leaderboard's own scoring harness is private; `scoring/hebocr/` is the reconstruction published with
  Mishkefet. Our Mishkefet median (0.189) is close to its leaderboard 0.175.

## Data policy

- `data/ivrit/` is git-ignored. Do not commit the benchmark images or transcriptions; download them with your
  own Hub access. `results/predictions/ivrit/` holds model outputs only.
- Remote runners refuse any image outside `data/` (`common.guard`). Never point them at real documents.

## Licences

- Code in this repo: MIT (see `LICENSE`). `scoring/hebocr/` is MIT, © Itay Inbar (heb-ocr).
- `digits/models/*.pt` and `data/id100/`: ours, CC-BY-4.0. The id100 digits are drawn from MNIST (CC BY-SA 3.0) and
  OFL-licensed Hebrew handwriting fonts (Playpen Sans Hebrew, Karantina, Amatic SC, Gveret Levin).
- Third-party engines keep their own licences; Mishkefet-v1 weights are CC-BY-NC-SA-4.0.
- The ivrit.ai benchmark is under the [ivrit.ai licence](https://www.ivrit.ai/en/the-license/).
