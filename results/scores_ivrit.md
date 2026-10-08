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
