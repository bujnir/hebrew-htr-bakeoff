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
