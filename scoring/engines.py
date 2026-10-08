"""Display name and where each engine ran, keyed by prediction file name (results/predictions/<set>/<key>.jsonl)."""
ENGINES = {
    'local__mishkefet-v1': ('Mishkefet-v1', 'local CPU'),
    'local__hebhtr': ('HebHTR (2021)', 'local CPU'),
    'local__trocr-hebrew': ('TrOCR-Hebrew (cyttic exp15)', 'local CPU'),
    'local__glm-ocr': ('GLM-OCR 0.9B', 'local CPU'),
    'local__our-digit-readers': ('Our digit readers (CellNet + NumCRNN)', 'local CPU'),
    'gpu__qwen3-vl-8b-hebrew-ft': ('Qwen3-VL-8B Hebrew-manuscript fine-tune (isaacmg)', 'own GPU (vLLM)'),
    'gpu__paddleocr-vl-1.6': ('PaddleOCR-VL 1.6', 'own GPU (vLLM)'),
    'or__mistral-ocr': ('Mistral OCR', 'cloud (OpenRouter file parser)'),
    'or__google_gemini-3.8-flash': ('Gemini 3.8 Flash', 'cloud'),
    'or__google_gemini-3.1-pro-preview': ('Gemini 3.1 Pro', 'cloud'),
    'or__anthropic_claude-opus-5.5': ('Claude Opus 5.5', 'cloud'),
    'or__openai_gpt-5.6-sol': ('GPT-5.6-Sol', 'cloud'),
    'or__google_gemma-4-31b-it': ('Gemma 4 31B', 'open weights (run via OpenRouter)'),
    'or__meta-llama_llama-4-maverick': ('Llama 4 Maverick', 'open weights (run via OpenRouter)'),
    'or__qwen_qwen3.8-27b': ('Qwen3.8-27B', 'open weights (run via OpenRouter)'),
    'or__qwen_qwen3-vl-32b-instruct': ('Qwen3-VL-32B', 'open weights (run via OpenRouter)'),
    'or__qwen_qwen3-vl-8b-instruct': ('Qwen3-VL-8B', 'open weights (run via OpenRouter)'),
    'or__qwen_qwen3.6-35b-a3b': ('Qwen3.6-35B-A3B', 'open weights (run via OpenRouter)'),
    'or__z-ai_glm-4.6v': ('GLM-4.6V', 'open weights (run via OpenRouter)'),
    'or__mistralai_mistral-small-3.2-24b-instruct': ('Mistral Small 3.2', 'open weights (run via OpenRouter)'),
}


def name(key):
    return ENGINES.get(key, (key, '?'))
