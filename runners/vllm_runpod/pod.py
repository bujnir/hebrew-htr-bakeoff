"""Start, wait for, and stop a one-GPU vLLM server on RunPod (OpenAI-compatible API on port 8000).

    python runners/vllm_runpod/pod.py start hebvl isaacmg/qwen3-vl-8b-hebrew-v20a-merged hebvl
    python runners/vllm_runpod/pod.py start paddle PaddlePaddle/PaddleOCR-VL paddleocr-vl-1.6 "--trust-remote-code"
    python runners/vllm_runpod/pod.py wait <pod_id>
    python runners/vllm_runpod/pod.py stop <pod_id>

Needs RUNPOD_API_KEY, HF_TOKEN and VLLM_API_KEY (any random string; the server requires it).
Logs are served on port 8001 (https://<pod_id>-8001.proxy.runpod.net/vllm.txt).
Then point runners/openai_compat.py at it:
    BASE_URL=https://<pod_id>-8000.proxy.runpod.net/v1 API_KEY=$VLLM_API_KEY EFFORT=omit MAXTOK=300 ...
"""
import json
import os
import sys
import time
import urllib.error
import urllib.request

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from common import env

UA = {'User-Agent': 'hebrew-htr-bakeoff/1.0'}   # RunPod's CDN rejects requests without one


def rest(method, path, body=None):
    h = {'Authorization': 'Bearer ' + env('RUNPOD_API_KEY'), 'Content-Type': 'application/json', **UA}
    req = urllib.request.Request('https://rest.runpod.io/v1' + path, method=method, headers=h,
                                 data=json.dumps(body).encode() if body is not None else None)
    try:
        t = urllib.request.urlopen(req, timeout=120).read()
        return json.loads(t) if t else {}
    except urllib.error.HTTPError as e:
        raise RuntimeError(f'{e.code} {e.read()[:500]}')


def start(name, model, served, extra=''):
    cmd = ("mkdir -p /tmp/logs && cd /tmp/logs && (python3 -m http.server 8001 >/dev/null 2>&1 &) && "
           f"vllm serve {model} --port 8000 --max-model-len 8192 --api-key {env('VLLM_API_KEY')} --gpu-memory-utilization 0.9 "
           f"--served-model-name {served} --limit-mm-per-prompt '{{\"image\":1}}' {extra} > /tmp/logs/vllm.txt 2>&1; sleep 3600")
    body = {'name': name, 'imageName': 'vllm/vllm-openai:latest',
            'gpuTypeIds': ['NVIDIA A40', 'NVIDIA GeForce RTX 4090', 'NVIDIA L40S', 'NVIDIA RTX 6000 Ada Generation'],
            'gpuCount': 1, 'cloudType': 'SECURE', 'containerDiskInGb': 60, 'ports': ['8000/http', '8001/http'],
            'env': {'HF_TOKEN': env('HF_TOKEN')}, 'dockerEntrypoint': ['bash', '-c'], 'dockerStartCmd': [cmd]}
    p = rest('POST', '/pods', body)
    print(p['id'], p.get('costPerHr'), p.get('machine', {}).get('gpuTypeId'))


def wait(pid, timeout=900):
    url = f'https://{pid}-8000.proxy.runpod.net/v1/models'
    t0, last = time.time(), ''
    while time.time() - t0 < timeout:
        try:
            req = urllib.request.Request(url, headers={'Authorization': 'Bearer ' + env('VLLM_API_KEY'), **UA})
            r = json.load(urllib.request.urlopen(req, timeout=20))
            print('UP', r['data'][0]['id'], round(time.time() - t0), 's')
            return
        except urllib.error.HTTPError as e:
            last = f'HTTP {e.code}'
        except Exception as e:
            last = str(e)[:80]
        time.sleep(20)
    print('not up yet:', last)


if __name__ == '__main__':
    cmd, args = sys.argv[1], sys.argv[2:]
    if cmd == 'start':
        start(*args)
    elif cmd == 'wait':
        wait(args[0])
    elif cmd == 'stop':
        rest('DELETE', f'/pods/{args[0]}')
        print('terminated', args[0])
