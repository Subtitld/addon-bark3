"""Write bark3_addon/pins.json: each model's file and the tokenizer, pinned.

    python tools/pin_models.py

Pins each repository's current head. Sizes and sha256 come from Hugging Face's
LFS metadata; a file that is not in LFS is downloaded and hashed.
"""
import hashlib
import json
import sys
import urllib.request
from pathlib import Path

HERE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(HERE))
from bark3_addon.models import MODELS, PINS_FILE, TOKENIZER  # noqa: E402


def get(url: str) -> bytes:
    with urllib.request.urlopen(url, timeout=60) as response:
        return response.read()


def pin(repo: str, path: str) -> dict:
    revision = json.loads(get(f'https://huggingface.co/api/models/{repo}'))['sha']
    tree = {item['path']: item for item in json.loads(get(
        f'https://huggingface.co/api/models/{repo}/tree/{revision}?recursive=1'))}
    meta = tree[path]
    sha = (meta.get('lfs') or {}).get('oid')
    if not sha:
        data = get(f'https://huggingface.co/{repo}/resolve/{revision}/{path}')
        if len(data) != meta['size']:
            raise SystemExit(f'{repo}/{path}: got {len(data)} bytes, the tree says {meta["size"]}')
        sha = hashlib.sha256(data).hexdigest()
    print(f'{repo}/{path}: {meta["size"] / 1e6:.0f} MB at {revision}')
    return {'repo': repo, 'revision': revision, 'path': path, 'size': meta['size'], 'sha256': sha}


def main() -> int:
    out = {key: pin(model['repo'], model['path']) for key, model in MODELS.items()}
    out['tokenizer'] = pin(TOKENIZER['repo'], TOKENIZER['path'])
    PINS_FILE.write_text(json.dumps(out, indent=1) + '\n', encoding='utf-8')
    return 0


if __name__ == '__main__':
    sys.exit(main())
