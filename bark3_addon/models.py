"""The four Bark 3.0 models, and the files each one needs.

Copyright 2026 Subtitld contributors
SPDX-License-Identifier: Apache-2.0

None of them is in the archive. The chosen one downloads from ConwayResearch's
Hugging Face repositories on its first use, pinned by revision, size and
sha256 in pins.json (written by tools/pin_models.py), together with
parakeet-tdt-0.6b-v3's tokenizer from NVIDIA's repository, which every Bark
3.0 model shares. All of them are CC-BY-4.0.

`pad_sec` is the silence the model was evaluated with before the speech:
ConwayResearch's runtime adds 0.25 s for every model but Clean.
"""

from __future__ import annotations

import json
from pathlib import Path

DEFAULT = 'bark3'

MODELS = {
    'bark3': {'name': 'Bark 3.0', 'repo': 'ConwayResearch/Underdog-Bark-3.0',
              'path': 'model.bark', 'pad_sec': 0.25},
    'bark3-plus': {'name': 'Bark 3.0 Plus', 'repo': 'ConwayResearch/Underdog-Bark-3.0-Plus',
                   'path': 'model.bark', 'pad_sec': 0.25},
    'bark3-clean': {'name': 'Bark 3.0 Clean', 'repo': 'ConwayResearch/Underdog-Bark-3.0-Clean',
                    'path': 'model.bark', 'pad_sec': 0.0},
    'bark3-pro': {'name': 'Bark 3.0 Pro', 'repo': 'ConwayResearch/Underdog-Bark-3.0-Pro',
                  'path': 'model.safetensors', 'pad_sec': 0.25},
}
TOKENIZER = {'repo': 'nvidia/parakeet-tdt-0.6b-v3', 'path': 'tokenizer.json'}

PINS_FILE = Path(__file__).resolve().parent / 'pins.json'


def pins() -> dict:
    """{model id or 'tokenizer': {repo, revision, path, size, sha256}}."""
    return json.loads(PINS_FILE.read_text(encoding='utf-8'))
