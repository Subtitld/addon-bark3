"""manifest.json, pins.json, the bundled graph and the code agree with each other.

Copyright 2026 Subtitld contributors
SPDX-License-Identifier: Apache-2.0
"""

from __future__ import annotations

import hashlib
import json
import re
import sys
import tomllib
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from bark3_addon import ADDON_ID, __version__, convert, models  # noqa: E402

MANIFEST = json.loads((REPO_ROOT / 'manifest.json').read_text(encoding='utf-8'))


def test_versions_agree():
    pyproject = tomllib.loads((REPO_ROOT / 'pyproject.toml').read_text(encoding='utf-8'))
    assert MANIFEST['version'] == __version__ == pyproject['project']['version']
    assert re.fullmatch(r'\d\d\.\d\d\.\d+', __version__), 'versions are YY.MM.N'
    assert MANIFEST['id'] == ADDON_ID


def test_manifest_lists_what_the_code_offers():
    assert MANIFEST['languages'] == ['en']
    assert MANIFEST['tasks'] == ['asr.transcribe']
    assert MANIFEST['compute'] == 'cpu'
    fields = {f['key']: f for f in MANIFEST['config_schema']['fields']}
    assert set(fields) == {'model', 'threads'}
    assert fields['model']['default'] == models.DEFAULT
    assert [o['value'] for o in fields['model']['options']] == [m['id'] for m in MANIFEST['models']] \
        == list(models.MODELS)
    assert [m['id'] for m in MANIFEST['models'] if m['default']] == [models.DEFAULT]


def test_models_match_the_pins():
    pins = models.pins()
    assert set(pins) == set(models.MODELS) | {'tokenizer'}
    for entry in MANIFEST['models']:
        pin = pins[entry['id']]
        assert entry['repo'] == pin['repo'] == models.MODELS[entry['id']]['repo']
        assert entry['revision'] == pin['revision']
        assert entry['size_mb'] == round(pin['size'] / 1e6)
        assert pin['path'] == models.MODELS[entry['id']]['path']
    for pin in pins.values():
        assert re.fullmatch(r'[0-9a-f]{64}', pin['sha256']) and pin['size'] > 0, pin
        assert re.fullmatch(r'[0-9a-f]{40}', pin['revision']), pin
    assert pins['tokenizer']['repo'] == models.TOKENIZER['repo']


def test_the_graph_and_its_layout_agree():
    graph = (convert.GRAPH_DIR / convert.GRAPH_FILE).read_bytes()
    layout = convert.layout()
    assert hashlib.sha256(graph).hexdigest() == layout['graph_sha256']
    # Weights are never in the graph itself.
    assert len(graph) < 4 * 2**20
    entries = sorted(layout['tensors'], key=lambda e: e['offset'])
    for before, after in zip(entries, entries[1:]):
        assert before['offset'] + before['nbytes'] <= after['offset'], (before['name'], after['name'])
    assert entries[-1]['offset'] + entries[-1]['nbytes'] == layout['weights_bytes']
    assert {e['recipe'] for e in entries} <= {'copy', 'transpose', 'qweight', 'qscale', 'qzero'}
    # Every quantized matrix has its scale and zero point.
    quantized = {e['source'] for e in entries if e['recipe'] == 'qweight'}
    assert len(quantized) > 200
    for recipe in ('qscale', 'qzero'):
        assert {e['source'] for e in entries if e['recipe'] == recipe} == quantized


def test_disk_and_memory_claims_cover_a_model():
    converted = convert.layout()['weights_bytes'] / 1e6
    assert MANIFEST['disk_required_mb'] >= converted + max(m['size_mb'] for m in MANIFEST['models'][:3])
    assert MANIFEST['ram_required_mb'] >= 1200
