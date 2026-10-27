# Bark 3.0 add-on for Subtitld

Offline English **speech-to-text** for [Subtitld](https://github.com/Subtitld/subtitld),
with [Conway Research's Bark 3.0](https://huggingface.co/ConwayResearch/Underdog-Bark-3.0)
models, fine-tunes of NVIDIA's
[parakeet-tdt-0.6b-v3](https://huggingface.co/nvidia/parakeet-tdt-0.6b-v3).

- Serves the `asr.transcribe` task over Subtitld's add-on protocol, with word
  times, regrouped into one subtitle row per sentence (or, with
  *One cue per sentence* off, rows of about two lines or 7 seconds at most).
- **English only.** Punctuation, capitals and numerals come out as written text.
- **CPU only, no torch, no cloud account.** ONNX Runtime runs the encoder;
  numpy runs the rest.
- Long files are transcribed in windows of about 30 seconds, with progress
  after each window. A cancel takes effect at the next window.

## Models

No model is in the archive. The one you pick downloads once, the first time
you use it, from Conway Research's Hugging Face repositories, at a pinned
revision, checked by sha256. NVIDIA's tokenizer, which they all share,
downloads with it (1 MB). The add-on then prepares the model for its graph,
once, in about half a minute. After that, transcription is fully offline.

| Model | Download | Notes |
|-------|----------|-------|
| `bark3` | 163 MB | **default**, verbatim |
| `bark3-plus` | 180 MB | verbatim, a little more accurate |
| `bark3-clean` | 179 MB | leaves out fillers (um, uh) and false starts: dictation style |
| `bark3-pro` | 1.25 GB | full precision, the most accurate |

Once prepared, every model takes about 650 MB of disk and runs at the same
speed: around 3% to 10% of the audio's length on a desktop CPU, with about
1.2 GB of memory. The downloaded file is deleted after preparation.

Word error rates measured with this add-on (8-bit encoder) on a small
LibriSpeech sample of 73 clips, 8 minutes; the differences between models
are within its noise:

| Model | Clips | The same clips as one file |
|-------|-------|----------------------------|
| Bark 3.0 | 4.5% | 4.1% |
| Plus | 3.8% | 4.6% |
| Clean | 4.4% | 3.9% |
| Pro | 3.6% | 4.0% |

The models are CC-BY-4.0. See [NOTICE](NOTICE) for the attribution.

### How a model is prepared

Bark 3.0, Plus and Clean are distributed as `model.bark`, an entropy-coded
format (ConwayResearch's `bark_loader.py` documents it); Pro as safetensors.
The archive carries the encoder's ONNX graph **without weights**
(`bark3_addon/graph/`, made by `tools/build_graph.py`) and the layout of the
file its weights are read from. On first use, `bark3_addon/unpack.py` decodes
each tensor, exactly, and `bark3_addon/convert.py` writes it into
`weights.bin` where the layout says: the encoder's matrices as 7-bit integers
(in int8, so the u8 × s8 kernel of CPUs without VNNI cannot saturate), one scale
per output channel, everything else as float32.

### Where models are stored

`$SUBTITLD_MODELS_DIR/bark3`, when the host exports it (Subtitld does), else
`~/.cache/subtitld/models/bark3`:

```
bark3/ConwayResearch--Underdog-Bark-3.0/<revision>/onnx/   prepared model
bark3/nvidia--parakeet-tdt-0.6b-v3/<revision>/tokenizer.json
```

Downloads go to a `.part` file, resume after an interruption, and are renamed
only once complete and checked. Uninstalling the add-on does **not** delete
models; remove the folder yourself to reclaim the space.

## Options

Set these in **Settings ▸ Add-ons ▸ Bark 3.0 ▸ Configure**.

| Option | Default | Env var | Notes |
|--------|---------|---------|-------|
| Model | `bark3` | `BARK3_MODEL` | See the table above. |
| CPU threads | `0` | `BARK3_THREADS` | `0` = half the logical processors, at most 8. |
| One cue per sentence | on | `BARK3_SLICE_BY_PHRASES` | On: one subtitle per sentence, however long (a sentence with no end closes at 500 characters or 30 s). Off: long sentences split at 84 characters or 7 s. |

Each request takes its option from the request's own `options`, then the
environment variable (Subtitld sets it from the Configure dialog), then the
default.

Other environment variables:

| Variable | Purpose |
|----------|---------|
| `SUBTITLD_FFMPEG_EXECUTABLE`, `FFMPEG_EXECUTABLE` | ffmpeg to use, checked in that order before `PATH`. |
| `SUBTITLD_MODELS_DIR` | Host model cache root (see above). |
| `HF_ENDPOINT` | A Hugging Face mirror for the downloads. |
| `BARK3_ADDON_LOG_LEVEL` | `DEBUG` for more detail. Default `INFO`. |
| `SSL_CERT_FILE`, `SSL_CERT_DIR` | CA certificates for the downloads. |

## Requirements

- ffmpeg, which Subtitld provides.
- 64-bit Linux, macOS on Apple silicon, or Windows.
- About 900 MB of free disk for the default model, 2 GB for Pro (while it is
  prepared), and 1.5 GB of free memory.

## Development

```bash
python3.12 -m venv .venv
.venv/bin/pip install -c constraints.txt -e '.[dev]'
.venv/bin/pytest            # test_model.py downloads Bark 3.0 (163 MB) once
```

Rebuilding the graph (only when the export changes) needs torch,
transformers 5.19, onnx and onnxruntime in a separate venv:

```bash
python tools/build_graph.py path/to/model.bark
python tools/pin_models.py  # re-pins the model revisions
```

The release workflow (`.github/workflows/release.yml`) freezes the add-on
with PyInstaller on Linux, macOS and Windows, installs each archive the way
Subtitld does and runs `tests/test_installed.py` against it.

## Licence

The add-on is Apache-2.0 ([LICENSE](LICENSE)). Each release archive lists
everything frozen into it in `THIRD_PARTY_LICENSES/`.
