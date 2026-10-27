"""Build bark3_addon/graph/: the encoder graph without its weights, and their layout.

    python tools/build_graph.py MODEL.bark

Needs torch, transformers 5.19, onnx and onnxruntime (a build-time venv, not
the add-on's own). The .bark file gives real weights for the export and the
check, but none of them are written to the output: every weight tensor in the
graph becomes external data in `weights.bin`, which the add-on writes on the
user's machine from the model it downloads (bark3_addon/convert.py).

Steps:
1. Load transformers' ParakeetForTDT with the .bark weights.
2. Make every matrix product a MatMul by a named parameter: each nn.Linear's
   weight is stored transposed as `<linear>.weight_t`, and the convolution
   module's two 1x1 convolutions become `<conv>.pointwise_conv{1,2}.weight_t`.
   Then ONNX Runtime's dynamic quantization reaches all of them, and every
   initializer name says which original tensor it comes from.
3. Export encoder + encoder_projector with constant folding off, so the names
   survive. Input `features` [1, frames, 128], output `encoded` [1, steps, 640].
4. Quantize dynamically (int8 per output channel, MatMulInteger).
5. Write layout.json (each weight-derived tensor: its source tensor, recipe,
   dtype, shape and offset in weights.bin, plus the decoder's and joint's
   float32 tensors, which numpy reads) and encoder.onnx, with every weight
   pointing into weights.bin.
6. Check: fill weights.bin from the .bark with bark3_addon.convert, run both
   ONNX Runtime and torch on random features, and compare.
"""

from __future__ import annotations

import hashlib
import json
import sys
import tempfile
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(HERE))
from bark3_addon import convert, unpack  # noqa: E402

OUT = HERE / 'bark3_addon' / 'graph'
OPSET = 17
# Initializers at most this big that come from no weight may stay inline: shape
# constants and the like. Anything bigger fails the build.
_INLINE_LIMIT = 64


def load_model(path: Path):
    import torch
    from transformers import ParakeetForTDT, ParakeetTDTConfig
    model = ParakeetForTDT(ParakeetTDTConfig.from_dict(unpack.config(path)))
    state = {name: torch.from_numpy(np.ascontiguousarray(a)) for name, a in unpack.tensors(path)}
    missing, unexpected = model.load_state_dict(state, strict=False)
    if unexpected or any('num_batches_tracked' not in m for m in missing):
        raise SystemExit(f'weights do not fit the model: missing {missing[:5]}, unexpected {unexpected[:5]}')
    return model.eval()


def matmul_everywhere(model) -> None:
    """Step 2: every product is x @ <parameter>, the parameter named after its source."""
    import torch
    from torch import nn
    from transformers.models.parakeet import modeling_parakeet as mp

    class TransposedLinear(nn.Module):
        def __init__(self, linear: nn.Linear):
            super().__init__()
            self.weight_t = nn.Parameter(linear.weight.detach().T.contiguous())
            self.bias = None if linear.bias is None else nn.Parameter(linear.bias.detach().clone())

        def forward(self, x):
            y = torch.matmul(x, self.weight_t)
            return y if self.bias is None else y + self.bias

    def swap(parent: nn.Module) -> None:
        for name, child in parent.named_children():
            if isinstance(child, nn.Linear):
                setattr(parent, name, TransposedLinear(child))
            else:
                swap(child)

    for conv in [m for m in model.modules() if isinstance(m, mp.ParakeetEncoderConvolutionModule)]:
        for which in ('pointwise_conv1', 'pointwise_conv2'):
            layer = getattr(conv, which)
            if layer.bias is not None:
                raise SystemExit('a pointwise convolution with a bias: extend matmul_everywhere')
            holder = nn.Module()
            holder.weight_t = nn.Parameter(layer.weight.detach().squeeze(-1).T.contiguous())
            setattr(conv, which, holder)

        def forward(self, hidden_states, attention_mask=None):
            # The module's own forward, in (batch, time, channels) for the 1x1 products.
            hidden_states = torch.matmul(hidden_states, self.pointwise_conv1.weight_t)
            hidden_states = nn.functional.glu(hidden_states, dim=-1).transpose(1, 2)
            hidden_states = self.depthwise_conv(hidden_states)
            hidden_states = self.norm(hidden_states)
            hidden_states = self.activation(hidden_states).transpose(1, 2)
            return torch.matmul(hidden_states, self.pointwise_conv2.weight_t)

        conv.forward = forward.__get__(conv)
    swap(model.encoder)
    model.encoder_projector = TransposedLinear(model.encoder_projector)



def export(model, path: Path) -> None:
    import torch

    class Wrapped(torch.nn.Module):
        def __init__(self, m):
            super().__init__()
            self.m = m

        def forward(self, features):
            return self.m.encoder_projector(self.m.encoder(input_features=features).last_hidden_state)

    with torch.inference_mode():
        torch.onnx.export(Wrapped(model).eval(), (torch.randn(1, 800, 128),), str(path),
                          input_names=['features'], output_names=['encoded'],
                          dynamic_axes={'features': {1: 'frames'}, 'encoded': {1: 'steps'}},
                          opset_version=OPSET, do_constant_folding=False, dynamo=False)


def quantize(src: Path, dst: Path) -> None:
    from onnxruntime.quantization import QuantType, quantize_dynamic
    quantize_dynamic(str(src), str(dst), weight_type=QuantType.QInt8, per_channel=True,
                     op_types_to_quantize=['MatMul'], use_external_data_format=True)


def _source(name: str) -> tuple[str, str] | None:
    """(source tensor, recipe) for an initializer name from the export, or None."""
    name = name.removeprefix('m.')
    for suffix, recipe in (('.weight_t_quantized', 'qweight'), ('.weight_t_scale', 'qscale'),
                           ('.weight_t_zero_point', 'qzero'), ('.weight_t', 'transpose')):
        if name.endswith(suffix):
            return name[:-len(suffix)] + '.weight', recipe
    if name.startswith(('encoder.', 'encoder_projector.')):
        return name, 'copy'
    return None


def templatize(quantized: Path, state_names: set[str]):
    """Step 5: the graph with every weight external, and the layout."""
    import onnx
    from onnx import numpy_helper
    model = onnx.load(str(quantized))
    entries, offset, inline = [], 0, []
    for tensor in model.graph.initializer:
        source = _source(tensor.name)
        if source is None or source[0] not in state_names:
            array = numpy_helper.to_array(tensor)
            if array.size > _INLINE_LIMIT:
                raise SystemExit(f'{tensor.name}: {array.shape} {array.dtype} has no source weight')
            inline.append(tensor.name)
            continue
        array = numpy_helper.to_array(tensor)
        offset = -(-offset // 64) * 64
        entries.append({'name': tensor.name, 'source': source[0], 'recipe': source[1],
                        'dtype': array.dtype.name, 'shape': list(array.shape),
                        'offset': offset, 'nbytes': array.nbytes})
        tensor.ClearField('raw_data')
        for field in ('float_data', 'int32_data', 'int64_data'):
            tensor.ClearField(field)
        tensor.data_location = onnx.TensorProto.EXTERNAL
        del tensor.external_data[:]
        for key, value in (('location', convert.WEIGHTS_FILE), ('offset', str(offset)),
                           ('length', str(array.nbytes))):
            item = tensor.external_data.add()
            item.key, item.value = key, value
        offset += array.nbytes
    return model, entries, offset, inline


def main() -> int:
    if len(sys.argv) != 2:
        print(__doc__)
        return 2
    import gc

    import torch
    bark = Path(sys.argv[1])
    model = load_model(bark)
    names = set(model.state_dict())
    # The reference for step 6, taken before the model is changed: one copy of
    # the model fits in memory during the export, two do not.
    features = np.random.default_rng(0).standard_normal((1, 600, 128)).astype(np.float32)
    with torch.inference_mode():
        want = model.get_audio_features(input_features=torch.from_numpy(features)).pooler_output.numpy()
    numpy_state = {n: t.numpy().copy() for n, t in model.state_dict().items()
                   if n.startswith(('decoder.', 'joint.'))}
    matmul_everywhere(model)
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        export(model, tmp / 'fp32.onnx')
        del model
        gc.collect()
        quantize(tmp / 'fp32.onnx', tmp / 'int8.onnx')
        graph, entries, offset, inline = templatize(tmp / 'int8.onnx', names)
    # The decoder and joint run in numpy, in float32, from the same file.
    numpy_entries = []
    for name, array in numpy_state.items():
        offset = -(-offset // 64) * 64
        numpy_entries.append({'name': name, 'source': name, 'recipe': 'copy', 'dtype': 'float32',
                              'shape': list(array.shape), 'offset': offset, 'nbytes': array.nbytes})
        offset += array.nbytes
    OUT.mkdir(parents=True, exist_ok=True)
    data = graph.SerializeToString()
    del graph
    (OUT / 'encoder.onnx').write_bytes(data)
    layout = {
        'format': 1,
        'graph_sha256': hashlib.sha256(data).hexdigest(),
        'weights_bytes': offset,
        'tensors': entries + numpy_entries,
    }
    (OUT / 'layout.json').write_text(json.dumps(layout, indent=1) + '\n', encoding='utf-8')
    print(f'encoder.onnx: {len(data):,} bytes, {len(entries)} external tensors, inline: {inline}')
    print(f'weights.bin: {offset / 1e6:.0f} MB, {len(numpy_entries)} decoder/joint tensors')
    return check(bark, features, want, numpy_state)


def check(bark: Path, features: np.ndarray, want: np.ndarray, numpy_state: dict) -> int:
    """Step 6."""
    import onnxruntime as ort
    with tempfile.TemporaryDirectory() as tmp:
        folder = convert.build(bark, Path(tmp) / 'model', progress=lambda *_: None)
        session = ort.InferenceSession(str(folder / 'encoder.onnx'), providers=['CPUExecutionProvider'])
        got = session.run(None, {'features': features})[0]
        error = np.abs(got - want).max() / np.abs(want).max()
        weights = convert.numpy_tensors(folder)
        exact = all(np.array_equal(weights[n], a) for n, a in numpy_state.items())
        del session, weights
    print(f'check: encoder max error {error:.4f} of the largest output (int8), decoder/joint exact: {exact}')
    return 0 if error < 0.1 and exact else 1

if __name__ == '__main__':
    sys.exit(main())
