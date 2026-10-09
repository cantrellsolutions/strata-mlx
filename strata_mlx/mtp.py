"""The model's own draft head: one extra block that guesses the tokens after the next one.

Cell i of its sequence pairs the trunk's residual at position i (all four streams, before the final mixer) with
the token at position i + 1, and predicts the token at position i + 2. Its own residual stands in for the
trunk's at the next draft step, so a few steps guess a few tokens ahead. It only guesses: the trunk decides
every token that is written.

The head is not in the GGUF files. Its 31 tensors come from the original 16-bit checkpoint, as Strata's
`tools/mtp_fetch.py` leaves them: one raw file per tensor and `mtp-manifest.json`.
"""

import json
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn
import numpy as np
from mlx.utils import tree_flatten
from mlx_lm.models import qwen4_exp
from mlx_lm.models.base import create_attention_mask
from mlx_lm.models.switch_layers import QuantizedSwitchLinear

# norms the checkpoint stores zero-centred: the reference scales by 1 + weight
_ADD_ONE = ("q_layernorm.weight", "k_layernorm.weight", "q_norm.weight", "k_norm.weight", "hc_norm.weight",
            "pre_fc_norm_embedding.weight", "pre_fc_norm_hidden.weight")
_FULL_ATTENTION_LAYER = 3  # any index the port builds as a full-attention block without the n-gram input


class Mtp(nn.Module):
    def __init__(self, args: qwen4_exp.TextArgs):
        super().__init__()
        d, hc = args.hidden_size, args.hc_count
        self.d, self.hc = d, hc
        self.pre_fc_norm_embedding = qwen4_exp.RMSNorm(d, eps=args.rms_norm_eps)
        self.pre_fc_norm_hidden = qwen4_exp.RMSNorm(hc * d, group_size=d, eps=args.rms_norm_eps)
        self.fc_embedding = nn.Linear(d, d, bias=False)
        self.fc_hidden = nn.Linear(d, d, bias=False)
        self.layers = [qwen4_exp.DecoderLayer(args, _FULL_ATTENTION_LAYER)]
        self.hyper_connection_mixer = qwen4_exp.GatedResidual(args, use_combine=False)

    def make_cache(self):
        return qwen4_exp._AttnCache()

    def __call__(self, h: mx.array, emb: mx.array, rope, cache):
        """h (1, S, hc * d): the residual at each cell's position; emb (1, S, d): the embedding of the token
        after it. Returns (the head's hidden state, to go through the trunk's output layer; its residual)."""
        S = h.shape[1]
        e = self.fc_embedding(self.pre_fc_norm_embedding(emb))
        r = self.fc_hidden(self.pre_fc_norm_hidden(h).reshape(1, S, self.hc, self.d)) + e[:, :, None, :]
        r = r.reshape(1, S, -1)
        r = self.layers[0](r, rope, create_attention_mask(r, cache), None, cache, cache.indexer, None, None)
        return self.hyper_connection_mixer(r), r


def _bf16(raw: np.ndarray) -> np.ndarray:
    return (raw.astype(np.uint32) << 16).view(np.float32)


def _quantized_experts(raw: np.ndarray, bits: int, step: int = 16) -> dict:
    """A stack of experts (experts, rows, cols) of 16-bit floats, quantized a few experts at a time: the whole
    stack as floats is 3 GB, three times that on the way, and the head has three of them."""
    parts = [mx.quantize(mx.array(_bf16(raw[i : i + step])), group_size=64, bits=bits) for i in range(0, len(raw), step)]
    for part in parts:
        mx.eval(part)
    return dict(zip(("weight", "scales", "biases"), (mx.concatenate(column) for column in zip(*parts))))


def load_mtp(folder: str, args: qwen4_exp.TextArgs, dtype: mx.Dtype = mx.float16, expert_bits: int = 4) -> Mtp:
    folder = Path(folder)
    mtp = Mtp(args)
    glu = mtp.layers[0].mlp.switch_mlp
    for name in ("gate_proj", "up_proj", "down_proj"):  # 5 GB of experts at 16 bits; a few bits are plenty for a guess
        setattr(glu, name, QuantizedSwitchLinear(64, 1, 1, bias=False, group_size=64, bits=expert_bits))
    weights = {}
    for entry in json.loads((folder / "mtp-manifest.json").read_text()):
        if entry["dtype"] != "BF16":
            raise ValueError(f"{entry['name']} is {entry['dtype']}; the checkpoint's head is BF16")
        raw = np.memmap(folder / entry["file"], dtype="<u2", mode="r", shape=tuple(entry["shape"]))
        name = entry["name"].removeprefix("mtp.")
        stacks = {}
        if name.endswith("experts.gate_up_proj"):
            mid = raw.shape[1] // 2
            stacks = {"gate_proj": raw[:, :mid], "up_proj": raw[:, mid:]}
        elif name.endswith("experts.down_proj"):
            stacks = {"down_proj": raw}
        else:
            w = _bf16(np.asarray(raw))
            weights[name] = mx.array(w + 1 if name.endswith(_ADD_ONE) else w)
        for part, stack in stacks.items():
            for key, value in _quantized_experts(stack, expert_bits).items():
                setattr(getattr(glu, part), key, value)
            mx.eval(getattr(glu, part).parameters())
            mx.clear_cache()
    rest = {n for n, _ in tree_flatten(mtp.parameters()) if ".switch_mlp." not in n}
    if rest != set(weights):
        raise ValueError(f"the draft head's files do not match its layers: {sorted(rest ^ set(weights))[:5]}")
    mtp.load_weights(list(weights.items()), strict=False)
    mtp.set_dtype(dtype)
    mx.eval(mtp.parameters())
    del weights
    mx.clear_cache()  # what loading left in MLX's pool made the first long prompt four times slower
    mtp.eval()
    return mtp
