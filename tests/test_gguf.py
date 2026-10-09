"""Every GGUF type the loader takes must come out with the weights gguf-py's own dequantizer gives."""

import gguf
import mlx.core as mx
import numpy as np
import pytest

from strata_mlx.gguf import TYPES, Tensor, pack, to_affine, to_float

ROWS, COLS = 6, 2560
# where each block keeps its 16-bit float scales
SCALE_BYTES = {"Q2_0": [0], "Q4_0": [0], "Q5_0": [0], "Q8_0": [0], "IQ4_NL": [0], "IQ4_XS": [0],
               "Q4_K": [0, 2], "Q5_K": [0, 2], "Q3_K": [108], "Q6_K": [208],
               "IQ3_S": [0], "IQ3_XXS": [0], "IQ2_S": [0], "IQ2_XS": [0], "IQ2_XXS": [0], "IQ1_M": []}


def random_tensor(kind: str, seed: int = 0) -> Tensor:
    _, per_block, block_bytes = next(v for v in TYPES.values() if v[0] == kind)
    rng = np.random.default_rng(seed)
    n = ROWS * COLS // per_block
    blocks = rng.integers(0, 256, (n, block_bytes), dtype=np.uint8)
    for at in SCALE_BYTES[kind]:
        blocks[:, at : at + 2] = rng.uniform(-0.05, 0.05, (n, 1)).astype("<f2").view(np.uint8)
    if kind == "IQ1_M":  # its scale is spread over the top four bits of the four words at byte 48
        d = rng.uniform(-0.05, 0.05, n).astype("<f2").view("<u2")
        words = blocks[:, 48:56].copy().view("<u2")
        for i in range(4):
            words[:, i] = (words[:, i] & 0x0FFF) | (((d >> (4 * i)) & 15) << 12)
        blocks[:, 48:56] = words.view(np.uint8)
    return Tensor(kind, (ROWS, COLS), kind, blocks.reshape(-1))


def reference(t: Tensor) -> np.ndarray:
    if t.type == "Q2_0":  # gguf-py has no Q2_0; ggml's dequantize_row_q2_0 is (code - 1) * d
        b = t.data.reshape(-1, 18)
        d = b[:, :2].copy().view("<f2").astype(np.float32)
        code = np.stack([(b[:, 2:] >> s) & 3 for s in (0, 2, 4, 6)], axis=2).reshape(-1, 64)
        return ((code.astype(np.float32) - 1) * d).reshape(t.shape)
    qtype = getattr(gguf.GGMLQuantizationType, t.type)
    return gguf.quants.dequantize(t.data.reshape(ROWS, -1), qtype).reshape(t.shape)


def close(got: np.ndarray, want: np.ndarray):
    assert got.shape == want.shape
    assert np.abs(got - want).max() <= 1e-6 * np.abs(want).max()


@pytest.mark.parametrize("kind", ["Q2_0", "Q4_0", "Q5_0", "Q8_0", "Q4_K", "Q5_K", "Q3_K", "IQ4_NL", "IQ4_XS",
                                  "IQ3_S"])
def test_affine_form_reproduces_the_weights(kind):
    t = random_tensor(kind)
    want = reference(t)
    a = to_affine(t)
    assert a.codes.max() < 1 << a.bits
    in_numpy = a.codes.astype(np.float32) * np.repeat(a.scales, a.group, -1) + np.repeat(a.biases, a.group, -1)
    close(in_numpy, want)
    in_mlx = mx.dequantize(mx.array(pack(a.codes, a.bits)), mx.array(a.scales), mx.array(a.biases),
                           group_size=a.group, bits=a.bits)
    close(np.array(in_mlx), want)


def test_q6_k_as_float():
    t = random_tensor("Q6_K")
    close(to_float(t), reference(t))


def test_bf16_is_exact():
    bits = np.array([0x3F80, 0xBF00, 0x37FB, 0x4480], dtype="<u2")  # 1, -0.5, about 3e-5, 1024
    want = (bits.astype(np.uint32) << 16).view(np.float32)
    assert want[0] == 1.0 and want[3] == 1024.0
    np.testing.assert_array_equal(to_float(Tensor("t", (4,), "BF16", bits.view(np.uint8))), want)



def test_the_ngram_rows_worked_out_on_the_cpu_are_the_ports():
    """Tokens of every kind: ends of text in the context and in the window, ids whose products overflow."""
    from mlx_lm.models import qwen4_exp

    from strata_mlx.load import HostNgram

    args = qwen4_exp.TextArgs(vocab_size=250000, ngram_vocab_size_base=100003, split_ngram_parts=4, ple_embed_dim=64,
                              eos_token_id=5)
    emb = qwen4_exp.NGramEmbedding(args, args.ple_embed_dim, ple_layer_index=1)
    seen = []
    emb.ngram_embedding = lambda gid: (seen.append(gid), mx.zeros((*gid.shape, 4)))[1]
    host = HostNgram(emb, None)
    rng = np.random.default_rng(0)
    for T in (1, 2, 8, 40):
        ids = rng.integers(0, 250000, (1, T))
        context = rng.integers(0, 250000, (1, emb.ngram_size - 1))
        ids[0, rng.integers(0, T)] = 5
        if T > 2:
            context[0, -1] = 5
        emb(mx.array(ids), mx.array(context))
        want = np.array(seen[-1])
        got = host.rows(ids, context)
        assert got.shape == want.shape and (got == want).all()
        assert (got >= 0).all()
