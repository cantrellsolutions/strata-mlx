"""Reads a GGUF file and hands each tensor to MLX without changing a weight.

MLX keeps a quantized weight as integer codes with one scale and one bias per group of 32 or 64 weights:
`w = code * scale + bias`. Most GGUF types are that already, or become that at 8 bits:

    Q2_0                 2 bits, groups of 64      w = d * (code - 1)
    Q4_0, Q5_0, Q8_0     4 / 5 / 8 bits            one scale per block of 32
    Q4_K, Q5_K           4 / 5 bits                a scale and a minimum per sub-block of 32
    Q3_K                 8 bits                    the code is (sub-block scale * 3-bit value), an integer
    IQ4_NL, IQ4_XS       8 bits                    the code is the codebook entry, an integer
    IQ3_S                4 bits                    its values are the odd numbers from -15 to 15

Q6_K has a scale per 16 weights and products too wide for 8 bits, so it is returned as float32, as are the
float types. Rows are quantized one at a time in GGUF, so a tensor's leading axes can be reordered freely.
"""

import mmap
import re
import struct
from dataclasses import dataclass

import numpy as np

# type id -> (name, weights per block, bytes per block)
TYPES = {
    0: ("F32", 1, 4), 1: ("F16", 1, 2), 30: ("BF16", 1, 2),
    2: ("Q4_0", 32, 18), 6: ("Q5_0", 32, 22), 8: ("Q8_0", 32, 34),
    11: ("Q3_K", 256, 110), 12: ("Q4_K", 256, 144), 13: ("Q5_K", 256, 176), 14: ("Q6_K", 256, 210),
    20: ("IQ4_NL", 32, 18), 23: ("IQ4_XS", 256, 136), 42: ("Q2_0", 64, 18),
    # codebook types: IQ3_S has an affine form, the others are computed on their blocks (blocks.py)
    16: ("IQ2_XXS", 256, 66), 17: ("IQ2_XS", 256, 74), 18: ("IQ3_XXS", 256, 98), 21: ("IQ3_S", 256, 110),
    22: ("IQ2_S", 256, 82), 29: ("IQ1_M", 256, 56),
}
KVALUES_IQ4NL = np.array([-127, -104, -83, -65, -49, -35, -22, -10, 1, 13, 25, 38, 53, 69, 89, 113], dtype=np.int16)
_SCALAR = {0: "B", 1: "b", 2: "H", 3: "h", 4: "I", 5: "i", 6: "f", 7: "?", 10: "Q", 11: "q", 12: "d"}


@dataclass
class Tensor:
    name: str
    shape: tuple  # numpy order: the last axis is the one a row runs along
    type: str
    data: np.ndarray  # the raw bytes, a view into the file
    path: str = ""  # the file, and where in it the bytes start
    offset: int = 0

    @property
    def quantized(self) -> bool:
        return self.type not in ("F32", "F16", "BF16")


@dataclass
class Affine:
    """A weight as MLX holds it: `codes * scales + biases`, `group` weights to each scale."""

    codes: np.ndarray  # uint8, the tensor's shape
    scales: np.ndarray  # float32, the last axis divided by `group`
    biases: np.ndarray
    bits: int
    group: int


class GGUF:
    def __init__(self, path: str):
        self.path = path
        self._f = open(path, "rb")
        self._map = mmap.mmap(self._f.fileno(), 0, access=mmap.ACCESS_READ)
        buf = self._map
        self._pos = 0

        def rd(fmt):
            out = struct.unpack_from("<" + fmt, buf, self._pos)
            self._pos += struct.calcsize("<" + fmt)
            return out if len(out) > 1 else out[0]

        def string():
            n = rd("Q")
            s = bytes(buf[self._pos : self._pos + n]).decode("utf-8", "replace")
            self._pos += n
            return s

        def value(t):
            if t == 8:
                return string()
            if t == 9:
                et, n = rd("IQ")
                return [value(et) for _ in range(n)]
            return rd(_SCALAR[t])

        magic = bytes(buf[:4])
        self._pos = 4
        if magic != b"GGUF":
            raise ValueError(f"{path} is not a GGUF file")
        version, n_tensors, n_kv = rd("IQQ")
        self.meta = {}
        for _ in range(n_kv):
            key = string()
            self.meta[key] = value(rd("I"))
        infos = []
        for _ in range(n_tensors):
            name = string()
            n_dims = rd("I")
            dims = struct.unpack_from(f"<{n_dims}Q", buf, self._pos)
            self._pos += 8 * n_dims
            infos.append((name, tuple(reversed(dims)), *rd("IQ")))
        align = self.meta.get("general.alignment", 32)
        start = (self._pos + align - 1) // align * align
        raw = np.frombuffer(buf, dtype=np.uint8)
        self.tensors = {}
        for name, shape, type_id, offset in infos:
            if type_id not in TYPES:
                raise ValueError(f"{name}: GGUF tensor type {type_id} is not supported")
            kind, per_block, block_bytes = TYPES[type_id]
            n_bytes = int(np.prod(shape)) // per_block * block_bytes
            self.tensors[name] = Tensor(name, shape, kind, raw[start + offset : start + offset + n_bytes],
                                        path, start + offset)

    def __getitem__(self, name: str) -> Tensor:
        return self.tensors[name]

    def __contains__(self, name: str) -> bool:
        return name in self.tensors


_SPLIT = re.compile(r"-(\d{5})-of-(\d{5})\.gguf$")


def shards(path: str) -> list:
    """A model's files: the one named and, where its name says it is one of several, the others beside it."""
    m = _SPLIT.search(path)
    count = int(m.group(2)) if m else 1
    return [GGUF(path)] + [GGUF(_SPLIT.sub(f"-{i:05d}-of-{m.group(2)}.gguf", path)) for i in range(2, count + 1)]


def to_float(t: Tensor) -> np.ndarray:
    """A float tensor, or a Q6_K one, as float32."""
    if t.type == "F32":
        return t.data.view("<f4").reshape(t.shape)
    if t.type == "F16":
        return t.data.view("<f2").astype(np.float32).reshape(t.shape)
    if t.type == "BF16":
        return (t.data.view("<u2").astype(np.uint32) << 16).view(np.float32).reshape(t.shape)
    if t.type == "Q6_K":
        codes, scales, d = q6k_parts(t)
        scale = (d.astype(np.float32)[:, :, None] * scales.astype(np.float32).reshape(len(d), -1, 16)).reshape(scales.shape)
        return (codes.astype(np.float32).reshape(*scales.shape, 16) * scale[:, :, None]).reshape(t.shape)
    raise ValueError(f"{t.name}: {t.type} has an exact quantized form, use to_affine")


def q6k_parts(t: Tensor):
    """A Q6_K tensor of rows as its three parts: a weight is code * scale * d.

    codes   int8 (rows, cols)          the block's six bits a weight, less 32
    scales  int8 (rows, cols / 16)
    d       float16 (rows, cols / 256)
    """
    rows = t.shape[0]
    b = t.data.reshape(-1, 210)
    ql, qh = b[:, :128].reshape(-1, 2, 64), b[:, 128:192].reshape(-1, 2, 32)
    # each half of the block: four runs of 32, their high two bits in the four bit pairs of qh
    hi = np.stack([(qh >> s) & 3 for s in (0, 2, 4, 6)], axis=2)  # (n, 2, 4, 32)
    lo = np.stack([ql[..., :32] & 15, ql[..., 32:] & 15, ql[..., :32] >> 4, ql[..., 32:] >> 4], axis=2)
    codes = ((lo | (hi << 4)).astype(np.int16) - 32).astype(np.int8)
    return (codes.reshape(rows, -1), np.ascontiguousarray(b[:, 192:208]).view(np.int8).reshape(rows, -1),
            np.ascontiguousarray(b[:, 208:210]).view("<f2").reshape(rows, -1))


def _nibbles(qs: np.ndarray, run: int = 16) -> np.ndarray:
    """Runs of `run` bytes -> twice as many values: the low nibbles, then the high ones."""
    qs = qs.reshape(-1, run)
    return np.concatenate([qs & 15, qs >> 4], axis=1)


def _k4_scales(packed: np.ndarray):
    """The eight 6-bit (scale, minimum) pairs of a Q4_K / Q5_K block, from their 12 bytes."""
    p = packed.astype(np.uint8)
    sc = np.concatenate([p[:, :4] & 63, (p[:, 8:12] & 15) | ((p[:, :4] >> 6) << 4)], axis=1)
    mn = np.concatenate([p[:, 4:8] & 63, (p[:, 8:12] >> 4) | ((p[:, 4:8] >> 6) << 4)], axis=1)
    return sc.astype(np.float32), mn.astype(np.float32)


def to_affine(t: Tensor) -> Affine:
    """The tensor as codes, scales and biases that reproduce every weight."""
    f32 = np.float32
    kind = t.type
    _, per_block, block_bytes = next(v for v in TYPES.values() if v[0] == kind)
    b = t.data.reshape(-1, block_bytes)
    n = b.shape[0]
    group = 32

    if kind == "Q2_0":
        d = b[:, :2].copy().view("<f2").astype(f32)
        codes = np.stack([(b[:, 2:] >> s) & 3 for s in (0, 2, 4, 6)], axis=2)
        bits, group, scales, biases = 2, 64, d, -d
    elif kind == "Q4_0":
        d = b[:, :2].copy().view("<f2").astype(f32)
        codes, bits, scales, biases = _nibbles(b[:, 2:]), 4, d, -8 * d
    elif kind == "Q5_0":
        d = b[:, :2].copy().view("<f2").astype(f32)
        high = (b[:, 2:6].copy().view("<u4") >> np.arange(32, dtype=np.uint32)) & 1
        codes = _nibbles(b[:, 6:]) | (high.astype(np.uint8) << 4)
        bits, scales, biases = 5, d, -16 * d
    elif kind == "Q8_0":
        d = b[:, :2].copy().view("<f2").astype(f32)
        codes = (b[:, 2:].view(np.int8).astype(np.int16) + 128).astype(np.uint8)
        bits, scales, biases = 8, d, -128 * d
    elif kind in ("Q4_K", "Q5_K"):
        d = b[:, 0:2].copy().view("<f2").astype(f32)
        dmin = b[:, 2:4].copy().view("<f2").astype(f32)
        sc, mn = _k4_scales(b[:, 4:16])
        scales, biases = d * sc, -(dmin * mn)
        if kind == "Q4_K":
            codes, bits = _nibbles(b[:, 16:], 32), 4
        else:
            qh = b[:, 16:48]
            high = np.stack([(qh >> s) & 1 for s in range(8)], axis=1)  # (n, 8 sub-blocks, 32)
            codes = _nibbles(b[:, 48:], 32).reshape(n, 8, 32) | (high << 4)
            bits = 5
    elif kind == "Q3_K":
        hmask, qs = b[:, :32], b[:, 32:96].reshape(n, 2, 32)
        d = b[:, 108:110].copy().view("<f2").astype(f32)
        a = b[:, 96:108].copy().view("<u4")
        tmp = a[:, 2]
        k1, k2 = np.uint32(0x03030303), np.uint32(0x0F0F0F0F)
        sub = np.stack([
            (a[:, 0] & k2) | ((tmp & k1) << 4),
            (a[:, 1] & k2) | (((tmp >> 2) & k1) << 4),
            ((a[:, 0] >> 4) & k2) | (((tmp >> 4) & k1) << 4),
            ((a[:, 1] >> 4) & k2) | (((tmp >> 6) & k1) << 4),
        ], axis=1).astype("<u4").view(np.uint8).astype(np.int16) - 32  # (n, 16)
        # run r of 32 weights (r = 4 * half + j): two bits from qs[half] at shift 2j, a sign bit r of hmask
        two = np.stack([(qs >> s) & 3 for s in (0, 2, 4, 6)], axis=2).reshape(n, 8, 32)
        high = np.stack([(hmask >> r) & 1 for r in range(8)], axis=1)
        value = two.astype(np.int16) - np.where(high == 1, 0, 4)  # -4 .. 3
        product = value.reshape(n, 16, 16) * sub[:, :, None]  # -124 .. 128
        codes = (product + 124).astype(np.uint8)
        d8 = np.repeat(d, 8, axis=1)
        bits, scales, biases = 8, d8, -124 * d8
    elif kind == "IQ4_NL":
        d = b[:, :2].copy().view("<f2").astype(f32)
        codes = (KVALUES_IQ4NL[_nibbles(b[:, 2:])] + 127).astype(np.uint8)
        bits, scales, biases = 8, d, -127 * d
    elif kind == "IQ4_XS":
        d = b[:, 0:2].copy().view("<f2").astype(f32)
        high = b[:, 2:4].copy().view("<u2").astype(np.uint32)
        low = b[:, 4:8]
        ib = np.arange(8)
        ls = ((low[:, ib // 2] >> (4 * (ib % 2))) & 15) | (((high >> (2 * ib)) & 3) << 4)
        dl = d * (ls.astype(f32) - 32)
        codes = (KVALUES_IQ4NL[_nibbles(b[:, 8:])] + 127).astype(np.uint8)
        bits, scales, biases = 8, dl, -127 * dl
    elif kind == "IQ3_S":
        from gguf import quants

        quants.IQ3_S.init_grid()
        grid = quants.IQ3_S.grid.reshape(512, 4).astype(np.int8)
        d = b[:, 0:2].copy().view("<f2").astype(f32)
        sub = b[:, 106:110]
        db = d * (1 + 2 * np.stack([sub & 15, sub >> 4], axis=2).reshape(n, 8).astype(f32))
        # 64 groups of four weights: eight index bits in qs, the ninth in qh; a sign bit for every weight
        index = b[:, 2:66].astype(np.uint16) | (np.unpackbits(b[:, 66:74], axis=1, bitorder="little").astype(np.uint16) << 8)
        negative = np.unpackbits(b[:, 74:106], axis=1, bitorder="little").astype(np.int8)
        value = grid[index].reshape(n, 256) * (1 - 2 * negative)  # odd, -15 .. 15
        codes = ((value + 15) // 2).astype(np.uint8)
        bits, scales, biases = 4, 2 * db, -15 * db
    else:
        raise ValueError(f"{t.name}: {kind} has no exact affine form, use to_float")

    lead, last = t.shape[:-1], t.shape[-1]
    return Affine(
        np.ascontiguousarray(codes).reshape(*lead, last),
        np.ascontiguousarray(scales, dtype=f32).reshape(*lead, last // group),
        np.ascontiguousarray(biases, dtype=f32).reshape(*lead, last // group),
        bits,
        group,
    )


def pack(codes: np.ndarray, bits: int) -> np.ndarray:
    """Codes -> MLX's packed words: each row a little-endian bit stream, cut into uint32."""
    lead, last = codes.shape[:-1], codes.shape[-1]
    c = codes.reshape(-1, last)
    if bits == 8:
        packed = c
    elif bits == 4:
        packed = c[:, 0::2] | (c[:, 1::2] << 4)
    else:
        # a few thousand rows at a time: the bit stream is eight times the size of the codes
        step = max(1, (1 << 26) // last)
        packed = np.empty((c.shape[0], last * bits // 8), dtype=np.uint8)
        for i in range(0, c.shape[0], step):
            stream = np.unpackbits(c[i : i + step, :, None], axis=2, bitorder="little")[:, :, :bits]
            packed[i : i + step] = np.packbits(stream.reshape(stream.shape[0], -1), axis=1, bitorder="little")
    return np.ascontiguousarray(packed).view("<u4").reshape(*lead, last * bits // 32)
