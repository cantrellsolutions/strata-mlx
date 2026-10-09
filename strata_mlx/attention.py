"""Sparse attention whose cost does not grow with the conversation.

The model's attention layers look at 2,048 cells an indexer chooses (512 blocks of four) and at the few cells of
the block the query itself is in. The port this engine builds on computes that as attention over every cached
cell under a mask, and pools the indexer's keys of the whole conversation again for every pass. With 128K tokens
cached that made a pass of four tokens seven times slower than at 1K (MLX's attention kernel spreads the two
key heads over the 24 query heads once there are several queries: 1.6 GB a layer), a single token twice as slow,
and a chunk of prompt reach 16 GB above what was held.

Here:

  * the pooled key of a block is computed once, when the block is complete, and kept in the cache;
  * a pass gathers the cells its queries chose and attends to those, 2,052 a query however long the
    conversation is (0.35 ms a layer for one query at 4K cells and at 128K; through the mask 0.36 and 0.57);
  * a chunk of prompt, hundreds of queries, does so once the conversation is long enough for gathering to be
    the faster way, and until then attends through the mask as before. Alone, one layer's attention for 2,048
    queries is equal the two ways at 27K cells (110-140 ms gathered, at any length); inside a real read, where
    the mask's way asks for new gigabytes at every layer, they are equal between 8K and 18K. Gathering copies
    8.6 GB of keys and values a layer; the queries of a chunk do not choose the same blocks (256 of them choose
    two thirds of all there are at 32K), so there is no smaller set to gather once.

The choice of cells is the port's, step for step; so is the result, up to the order of the additions.
"""

import math

import mlx.core as mx
from mlx_lm.models import qwen4_exp
from mlx_lm.models.base import scaled_dot_product_attention

WINDOW = 8  # up to this many queries gather at any length
GATHER_FROM = 12_000  # cached cells from which more queries than a window gather too
GATHER_BYTES = 1 << 29  # of gathered keys and values at a time


class IndexerCache(qwen4_exp._IndexerCache):
    """The indexer's keys, one per token, in a buffer that grows in steps, and the pooled key of every whole block."""

    step = 1024

    def __init__(self):
        self.keys = None  # (1, capacity, head_dim); the first `offset` rows are real
        self.offset = 0
        self.pooled = None  # (1, blocks, head_dim), normed and rotated
        self.pooled_tokens = 0  # how many tokens the pooled keys still stand for

    def append(self, k: mx.array):
        n = k.shape[1]
        if self.keys is None or self.offset + n > self.keys.shape[1]:
            room = mx.zeros((1, -(-n // self.step) * self.step, k.shape[2]), dtype=k.dtype)
            self.keys = room if self.keys is None else mx.concatenate([self.keys[:, : self.offset], room], axis=1)
        self.keys[:, self.offset : self.offset + n] = k
        self.offset += n

    def blocks(self, indexer, rope) -> mx.array:
        """The pooled keys of every whole block, pooling only the blocks that are new."""
        ratio = indexer.compress_ratio
        n, have = self.offset // ratio, self.pooled_tokens // ratio
        if have < n:
            raw = self.keys[:, have * ratio : n * ratio].reshape(1, n - have, ratio, -1)
            new = indexer.k_layernorm(raw.astype(mx.float32).mean(axis=2).astype(raw.dtype))
            cos, sin = rope((mx.arange(have, n) * ratio)[None, :])
            new = qwen4_exp._rope_partial(new, cos, sin)
            self.pooled = new if have == 0 else mx.concatenate([self.pooled[:, :have], new], axis=1)
            self.pooled_tokens = n * ratio
        return self.pooled[:, :n]

    def trim(self, size: int):
        self.offset = min(self.offset, size)
        self.pooled_tokens = min(self.pooled_tokens, self.offset)

    @property
    def state(self):
        return mx.array([]) if self.keys is None else self.keys[:, : self.offset]

    @state.setter
    def state(self, v):
        self.keys, self.offset = (v, v.shape[1]) if v.size > 0 else (None, 0)
        self.pooled, self.pooled_tokens = None, 0


class AttnCache(qwen4_exp._AttnCache):
    def __init__(self):
        super().__init__()
        self.indexer = IndexerCache()


def choose(indexer, x: mx.array, rope, cache: IndexerCache, offset: int):
    """Add the queries' own keys to the cache and choose each query's blocks.

    None while every cached cell fits in the budget. Else (top, picked, whole): for each of the S queries the
    blocks with the highest scores (S, k), which of those lie wholly in the query's past (S, k), and how many
    blocks do (S,).
    """
    S = x.shape[1]
    qk = indexer.index_qk_proj(x)
    split = indexer.n_heads * indexer.head_dim
    q = qk[..., :split].reshape(1, S, indexer.n_heads, indexer.head_dim)
    cache.append(qk[..., split:])
    kv_len = cache.offset
    if kv_len <= indexer.token_budget:
        return None
    pooled = cache.blocks(indexer, rope)
    n_blocks = pooled.shape[1]
    q_pos = qwen4_exp._positions(offset, S)
    cos, sin = rope(q_pos)
    q = qwen4_exp._rope_partial(indexer.q_layernorm(q), cos[:, :, None, :], sin[:, :, None, :])
    scores = mx.einsum("bshd,bnd->bsnh", q.astype(mx.float32), pooled.astype(mx.float32))
    scores = mx.maximum(scores, 0).sum(axis=-1) / math.sqrt(indexer.head_dim)
    whole = (q_pos + 1) // indexer.compress_ratio
    visible = mx.arange(n_blocks)[None, None, :] < whole[..., None]
    scores = mx.where(visible, scores, -mx.inf)
    k = min(indexer.block_topk, n_blocks)
    top = mx.argpartition(-scores, k - 1, axis=-1)[..., :k]
    return top[0], mx.take_along_axis(visible, top, axis=-1)[0], whole[0]


def _mask(chosen, ratio: int, S: int, kv_len: int) -> mx.array:
    """(1, 1, S, kv_len): the cells each query attends to. Every one is in its past, so no causal mask is needed."""
    top, picked, whole = chosen
    n_blocks = kv_len // ratio
    blocks = mx.put_along_axis(mx.zeros((S, n_blocks + 1), dtype=mx.bool_), mx.where(picked, top, n_blocks),
                               mx.array(True), axis=-1)[:, :n_blocks]
    keep = mx.repeat(blocks, ratio, axis=-1)
    if kv_len > n_blocks * ratio:
        keep = mx.concatenate([keep, mx.zeros((S, kv_len - n_blocks * ratio), dtype=mx.bool_)], axis=-1)
    cells = mx.arange(kv_len)[None, :]
    own = (cells >= (whole * ratio)[:, None]) & (cells <= mx.arange(kv_len - S, kv_len)[:, None])
    return (keep | own)[None, None]


def _cells(chosen, ratio: int, S: int, kv_len: int):
    """The same cells as a list for each query, (S, L), and which entries of it are real, (S, L)."""
    top, picked, whole = chosen
    within = mx.arange(ratio)
    chosen_cells = (top[..., None] * ratio + within).reshape(S, -1)
    chosen_real = mx.repeat(picked, ratio, axis=-1)
    own = (whole * ratio)[:, None] + within  # the block the query is in, up to the query
    own_real = own <= mx.arange(kv_len - S, kv_len)[:, None]
    return (mx.concatenate([chosen_cells, mx.minimum(own, kv_len - 1)], axis=-1),
            mx.concatenate([chosen_real, own_real], axis=-1))


def _gathered(q: mx.array, k: mx.array, v: mx.array, cells: mx.array, real: mx.array, scale: float, cache):
    """q (1, heads, S, d) attending to its own cells of k, v (1, kv heads, kv_len, d) -> (1, S, heads * d).

    Each query is a row of its own with its cells beside it, the query heads that share a key head as that
    head's queries: nothing is spread over heads, nothing is as long as the conversation.
    """
    _, H, S, D = q.shape
    G, L = k.shape[1], cells.shape[1]
    step = max(1, GATHER_BYTES // (2 * G * L * D * k.dtype.size))
    out = []
    for a in range(0, S, step):
        b = min(S, a + step)
        flat = cells[a:b].reshape(-1)
        # from the cache as it is: k[0] would copy all of it first, 2 ms a layer at 128K cells
        ks = mx.take(k, flat, axis=2).reshape(G, b - a, L, D).transpose(1, 0, 2, 3)
        vs = mx.take(v, flat, axis=2).reshape(G, b - a, L, D).transpose(1, 0, 2, 3)
        qs = q[0, :, a:b].transpose(1, 0, 2).reshape(b - a, G, H // G, D)
        o = scaled_dot_product_attention(qs, ks, vs, cache=cache, scale=scale, mask=real[a:b, None, None, :])
        out.append(o.reshape(1, b - a, H * D))
        if step < S:
            mx.eval(out[-1])  # one part's keys and values at a time
    return out[0] if len(out) == 1 else mx.concatenate(out, axis=1)


class Attention(qwen4_exp.Attention):
    def __call__(self, x, rope, mask, cache, idx_cache) -> mx.array:
        B, S, _ = x.shape
        if B != 1 or not isinstance(idx_cache, IndexerCache):
            raise ValueError("one conversation, with this module's cache")
        offset = cache.offset
        chosen = choose(self.indexer, x, rope, idx_cache, offset)

        q, gate = mx.split(self.q_proj(x).reshape(1, S, self.n_heads, -1), 2, axis=-1)
        q = self.q_norm(q).transpose(0, 2, 1, 3)
        k = self.k_norm(self.k_proj(x).reshape(1, S, self.n_kv_heads, -1)).transpose(0, 2, 1, 3)
        v = self.v_proj(x).reshape(1, S, self.n_kv_heads, -1).transpose(0, 2, 1, 3)
        cos, sin = rope(qwen4_exp._positions(offset, S))
        q, k = qwen4_exp._rope_partial(q, cos[:, None], sin[:, None]), qwen4_exp._rope_partial(k, cos[:, None], sin[:, None])
        k, v = cache.update_and_fetch(k, v)

        kv_len, ratio = k.shape[2], self.indexer.compress_ratio
        if chosen is not None and (S <= WINDOW or kv_len >= GATHER_FROM):
            out = _gathered(q, k, v, *_cells(chosen, ratio, S, kv_len), self.scale, cache)
        else:
            if chosen is not None:
                mask = _mask(chosen, ratio, S, kv_len)
            out = scaled_dot_product_attention(q, k, v, cache=cache, scale=self.scale, mask=mask)
            out = out.transpose(0, 2, 1, 3).reshape(1, S, -1)
        return self.o_proj(out * mx.sigmoid(gate.reshape(1, S, -1)))


def install(layers) -> None:
    """Make the attention layers among `layers` this module's."""
    for layer in layers:
        if getattr(layer, "self_attn", None) is not None and type(layer.self_attn) is qwen4_exp.Attention:
            layer.self_attn.__class__ = Attention
