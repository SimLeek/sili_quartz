"""One-token decode for Qwen3.5 (GameScaling) as a single CUDA graph.

A decode step through transformers launches about 1,200-1,500 small kernels
(about 17 us each on the P40, 2060), so it is bound by launch overhead, not
by arithmetic. This runs the same step on preallocated buffers and captures
it once as a CUDA graph:

- linear-attention layers: their conv state and recurrent state are fixed
  size, updated in place;
- full-attention layers: a KV buffer of max_len positions, and an additive
  mask that opens one more position per step; one graph per length bucket
  (1k, 2k, 4k, ... max_len) so attention reads only the part in use;
- position: text tokens have the same position on all three M-RoPE axes,
  which makes M-RoPE plain RoPE at (index + rope delta).

Use: load(hf_cache, length, rope_delta) after a prefill, step(token) per
token (returns logits), store(hf_cache) to hand the state back so the next
prefill can continue from it.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F


class GraphDecoder:
    def __init__(self, model, max_len: int = 16384, use_graph: bool = True, vocab_ids=None):
        """vocab_ids: compute logits only for these token ids (step() then
        returns logits over them, in that order). The output head is a third
        of the bytes a step reads, so this is the cheapest speedup when the
        output is known to stay inside a small vocabulary."""
        lm = model.model.language_model
        cfg = model.config.text_config
        self.model, self.lm, self.cfg = model, lm, cfg
        p = next(lm.parameters())
        self.dev, self.dtype = p.device, p.dtype
        self.max_len = max_len
        self.use_graph = use_graph
        self.eps = cfg.rms_norm_eps
        self.types = list(cfg.layer_types[: cfg.num_hidden_layers])
        self.hd = cfg.head_dim
        self.n_q, self.n_kv = cfg.num_attention_heads, cfg.num_key_value_heads
        rp = cfg.rope_parameters
        self.rot = int(self.hd * rp.get("partial_rotary_factor", 1.0))
        self.inv_freq = 1.0 / (rp["rope_theta"] ** (torch.arange(0, self.rot, 2, device=self.dev, dtype=torch.float32) / self.rot))
        dev, dt = self.dev, self.dtype
        # static inputs
        self.token = torch.zeros(1, dtype=torch.long, device=dev)
        self.pos = torch.zeros(1, dtype=torch.float32, device=dev)  # index + rope delta
        self.write_idx = torch.zeros(1, dtype=torch.long, device=dev)
        self.mask = torch.full((1, 1, 1, max_len), float("-inf"), device=dev, dtype=torch.float32)
        # static state
        self.conv, self.rec, self.k, self.v = {}, {}, {}, {}
        for i, t in enumerate(self.types):
            layer = lm.layers[i]
            if t == "linear_attention":
                la = layer.linear_attn
                self.conv[i] = torch.zeros(1, la.conv_dim, la.conv_kernel_size, device=dev, dtype=dt)
                self.rec[i] = torch.zeros(1, la.num_v_heads, la.head_k_dim, la.head_v_dim, device=dev, dtype=torch.float32)
            else:
                self.k[i] = torch.zeros(1, self.n_kv, max_len, self.hd, device=dev, dtype=dt)
                self.v[i] = torch.zeros(1, self.n_kv, max_len, self.hd, device=dev, dtype=dt)
        self.vocab_ids = None
        self.head_w = None
        if vocab_ids is not None:
            self.vocab_ids = torch.as_tensor(list(vocab_ids), dtype=torch.long, device=dev)
            self.head_w = model.lm_head.weight[self.vocab_ids].detach().contiguous()
        self.logits = None
        self.graphs: dict[int, tuple] = {}  # window -> (graph, its logits tensor)
        self.pool = None
        self.buckets = sorted({min(max_len, 1024 * 2 ** j) for j in range(16) if 1024 * 2 ** j < max_len} | {max_len})
        self.window = max_len
        self.length = 0

    # -- the step -------------------------------------------------------------------

    def _rms(self, x, w):  # Qwen3_5RMSNorm: weight is zero-centred
        xf = x.float()
        return (xf * torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + self.eps) * (1.0 + w.float())).to(x.dtype)

    def _rope(self, x, cos, sin):
        r = self.rot
        xr, xp = x[..., :r], x[..., r:]
        x1, x2 = xr[..., : r // 2], xr[..., r // 2:]
        rotated = torch.cat((-x2, x1), dim=-1)
        return torch.cat((xr * cos + rotated * sin, xp), dim=-1)

    def _linear(self, i, la, h):
        qkv = la.in_proj_qkv(h)  # (1, conv_dim)
        z = la.in_proj_z(h)
        b, a = la.in_proj_b(h), la.in_proj_a(h)
        conv = self.conv[i]
        conv.copy_(torch.cat([conv[:, :, 1:], qkv[:, :, None].to(conv.dtype)], dim=-1))
        mixed = F.silu((conv * la.conv1d.weight.squeeze(1)[None]).sum(-1))  # (1, conv_dim)
        q, k, v = torch.split(mixed, [la.key_dim, la.key_dim, la.value_dim], dim=-1)
        H, dk, dv = la.num_v_heads, la.head_k_dim, la.head_v_dim
        q = q.reshape(1, la.num_k_heads, dk).float()
        k = k.reshape(1, la.num_k_heads, dk).float()
        v = v.reshape(1, H, dv).float()
        if H // la.num_k_heads > 1:
            q = q.repeat_interleave(H // la.num_k_heads, dim=1)
            k = k.repeat_interleave(H // la.num_k_heads, dim=1)
        q = F.normalize(q, dim=-1, eps=1e-6) / (dk ** 0.5)
        k = F.normalize(k, dim=-1, eps=1e-6)
        beta = b.float().sigmoid()[..., None]  # (1, H, 1)
        g = (-la.A_log.float().exp() * F.softplus(a.float() + la.dt_bias)).exp()  # (1, H)
        S = self.rec[i]
        S.mul_(g[..., None, None])
        kv_mem = (S * k[..., None]).sum(-2)  # (1, H, dv)
        delta = (v - kv_mem) * beta
        S.add_(k[..., None] * delta[..., None, :])
        out = (S * q[..., None]).sum(-2)  # (1, H, dv)
        out = out.to(h.dtype).reshape(-1, dv)
        out = la.norm(out, z.reshape(-1, dv)).reshape(1, -1)
        return la.out_proj(out)

    def _attention(self, i, at, h, cos, sin):
        qg = at.q_proj(h).view(1, self.n_q, self.hd * 2)
        q, gate = qg[..., : self.hd], qg[..., self.hd:].reshape(1, -1)
        q = self._rms(q, at.q_norm.weight)  # (1, n_q, hd)
        k = self._rms(at.k_proj(h).view(1, self.n_kv, self.hd), at.k_norm.weight)
        v = at.v_proj(h).view(1, self.n_kv, self.hd)
        q, k = self._rope(q, cos, sin), self._rope(k, cos, sin)
        self.k[i].index_copy_(2, self.write_idx, k[:, :, None].to(self.dtype))
        self.v[i].index_copy_(2, self.write_idx, v[:, :, None].to(self.dtype))
        rep = self.n_q // self.n_kv
        qh = q.view(1, self.n_kv, rep, self.hd)  # group query heads by their kv head
        W = self.window  # attend over the first W positions only (a length bucket)
        keys, values = self.k[i][:, :, :W], self.v[i][:, :, :W]
        scores = torch.einsum("bgrd,bgld->bgrl", qh.to(self.dtype), keys).float() * (self.hd ** -0.5)
        scores = scores + self.mask[0, 0, :, :W]  # (1, g, rep, W)
        attn = scores.softmax(-1).to(self.dtype)
        out = torch.einsum("bgrl,bgld->bgrd", attn, values).reshape(1, -1).to(h.dtype)
        return at.o_proj(out * torch.sigmoid(gate))

    def _step(self):
        lm = self.lm
        self.mask.index_fill_(3, self.write_idx, 0.0)
        h = lm.embed_tokens(self.token)  # (1, hidden)
        freqs = self.pos[:, None] * self.inv_freq[None]  # (1, rot/2)
        emb = torch.cat((freqs, freqs), dim=-1)
        cos, sin = emb.cos().to(self.dtype)[:, None], emb.sin().to(self.dtype)[:, None]  # (1, 1, rot)
        for i, t in enumerate(self.types):
            layer = lm.layers[i]
            x = self._rms(h, layer.input_layernorm.weight)
            if t == "linear_attention":
                x = self._linear(i, layer.linear_attn, x)
            else:
                x = self._attention(i, layer.self_attn, x, cos, sin)
            h = h + x
            h = h + layer.mlp(self._rms(h, layer.post_attention_layernorm.weight))
        h = self._rms(h, lm.norm.weight)
        if self.head_w is not None:
            return h @ self.head_w.T
        return self.model.lm_head(h)

    # -- state in and out ----------------------------------------------------------------

    @torch.no_grad()
    def load(self, cache, length: int, rope_delta: int) -> None:
        """Copy a transformers cache (after a prefill of `length` tokens) in."""
        if length + 512 > self.max_len:
            raise ValueError(f"context {length} too long for max_len {self.max_len}")
        for i, t in enumerate(self.types):
            layer = cache.layers[i]
            if t == "linear_attention":
                cs = layer.conv_states[0]
                self.conv[i].zero_()
                self.conv[i][..., -cs.shape[-1]:].copy_(cs)
                self.rec[i].copy_(layer.recurrent_states[0])
            else:
                self.k[i][:, :, :length].copy_(layer.keys)
                self.v[i][:, :, :length].copy_(layer.values)
        self.mask.fill_(float("-inf"))
        self.mask[..., :length] = 0.0
        self.length, self.rope_delta = length, int(rope_delta)

    @torch.no_grad()
    def store(self, cache) -> None:
        """Hand the state back to the transformers cache."""
        for i, t in enumerate(self.types):
            layer = cache.layers[i]
            if t == "linear_attention":
                cs = layer.conv_states[0]
                cs.copy_(self.conv[i][..., -cs.shape[-1]:])
                layer.recurrent_states[0].copy_(self.rec[i])
            else:
                layer.keys = self.k[i][:, :, : self.length].clone()
                layer.values = self.v[i][:, :, : self.length].clone()

    @torch.no_grad()
    def step(self, token: int) -> torch.Tensor:
        """Feed one token at the next position; returns logits (1, vocab)."""
        self.token.fill_(token)
        self.write_idx.fill_(self.length)
        self.pos.fill_(float(self.length + self.rope_delta))
        # smallest captured length bucket that holds the new position
        self.window = next(b for b in self.buckets if b > self.length)
        if not self.use_graph:
            self.logits = self._step()
        elif self.window not in self.graphs:
            self._capture()  # ends with one real replay
        else:
            graph, self.logits = self.graphs[self.window]
            graph.replay()
        self.length += 1
        return self.logits

    def _capture(self) -> None:
        # warm up on a side stream (required before capture), then restore the state
        saved = ({i: c.clone() for i, c in self.conv.items()}, {i: r.clone() for i, r in self.rec.items()},
                 self.mask.clone())
        s = torch.cuda.Stream(self.dev)
        s.wait_stream(torch.cuda.current_stream(self.dev))
        with torch.cuda.stream(s):
            for _ in range(2):
                self._step()
        torch.cuda.current_stream(self.dev).wait_stream(s)
        graph = torch.cuda.CUDAGraph()
        if self.pool is None:
            self.pool = torch.cuda.graph_pool_handle()
        with torch.cuda.graph(graph, pool=self.pool):
            self.logits = self._step()
        self.graphs[self.window] = (graph, self.logits)
        for i, c in saved[0].items():
            self.conv[i].copy_(c)
        for i, r in saved[1].items():
            self.rec[i].copy_(r)
        self.mask.copy_(saved[2])
        # capture does not run the step; replay now does it once for real
        graph.replay()
