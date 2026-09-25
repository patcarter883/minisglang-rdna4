"""Gemma-4 vision tower (Gemma4VisionModel + its multimodal projector), shared by `gemma4` and
`diffusion_gemma` — the two checkpoints ship the same 27-layer tower under different namespaces.

Every piece runs on a native kernel; nothing here is a torch re-implementation of the math:

  patch input      tail_hip.vision_patch_in    uint8 patches -> 2*(p/255-0.5), the reference's fp32 order
  input_proj       dense GEMM (minv_linear)    [n, 768] @ [768, 1152]
  2-D position     tail_hip.vision_pos_add     x-table[x] + y-table[y]
  per layer, x27:
    q|k|v proj     dense GEMM, ONE fused weight [3*H*D, C]
    norms + rope   tail_hip.vision_qkv_prep    q/k weighted RMSNorm, v unweighted, axial 2-D RoPE, and a
                                               zero pad head_dim 72 -> 80 so the WMMA attention can run
    attention      attn_hip.flash_prefill      head_dim 80, bidirectional, one launch per image
    o_proj         dense GEMM on the PADDED [n, H*80] output against a weight padded with zero input
                   columns at load, so the pad is never sliced away
    post/pre norm  tail_hip.vision_sandwich    residual += norm(o)*w_post, then the next pre-norm
    gate|up        dense GEMM, ONE fused weight [2*I, C] + tail_hip.gelu_tanh_and_mul
    down           dense GEMM
    post/pre norm  tail_hip.vision_sandwich    (fused with the NEXT layer's input norm)
  pool             tail_hip.vision_pool        3x3 by position, * sqrt(C), standardize, projector pre-norm
  projection       dense GEMM                  [m, 1152] @ [1152, text_hidden]

Only an image's REAL patches are encoded. The reference pads every image to 2520 patches and masks the
padding out of attention, the pooler and the output; since padding keys are masked, padding is inert
for every real token, so skipping it is exact and removes up to all but a sliver of the work.

The tower is REPLICATED on every TP rank (fp16, ~1.1 GB): at this width a TP split would pay two
all-reduces per layer of [patches, 1152] over the PCIe link for ~half the FLOPs, which is a loss.
"""

from __future__ import annotations

from typing import List, Sequence, Tuple

import torch

from minisgl.layers.base import BaseOP, OPList

_PAD_HD = 80  # vision head_dim 72 -> the next multiple of the 16-wide WMMA tile


class _VisionLayer(BaseOP):
    def __init__(self, C: int, H: int, D: int, I: int) -> None:
        super().__init__()
        e = lambda *s: torch.empty(*s)
        self.q_proj, self.k_proj, self.v_proj, self.o_proj = e(H * D, C), e(H * D, C), e(H * D, C), e(C, H * D)
        self.q_norm, self.k_norm = e(D), e(D)
        self.gate_proj, self.up_proj, self.down_proj = e(I, C), e(I, C), e(C, I)
        self.ln_in, self.ln_post_attn, self.ln_pre_ff, self.ln_post_ff = e(C), e(C), e(C), e(C)
        self._H, self._D = H, D

    def post_load(self) -> None:
        H, D = self._H, self._D
        # Fused and padded forms, built once; the checkpoint-shaped tensors are then dropped.
        self._qkv = torch.cat([self.q_proj, self.k_proj, self.v_proj], dim=0).contiguous()
        C = self.o_proj.shape[0]
        o = self.o_proj.view(C, H, D)
        o_pad = o.new_zeros(C, H, _PAD_HD)
        o_pad[..., :D] = o
        self._o = o_pad.view(C, H * _PAD_HD).contiguous()
        # Intermediate padded to a multiple of 64 with ZERO rows/columns: 4304 is not a multiple of any
        # dense-GEMM N-tile, which would force the slower ragged-N kernel on both MLP GEMMs. Exact:
        # a zero gate row gives gelu(0) * up = 0, and a zero down column adds nothing.
        I = self.gate_proj.shape[0]
        Ip = -(-I // 64) * 64
        def pad_rows(w):
            return w if Ip == I else torch.cat([w, w.new_zeros(Ip - I, w.shape[1])], dim=0)
        self._gate_up = torch.cat([pad_rows(self.gate_proj), pad_rows(self.up_proj)], dim=0).contiguous()
        down = self.down_proj
        self._down = (down if Ip == I else torch.cat([down, down.new_zeros(down.shape[0], Ip - I)], dim=1)).contiguous()
        for name in ("q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"):
            setattr(self, name, torch.empty(0, device=self._qkv.device, dtype=self._qkv.dtype))
        self._post_load_done = True

    def forward(self, *args, **kwargs):  # driven by Gemma4VisionTower.forward
        raise NotImplementedError


class Gemma4VisionTower(BaseOP):
    def __init__(self, vision: dict, text_hidden: int) -> None:
        super().__init__()
        C, H, D, I = vision["hidden_size"], vision["num_heads"], vision["head_dim"], vision["intermediate_size"]
        assert D <= _PAD_HD and (H * _PAD_HD) % 16 == 0, (H, D)
        p = vision["patch_size"]
        self.patch_proj = torch.empty(C, 3 * p * p)
        self.pos_table = torch.empty(2, vision["position_embedding_size"], C)
        self.std_bias = torch.empty(C)
        self.std_scale = torch.empty(C)
        self.proj = torch.empty(text_hidden, C)
        self.layers = OPList([_VisionLayer(C, H, D, I) for _ in range(vision["num_layers"])])
        self._C, self._H, self._D = C, H, D
        self._eps = vision["rms_norm_eps"]
        self._pool = vision["pooling_kernel_size"]
        self._standardize = vision["standardize"]
        self.image_token_id = int(vision["image_token_id"])  # what the engine writes over the pad ids
        self._rope_theta = float(vision["rope_theta"])
        self._inv_freq = None

    def post_load(self) -> None:
        self.layers.post_load()
        # Axial RoPE inverse frequencies exactly as transformers computes them (fp32, base ** (i/half)).
        # Built HERE, not in __init__: the model is constructed on the meta device, where a tensor made
        # at construction has no data to copy.
        half = self._D // 2
        ar = torch.arange(0, half, 2, dtype=torch.float, device="cpu")
        self._inv_freq = (1.0 / (self._rope_theta ** (ar / half))).to(self.patch_proj.device)
        self._post_load_done = True

    def forward(self, *args, **kwargs):
        raise NotImplementedError("use encode()")

    @torch.inference_mode()
    def encode(self, images: Sequence[Tuple[torch.Tensor, int, int]]) -> List[torch.Tensor]:
        """[(uint8 patches [pw*ph, 768] on device, pw, ph)] -> per image [(pw/k)*(ph/k), text_hidden]."""
        import attn_hip
        import tail_hip

        from minisgl.layers.minv import minv_linear

        dev, dt = self.patch_proj.device, self.patch_proj.dtype
        C, H, eps, k = self._C, self._H, self._eps, self._pool
        sizes = [int(px.shape[0]) for px, _, _ in images]
        bounds = [0]
        for n in sizes:
            bounds.append(bounds[-1] + n)
        pixels = torch.cat([px for px, _, _ in images]) if len(images) > 1 else images[0][0]
        # Raster patch positions (x, y): the processor's meshgrid(arange(pw), arange(ph), "xy").
        pos = torch.cat([
            torch.stack([torch.arange(pw * ph, device=dev) % pw, torch.arange(pw * ph, device=dev) // pw], -1)
            for _, pw, ph in images
        ]).to(torch.int32).contiguous()

        x = tail_hip.vision_patch_in(pixels.contiguous(), 0.00392156862745098, dt)
        h = minv_linear(x, self.patch_proj)
        tail_hip.vision_pos_add(h, self.pos_table, pos)

        residual = h
        layers = self.layers.op_list
        x = tail_hip.rms_norm(residual, layers[0].ln_in, eps, 0)
        attn = torch.empty(residual.shape[0], H, _PAD_HD, device=dev, dtype=dt)
        for li, layer in enumerate(layers):
            qkv = minv_linear(x, layer._qkv)
            q, kk, v = tail_hip.vision_qkv_prep(qkv, layer.q_norm, layer.k_norm, pos, self._inv_freq,
                                                H, _PAD_HD, eps)
            for s, e in zip(bounds[:-1], bounds[1:]):
                attn[s:e] = attn_hip.flash_prefill(q[s:e], kk[s:e], v[s:e], 1.0, 0, 0)
            o = minv_linear(attn.view(-1, H * _PAD_HD), layer._o)
            x = tail_hip.vision_sandwich(o, residual, layer.ln_post_attn, layer.ln_pre_ff, eps)
            gu = minv_linear(x, layer._gate_up)
            d = minv_linear(tail_hip.gelu_tanh_and_mul(gu), layer._down)
            nxt = layers[li + 1].ln_in if li + 1 < len(layers) else None
            x = tail_hip.vision_sandwich(d, residual, layer.ln_post_ff, nxt, eps)

        pooled = [
            tail_hip.vision_pool(residual[s:e], self.std_bias if self._standardize else None,
                                 self.std_scale if self._standardize else None, pw, ph, k, eps)
            for (s, e), (_, pw, ph) in zip(zip(bounds[:-1], bounds[1:]), images)
        ]
        counts = [int(t.shape[0]) for t in pooled]
        out = minv_linear(torch.cat(pooled) if len(pooled) > 1 else pooled[0], self.proj)
        return list(out.split(counts))


def gemma4_vision_remap(name: str) -> str | None:
    """Checkpoint vision key -> the tower's native key under `vision.`, or None when not a vision key.

    Unknown keys inside the vision namespaces RAISE: a dropped tower weight would leave uninitialised
    memory that reads as a plausible-looking but wrong image embedding."""
    for pre in ("model.vision_tower.", "model.encoder.vision_tower."):
        if name.startswith(pre):
            rest = name[len(pre):]
            break
    else:
        for pre in ("model.embed_vision.", "model.encoder.embed_vision."):
            if name.startswith(pre):
                if name[len(pre):] == "embedding_projection.weight":
                    return "vision.proj"
                raise ValueError(f"Gemma-4 vision loader: unrecognised projector key {name!r}")
        return None
    direct = {
        "patch_embedder.input_proj.weight": "vision.patch_proj",
        "patch_embedder.position_embedding_table": "vision.pos_table",
        "std_bias": "vision.std_bias",
        "std_scale": "vision.std_scale",
    }
    if rest in direct:
        return direct[rest]
    parts = rest.split(".")
    if parts[:2] == ["encoder", "layers"] and len(parts) >= 5:
        idx, tail = parts[2], ".".join(parts[3:])
        leaf = {
            "input_layernorm.weight": "ln_in",
            "post_attention_layernorm.weight": "ln_post_attn",
            "pre_feedforward_layernorm.weight": "ln_pre_ff",
            "post_feedforward_layernorm.weight": "ln_post_ff",
            "self_attn.q_proj.linear.weight": "q_proj",
            "self_attn.k_proj.linear.weight": "k_proj",
            "self_attn.v_proj.linear.weight": "v_proj",
            "self_attn.o_proj.linear.weight": "o_proj",
            "self_attn.q_norm.weight": "q_norm",
            "self_attn.k_norm.weight": "k_norm",
            "mlp.gate_proj.linear.weight": "gate_proj",
            "mlp.up_proj.linear.weight": "up_proj",
            "mlp.down_proj.linear.weight": "down_proj",
        }.get(tail)
        if leaf is not None:
            return f"vision.layers.{idx}.{leaf}"
    raise ValueError(f"Gemma-4 vision loader: unrecognised tower key {name!r}")
