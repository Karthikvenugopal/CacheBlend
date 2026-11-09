"""Isolated float32, batch-one Llama feasibility mechanisms.

This module leaves the validated CacheEngine paths unchanged. Statistics are
valid only for the original token sequence, fixed positions and fixed weights.
"""
from dataclasses import dataclass
from time import perf_counter

import torch
from transformers.models.llama.modeling_llama import apply_rotary_pos_emb, eager_attention_forward, repeat_kv

from .engine import Edit


@dataclass
class AttentionMetadata:
    ids: tuple[int, ...]
    owner: object
    q: torch.Tensor
    output: torch.Tensor | None
    lse: torch.Tensor  # float64 scalar/head/row for subtractive update stability
    collection: dict | None = None

    @property
    def nbytes(self):
        return sum(t.numel() * t.element_size() for t in (self.q, self.output, self.lse) if t is not None)


def _check(engine):
    if engine.model.training or next(engine.model.parameters()).dtype != torch.float32:
        raise ValueError("Probe requires eval-mode float32")
    if engine.model.config.model_type != "llama" or engine.model.config._attn_implementation != "eager":
        raise ValueError("Probe requires native full-attention Llama eager implementation")
    if getattr(engine.model.config, "sliding_window", None):
        raise ValueError("Sliding-window attention is unsupported")


@torch.inference_mode()
def project(engine, ids, positions, layer_index=0):
    layer = engine.model.model.layers[layer_index]
    h = engine.model.model.embed_tokens(torch.tensor([ids], device=engine.device))
    x = layer.input_layernorm(h)
    a = layer.self_attn
    shape = (1, len(ids), -1, a.head_dim)
    q, k, v = (p(x).view(shape).transpose(1, 2) for p in (a.q_proj, a.k_proj, a.v_proj))
    cos, sin = engine.model.model.rotary_emb(h, torch.tensor([positions], device=engine.device))
    q, k = apply_rotary_pos_emb(q, k, cos, sin)
    return q, k, v


@torch.inference_mode()
def _normalizers(engine, q, k, chunk=128):
    a = engine.model.model.layers[0].self_attn
    keys = repeat_kv(k, a.num_key_value_groups)
    n = q.shape[-2]
    result = []
    key_pos = torch.arange(n, device=engine.device)
    for begin in range(0, n, chunk):
        end = min(n, begin + chunk)
        scores = (q[..., begin:end, :] @ keys.transpose(-1, -2)) * a.scaling
        scores.masked_fill_(key_pos[None, :] > torch.arange(begin, end, device=engine.device)[:, None], -torch.inf)
        result.append(scores.double().logsumexp(-1))
    return torch.cat(result, dim=-1)


@torch.inference_mode()
def collect(engine, ids, with_output=True):
    """Native original prefill + hooks + an extra chunked QK pass for LSE.

    PyTorch eager attention does not expose LSE, so that extra pass is charged;
    no claim of free metadata collection is made.
    """
    _check(engine)
    a = engine.model.model.layers[0].self_attn
    saved = {'hook_ms': 0.0}
    def pre(_module, _args, kwargs):
        saved['rope'] = kwargs['position_embeddings']
    def query(_module, _args, output):
        start = perf_counter()
        saved['raw_q'] = output.clone()
        saved['hook_ms'] += (perf_counter()-start)*1000
    def attention_output(_module, args):
        start = perf_counter()
        saved['o'] = args[0].clone()
        saved['hook_ms'] += (perf_counter()-start)*1000
    engine.sync()
    start = perf_counter()
    handles = [a.register_forward_pre_hook(pre, with_kwargs=True), a.q_proj.register_forward_hook(query)]
    if with_output:
        handles.append(a.o_proj.register_forward_pre_hook(attention_output))
    try:
        original = engine.run(ids)
    finally:
        for h in handles:
            h.remove()
    post_start = perf_counter()
    q = saved['raw_q'].view(1, len(ids), -1, a.head_dim).transpose(1, 2)
    cos, sin = saved['rope']
    q, _ = apply_rotary_pos_emb(q, q, cos, sin)
    q = q.contiguous()
    output = saved['o'].view(1, len(ids), -1, a.head_dim).transpose(1, 2).contiguous() if with_output else None
    normalizers = _normalizers(engine, q, original.state.kv[0][0])
    engine.sync()
    total_ms = (perf_counter()-start)*1000
    component_times = {'native_instrumented_prefill_ms':original.elapsed_ms,
                       'metadata_postprocessing_ms':(perf_counter()-post_start)*1000,
                       'hook_copy_host_ms':saved['hook_ms']}
    return original.state, AttentionMetadata(tuple(ids), engine.owner, q, output, normalizers, component_times), total_ms


@torch.inference_mode()
def direct_attention(engine, ids):
    """Native first-layer attention, captured before its output projection."""
    _check(engine)
    layer = engine.model.model.layers[0]
    a = layer.self_attn
    saved = []
    handle = a.o_proj.register_forward_pre_hook(lambda _m, args: saved.append(args[0].clone()))
    try:
        h = engine.model.model.embed_tokens(torch.tensor([ids], device=engine.device))
        pos = torch.arange(len(ids), device=engine.device)
        cos_sin = engine.model.model.rotary_emb(h, pos[None])
        mask = torch.zeros((1, 1, len(ids), len(ids)), device=engine.device)
        mask.masked_fill_(pos[None, :] > pos[:, None], -torch.inf)
        a(layer.input_layernorm(h), position_embeddings=cos_sin, attention_mask=mask)
    finally:
        handle.remove()
    return saved[0].view(1, len(ids), -1, a.head_dim).transpose(1, 2).contiguous()


@torch.inference_mode()
def changed_output(engine, old, metadata, new_ids, fallback_threshold=1e-4):
    """Return updated suffix head outputs; never interpret these as repaired KV.

    If old edited mass nearly exhausts a row, use direct full-row attention.
    Unselected row/head updates are evaluated in float64 after float32 scores.
    """
    _check(engine)
    engine._check(old)
    if metadata.owner is not engine.owner or metadata.ids != old.ids or metadata.output is None:
        raise ValueError("Metadata must include outputs and match the original engine and token sequence")
    edit = Edit.between(old.ids, tuple(new_ids))
    p, e, n = edit.prefix, edit.end, edit.length
    if p == n:
        return metadata.output.clone(), {'fallback_token_rows': 0, 'suffix_start': 0}
    a = engine.model.model.layers[0].self_attn
    q = metadata.q[..., e:, :]
    if e == n:
        return q.clone(), {'fallback_token_rows': 0, 'suffix_start': e}
    _, new_k, new_v = project(engine, new_ids[p:e], list(range(p, e)))
    ok, ov = old.kv[0]
    old_k, old_v = ok[..., p:e, :], ov[..., p:e, :]
    groups = a.num_key_value_groups
    old_scores = ((q @ repeat_kv(old_k, groups).transpose(-1, -2)) * a.scaling).double()
    new_scores = ((q @ repeat_kv(new_k, groups).transpose(-1, -2)) * a.scaling).double()
    lse = metadata.lse[..., e:]
    shift = torch.maximum(lse, new_scores.max(-1).values)
    z = (lse - shift).exp()
    ow = (old_scores - shift[..., None]).exp()
    nw = (new_scores - shift[..., None]).exp()
    remaining = 1 - (old_scores - lse[..., None]).exp().sum(-1)
    denominator = z + nw.sum(-1) - ow.sum(-1)
    numerator = (metadata.output[..., e:, :].double() * z[..., None]
                 + nw @ repeat_kv(new_v, groups).double()
                 - ow @ repeat_kv(old_v, groups).double())
    result = numerator / denominator[..., None]
    bad = ((remaining < fallback_threshold) | (denominator <= 1e-12)
           | ~torch.isfinite(result).all(-1)).any(dim=1)[0]
    rows = bad.nonzero().flatten()
    if len(rows):
        full_k, full_v = ok.clone(), ov.clone()
        full_k[..., p:e, :], full_v[..., p:e, :] = new_k, new_v
        positions = rows + e
        scores = (q[..., rows, :] @ repeat_kv(full_k, groups).transpose(-1, -2)) * a.scaling
        mask = torch.arange(n, device=engine.device)[None, :] > positions[:, None]
        scores.masked_fill_(mask, -torch.inf)
        direct = scores.softmax(-1, dtype=torch.float32) @ repeat_kv(full_v, groups)
        result[..., rows, :] = direct.double()
    return result.float(), {'fallback_token_rows': len(rows), 'suffix_start': e,
                            'minimum_remaining_mass': float(remaining.min())}


@torch.inference_mode()
def scores(engine, old, metadata, new_ids, selector):
    edit = Edit.between(old.ids, tuple(new_ids))
    n, p, e = edit.length, edit.prefix, edit.end
    result = torch.zeros(n, device=engine.device)
    diagnostics = {'fallback_token_rows': 0}
    if p == n or e == n:
        return result, diagnostics
    if selector == 'delta':
        updated, diagnostics = changed_output(engine, old, metadata, new_ids)
        difference = (updated - metadata.output[..., e:, :]).transpose(1, 2).reshape(n-e, -1)
        result[e:] = difference.norm(dim=-1)
    elif selector == 'attention':
        a = engine.model.model.layers[0].self_attn
        k = repeat_kv(old.kv[0][0][..., p:e, :], a.num_key_value_groups)
        attention_scores = (metadata.q[..., e:, :] @ k.transpose(-1, -2)) * a.scaling
        result[e:] = (attention_scores.double() - metadata.lse[..., e:, None]).exp().sum(-1).mean(1)[0].float()
    else:
        raise ValueError(selector)
    return result, diagnostics


def select_regions(edit, token_scores, mode, budget=32, region_size=16, seed=17):
    """Same disjoint region grid and exact additional-token count for every selector."""
    import random
    if budget <= 0 or budget % region_size:
        raise ValueError("Budget must be a positive multiple of region size")
    if edit.prefix == edit.length:
        return []
    blocks = [(a, a+region_size) for a in range(edit.end, edit.length-region_size+1, region_size)]
    if len(blocks) < budget // region_size:
        raise ValueError("Not enough downstream full regions for the matched budget")
    if mode == 'local':
        chosen = blocks[:budget//region_size]
    elif mode == 'random':
        chosen = random.Random(seed).sample(blocks, budget//region_size)
    elif mode in {'delta', 'attention'}:
        chosen = sorted(blocks, key=lambda ab: (-float(token_scores[ab[0]:ab[1]].sum()), ab[0]))[:budget//region_size]
    else:
        raise ValueError(mode)
    return sorted(i for a, b in chosen for i in range(a, b))


@torch.inference_mode()
def repair(engine, old, new_ids, positions):
    """Layer-major selected-row forward with full causal KV context.

    Returns K/V only: a partially repaired document does not have valid stored
    final logits unless its last position was computed. The fresh question is
    always run separately with the native CacheEngine forward.
    """
    _check(engine)
    engine._check(old)
    Edit.between(old.ids, tuple(new_ids))
    positions = sorted(set(positions))
    if any(i < 0 or i >= len(new_ids) for i in positions):
        raise ValueError("Invalid selected position")
    engine.sync()
    begin = perf_counter()
    kv = tuple((k.clone(), v.clone()) for k, v in old.kv)
    engine.sync()
    copy_ms = (perf_counter() - begin) * 1000
    if positions:
        pos = torch.tensor(positions, device=engine.device)
        h = engine.model.model.embed_tokens(torch.tensor([[new_ids[i] for i in positions]], device=engine.device))
        cos, sin = engine.model.model.rotary_emb(h, pos[None])
        key_positions = torch.arange(len(new_ids), device=engine.device)
        mask = torch.zeros((1, 1, len(positions), len(new_ids)), device=engine.device)
        mask.masked_fill_(key_positions[None, :] > pos[:, None], -torch.inf)
        for layer, (keys, values) in zip(engine.model.model.layers, kv):
            x = layer.input_layernorm(h)
            a = layer.self_attn
            shape = (1, len(positions), -1, a.head_dim)
            q, k, v = (p(x).view(shape).transpose(1, 2) for p in (a.q_proj, a.k_proj, a.v_proj))
            q, k = apply_rotary_pos_emb(q, k, cos, sin)
            keys[..., pos, :], values[..., pos, :] = k, v
            output, _ = eager_attention_forward(a, q, keys, values, mask, scaling=a.scaling, dropout=0.0)
            h = h + a.o_proj(output.reshape(1, len(positions), -1))
            h = h + layer.mlp(layer.post_attention_layernorm(h))
    engine.sync()
    return kv, {'repair_ms_including_copy': (perf_counter()-begin)*1000, 'cache_copy_ms': copy_ms,
                'token_layers_recomputed': len(positions)*len(kv), 'positions': positions}


@torch.inference_mode()
def fresh_question(engine, document_ids, kv, question_ids):
    from .engine import Snapshot
    updated, logits = engine._forward(tuple(question_ids), kv)
    return Snapshot(tuple(document_ids)+tuple(question_ids), updated, logits, engine.owner)
