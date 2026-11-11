"""Readable batch-one Qwen3 cache composition and selected-row execution.

No clean-cache transplant. Source caches are immutable; each run owns its tensors.
Only ordinary full-attention, default-RoPE, float32/eager Qwen3 is supported.
"""
import torch
from transformers.cache_utils import DynamicCache
from transformers.models.qwen3.modeling_qwen3 import rotate_half, apply_rotary_pos_emb, eager_attention_forward


def clone(cache):
    return DynamicCache(ddp_cache_data=[(x.keys.clone(), x.values.clone()) for x in cache.layers])


def pairs(cache):
    return [(x.keys, x.values) for x in cache.layers]


def sync(model):
    if next(model.parameters()).device.type == 'mps':
        torch.mps.synchronize()


@torch.inference_mode()
def compose(model, chunks, validate=False):
    """Independent prefills, then R(global) R(local)^-1 on post-RoPE K.

    Use the actual native coefficients, including their float32 roundoff,
    rather than relying on trig addition at distant positions on MPS.
    """
    assert model.config.model_type == 'qwen3'
    assert model.config.rope_parameters['rope_type'] == 'default'
    device = next(model.parameters()).device
    collected = [[] for _ in model.model.layers]
    rotation_checks = []
    offset = 0
    for ids in chunks:
        raw_keys = []
        handles = []
        if validate:
            for layer in model.model.layers:
                handles.append(layer.self_attn.k_norm.register_forward_hook(
                    lambda module, args, output: raw_keys.append(output.transpose(1, 2).clone())))
        try:
            out = model(torch.tensor([ids], device=device), use_cache=True, logits_to_keep=1)
        finally:
            for handle in handles:
                handle.remove()
        positions = torch.arange(len(ids), device=device)[None]
        dummy = model.model.embed_tokens.weight[None, :1]
        c0, s0 = model.model.rotary_emb(dummy, positions)
        c1, s1 = model.model.rotary_emb(dummy, positions + offset)
        denom = c0.square() + s0.square()
        cos = ((c1*c0+s1*s0)/denom)[:, None]
        sin = ((s1*c0-c1*s0)/denom)[:, None]
        for li, (dst, (k, v)) in enumerate(zip(collected, pairs(out.past_key_values))):
            corrected = k.clone() if offset == 0 else k*cos + rotate_half(k)*sin
            dst.append((corrected, v.clone()))
            if validate:
                direct = raw_keys[li]*c1[:,None] + rotate_half(raw_keys[li])*s1[:,None]
                rotation_checks.append({'chunk_offset':offset,'layer':li,**check(direct,corrected)})
        if offset == 0:
            first_logits = out.logits[0, -1].clone()
        offset += len(ids)
        del out
    cache = DynamicCache(ddp_cache_data=[(torch.cat([p[0] for p in xs], 2),
                                         torch.cat([p[1] for p in xs], 2)) for xs in collected])
    cache._cacheblend_model = model
    cache._cacheblend_ids = tuple(token for chunk in chunks for token in chunk)
    cache._cacheblend_first_logits = first_logits
    cache._cacheblend_rotation_checks = rotation_checks
    return cache


@torch.inference_mode()
def execute(model, ids, document_length, source, mode='full', ratio=0.16):
    """Custom full endpoint, naive reuse, or reference V-HKVD schedule.

    reference: layer 0 full; layer 1 full QKV, select Q/attention/MLP rows;
    layers 2+ selected rows. All fresh suffix rows are mandatory on every layer.
    zero is reference with ratio=0, including its two early refreshed KV layers.
    """
    assert mode in ('full', 'naive', 'reference') and 0 <= ratio <= 1
    assert next(model.parameters()).dtype == torch.float32
    assert model.config._attn_implementation == 'eager' and not model.training
    assert all(x.attention_type == 'full_attention' for x in model.model.layers)
    device = next(model.parameters()).device
    n, d = len(ids), document_length
    assert 0 < d < n and source.get_seq_length() == d
    assert source._cacheblend_model is model, 'cache belongs to another model'
    assert source._cacheblend_ids == tuple(ids[:d]), 'cache token IDs differ'
    pos = torch.arange(d if mode == 'naive' else 0, n, device=device)
    hidden = model.model.embed_tokens(torch.tensor([ids], device=device)[:, pos])
    result, work = [], []
    selected = None
    for i, layer in enumerate(model.model.layers):
        a = layer.self_attn
        residual = hidden
        x = layer.input_layernorm(hidden)
        shape = (*x.shape[:-1], -1, a.head_dim)
        q = a.q_norm(a.q_proj(x).view(shape)).transpose(1, 2)
        k = a.k_norm(a.k_proj(x).view(shape)).transpose(1, 2)
        v = a.v_proj(x).view(shape).transpose(1, 2)
        emb = model.model.rotary_emb(x, pos[None])
        q, k = apply_rotary_pos_emb(q, k, *emb)
        qkv_rows = len(pos)
        old_k, old_v = pairs(source)[i]
        if mode == 'reference' and i == 1:
            # The reference uses V only, before attention on the check layer.
            scores = (v[:, :, :d] - old_v).square().sum((0, 1, 3))
            count = int(d * ratio)
            chosen = torch.topk(scores, count).indices.sort().values
            selected = torch.cat((chosen, torch.arange(d, n, device=device)))
            keep = selected  # input still contains every row at layer 1
            pos = pos[keep]
            q, residual = q[:, :, keep], residual[:, keep]
        if mode == 'full' or (mode == 'reference' and i <= 1):
            keys, values = k, v
        else:
            keys = torch.cat((old_k.clone(), k.new_zeros((1, k.shape[1], n-d, k.shape[3]))), 2)
            values = torch.cat((old_v.clone(), v.new_zeros((1, v.shape[1], n-d, v.shape[3]))), 2)
            keys.index_copy_(2, pos, k)
            values.index_copy_(2, pos, v)
        mask = torch.zeros((1, 1, len(pos), n), device=device)
        mask.masked_fill_(torch.arange(n, device=device)[None] > pos[:, None], -torch.inf)
        out, _ = eager_attention_forward(a, q, keys, values, mask, scaling=a.scaling, dropout=0.0)
        hidden = residual + a.o_proj(out.reshape(1, len(pos), -1).contiguous())
        hidden = hidden + layer.mlp(layer.post_attention_layernorm(hidden))
        result.append((keys, values))
        work.append({'layer': i, 'qkv_rows': qkv_rows, 'attention_mlp_rows': len(pos),
                     'positions': pos.tolist()})
    logits = model.lm_head(model.model.norm(hidden[:, -1:]))[0, -1]
    return DynamicCache(ddp_cache_data=result), logits, work


@torch.inference_mode()
def generate(model, cache, logits, max_tokens=32):
    """Greedy unrestricted generation on an owned cache branch."""
    cache = clone(cache)
    ids = []
    eos = model.generation_config.eos_token_id
    eos = [eos] if isinstance(eos, int) else eos
    for _ in range(max_tokens):
        token = int(logits.argmax())
        ids.append(token)
        if token in eos:
            break
        out = model(torch.tensor([[token]], device=logits.device), past_key_values=cache,
                    use_cache=True, logits_to_keep=1)
        logits = out.logits[0, -1]
        cache = out.past_key_values
    return ids, ids[-1] in eos


def divergence(reference, candidate):
    # Statistics on CPU float64; MPS float64 is unsupported.
    p = reference.detach().cpu().double().log_softmax(-1)
    q = candidate.detach().cpu().double().log_softmax(-1)
    return {'kl_full_to_method': float((p.exp()*(p-q)).sum()),
            'total_variation': float((p.exp()-q.exp()).abs().sum()/2),
            'max_abs_logit_error': float((reference-candidate).abs().max()),
            'top1_agreement': bool(reference.argmax() == candidate.argmax())}


def check(reference, candidate):
    r, c = reference.detach().cpu(), candidate.detach().cpu()
    return {'passed': bool(torch.allclose(r, c, atol=2e-4, rtol=2e-4)),
            'max_abs': float((r-c).abs().max()), 'atol': 2e-4, 'rtol': 2e-4}
