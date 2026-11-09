"""Batch-one, unpadded, full-attention Llama cache experiments.

All ranges are zero-based and half-open. Snapshots own their storage; the
engine never supplies a snapshot's tensors directly to a mutable HF cache.
"""
from dataclasses import dataclass
from time import perf_counter

import torch
from transformers.cache_utils import DynamicCache


class UnsupportedEdit(ValueError):
    pass


@dataclass(frozen=True)
class Edit:
    prefix: int
    end: int
    length: int

    @classmethod
    def between(cls, old: tuple[int, ...], new: tuple[int, ...]):
        if not old or not new:
            raise UnsupportedEdit("Empty prompts are unsupported")
        if len(old) != len(new):
            raise UnsupportedEdit("Insertions/deletions or unequal token lengths are unsupported")
        p = 0
        while p < len(old) and old[p] == new[p]:
            p += 1
        e = len(old)
        while e > p and old[e - 1] == new[e - 1]:
            e -= 1
        return cls(p, e, len(old))


@dataclass(frozen=True)
class Snapshot:
    ids: tuple[int, ...]
    kv: tuple[tuple[torch.Tensor, torch.Tensor], ...]
    logits: torch.Tensor
    owner: object

    def clone(self):
        return Snapshot(self.ids, tuple((k.clone(), v.clone()) for k, v in self.kv),
                        self.logits.clone(), self.owner)

    @property
    def nbytes(self):
        return sum(t.numel() * t.element_size() for pair in self.kv for t in pair)


@dataclass
class Run:
    state: Snapshot
    mode: str
    reused: list[tuple[int, int]]
    recomputed: list[tuple[int, int]]
    elapsed_ms: float

    @property
    def tokens_recomputed(self):
        return sum(b - a for a, b in self.recomputed)


class CacheEngine:
    def __init__(self, model):
        if model.config.model_type != "llama" or getattr(model.config, "sliding_window", None):
            raise ValueError("Initial adapter supports full-attention Llama models only")
        self.model = model.eval()
        self.device = next(model.parameters()).device
        self.owner = object()

    def sync(self):
        if self.device.type == "mps":
            torch.mps.synchronize()
        elif self.device.type == "cuda":
            torch.cuda.synchronize()

    def _check(self, old):
        if old.owner is not self.owner:
            raise ValueError("Snapshot belongs to a different model/engine")

    def _slice(self, kv, end):
        return tuple((k[..., :end, :].clone(), v[..., :end, :].clone()) for k, v in kv)

    @torch.inference_mode()
    def _forward(self, ids, kv=()):
        if not ids:
            raise ValueError("Cannot forward an empty token span")
        start = kv[0][0].shape[-2] if kv else 0
        # Copy on entry: DynamicCache.update mutates its cache. Never alias snapshots.
        cache = DynamicCache(ddp_cache_data=[(k.clone(), v.clone()) for k, v in kv]) if kv else None
        positions = torch.arange(start, start + len(ids), device=self.device)
        out = self.model(
            input_ids=torch.tensor([ids], device=self.device),
            past_key_values=cache, use_cache=True,
            position_ids=positions.unsqueeze(0), cache_position=positions,
            attention_mask=torch.ones((1, start + len(ids)), dtype=torch.long, device=self.device),
            logits_to_keep=1,
        )
        owned = tuple((layer.keys.clone(), layer.values.clone()) for layer in out.past_key_values.layers)
        return owned, out.logits[0, -1].clone()

    @torch.inference_mode()
    def run(self, ids, mode="full", old=None):
        ids = tuple(ids)
        if not ids:
            raise ValueError("Empty prompts are unsupported")
        if len(ids) > self.model.config.max_position_embeddings:
            raise ValueError("Prompt exceeds configured context length")
        if mode not in {"full", "prefix", "naive"}:
            raise ValueError(f"Unknown mode: {mode}")
        if mode != "full" and old is None:
            raise ValueError("Reuse requires an original snapshot")
        if old is not None:
            self._check(old)
        # Time includes edit matching, cache copies, splicing, forward, synchronization.
        self.sync()
        begin = perf_counter()
        if mode == "full":
            kv, logits = self._forward(ids)
            reused, recomputed = [], [(0, len(ids))]
        else:
            edit = Edit.between(old.ids, ids)
            p, e, n = edit.prefix, edit.end, edit.length
            if p == n:
                state = old.clone()  # Stored final logits avoid forwarding the last token.
                kv, logits = state.kv, state.logits
                reused, recomputed = [(0, n)], []
            elif mode == "prefix":
                kv, logits = self._forward(ids[p:], self._slice(old.kv, p))
                reused, recomputed = ([(0, p)] if p else []), [(p, n)]
            else:
                # Approximate: compute changed envelope with an exact left prefix,
                # then retain old downstream dependencies. Replay the last token
                # to obtain new logits; a cached KV alone is not a final logit.
                kv, logits = self._forward(ids[p:e], self._slice(old.kv, p))
                reused = [(0, p)] if p else []
                recomputed = [(p, e)]
                if e < n:
                    kv = tuple((torch.cat((k, ok[..., e:n-1, :]), dim=-2),
                                torch.cat((v, ov[..., e:n-1, :]), dim=-2))
                               for (k, v), (ok, ov) in zip(kv, old.kv))
                    kv, logits = self._forward(ids[-1:], kv)
                    if e < n - 1:
                        reused.append((e, n - 1))
                    recomputed.append((n - 1, n))
        self.sync()
        elapsed = (perf_counter() - begin) * 1000
        return Run(Snapshot(ids, kv, logits, self.owner), mode, reused, recomputed, elapsed)

    @torch.inference_mode()
    def generate(self, state, max_new_tokens=12, eos_token_id=None):
        self._check(state)
        if max_new_tokens < 0 or len(state.ids) + max_new_tokens > self.model.config.max_position_embeddings:
            raise ValueError("Invalid continuation length")
        kv, logits = self._slice(state.kv, len(state.ids)), state.logits.clone()
        tokens = []
        for i in range(max_new_tokens):
            token = int(logits.argmax())
            tokens.append(token)
            if token == eos_token_id or i == max_new_tokens - 1:
                break
            kv, logits = self._forward((token,), kv)
        return tokens


@torch.inference_mode()
def logit_metrics(reference, candidate, atol=2e-4, rtol=2e-4):
    a, b = reference.float(), candidate.float()
    pa, pb = a.softmax(-1), b.softmax(-1)
    return {
        "max_abs": float((a - b).abs().max()),
        "mean_abs": float((a - b).abs().mean()),
        "kl_full_to_candidate": float((pa * (a.log_softmax(-1) - b.log_softmax(-1))).sum().clamp_min(0)),
        "total_variation": float((pa - pb).abs().sum() / 2),
        "top1_equal": int(a.argmax()) == int(b.argmax()),
        "within_tolerance": bool(torch.allclose(a, b, atol=atol, rtol=rtol)),
        "full_top1_margin": float(a.topk(2).values.diff().abs()[0]),
    }
