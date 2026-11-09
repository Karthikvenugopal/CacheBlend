import pytest
import torch
from tokenizers import Tokenizer, models
from transformers import PreTrainedTokenizerFast

from kvrepair.engine import CacheEngine, Edit, UnsupportedEdit, logit_metrics
from kvrepair.models import configure, load


@pytest.fixture
def engine():
    configure(17, 1)
    return CacheEngine(load(toy=True)[0])


@pytest.mark.parametrize("position", [None, 0, 5, 11])
def test_prefix_matches_full_and_cache_at_every_layer(engine, position):
    old_ids = tuple(range(1, 13))
    new_ids = list(old_ids)
    if position is not None:
        new_ids[position] = 80
    old = engine.run(old_ids).state
    full = engine.run(new_ids).state
    exact = engine.run(new_ids, "prefix", old).state
    assert logit_metrics(full.logits, exact.logits)["within_tolerance"]
    for (fk, fv), (ek, ev) in zip(full.kv, exact.kv):
        torch.testing.assert_close(fk, ek, atol=2e-5, rtol=2e-5)
        torch.testing.assert_close(fv, ev, atol=2e-5, rtol=2e-5)
    assert engine.generate(full, 8) == engine.generate(exact, 8)


def test_cache_ownership_and_order_independence(engine):
    ids = tuple(range(1, 13))
    changed = (1, 2, 99) + ids[3:]
    old = engine.run(ids).state
    backup = old.clone()
    prefix = engine.run(changed, "prefix", old).state
    naive = engine.run(changed, "naive", old).state
    engine.generate(naive, 5)
    again = engine.run(changed, "prefix", old).state
    torch.testing.assert_close(prefix.logits, again.logits, atol=0, rtol=0)
    states = [old, prefix, naive, again, engine.run(ids, "prefix", old).state]
    for layer in range(len(old.kv)):
        pointers = [s.kv[layer][0].data_ptr() for s in states]
        assert len(set(pointers)) == len(pointers)
    for (k, v), (bk, bv) in zip(old.kv, backup.kv):
        torch.testing.assert_close(k, bk, atol=0, rtol=0)
        torch.testing.assert_close(v, bv, atol=0, rtol=0)
    with torch.inference_mode():
        naive.kv[-1][0].zero_()  # Even deliberate mutation cannot alter siblings.
    torch.testing.assert_close(old.kv[-1][0], backup.kv[-1][0], atol=0, rtol=0)


def test_naive_reuses_stale_suffix_and_repairs_changed_span(engine):
    ids = tuple(range(1, 17))
    edited = ids[:4] + (99,) + ids[5:]
    old = engine.run(ids).state
    full = engine.run(edited).state
    naive = engine.run(edited, "naive", old)
    assert naive.recomputed == [(4, 5), (15, 16)]
    assert naive.reused == [(0, 4), (5, 15)]
    for (nk, nv), (ok, ov), (fk, fv) in zip(naive.state.kv, old.kv, full.kv):
        torch.testing.assert_close(nk[..., 5:15, :], ok[..., 5:15, :], atol=0, rtol=0)
        torch.testing.assert_close(nv[..., 5:15, :], ov[..., 5:15, :], atol=0, rtol=0)
        torch.testing.assert_close(nk[..., :5, :], fk[..., :5, :], atol=2e-5, rtol=2e-5)
        torch.testing.assert_close(nv[..., :5, :], fv[..., :5, :], atol=2e-5, rtol=2e-5)
    assert logit_metrics(full.logits, naive.state.logits)["max_abs"] > 1e-5


def test_tokenizer_boundary_retokenizes_left_of_character_edit(engine):
    # Replacing character b with c changes the merged token ab, which starts
    # BEFORE the character edit. No whitespace pretokenizer conceals the seam.
    backend = Tokenizer(models.BPE({"[UNK]":0,"a":1,"b":2,"c":3,"x":4,"ab":5,"ac":6},
                                   merges=[("a","b"),("a","c")], unk_token="[UNK]"))
    tok = PreTrainedTokenizerFast(tokenizer_object=backend, unk_token="[UNK]")
    old_text, new_text = "xabx", "xacx"
    a = tuple(tok.encode(old_text, add_special_tokens=False))
    b = tuple(tok.encode(new_text, add_special_tokens=False))
    edit = Edit.between(a, b)
    assert old_text[:2] == new_text[:2]  # Character prefix length 2, token prefix 1.
    assert edit.prefix == 1 and edit.end == 2
    old = engine.run(a).state
    full = engine.run(b).state
    exact = engine.run(b, "prefix", old).state
    assert logit_metrics(full.logits, exact.logits)["within_tolerance"]
    assert engine.generate(full, 6) == engine.generate(exact, 6)


@pytest.mark.parametrize("new", [(1, 2), (1, 2, 3, 4)])
@pytest.mark.parametrize("mode", ["prefix", "naive"])
def test_rejects_insertions_deletions(engine, new, mode):
    with pytest.raises(UnsupportedEdit, match="Insertions/deletions"):
        engine.run(new, mode, engine.run((1, 2, 3)).state)


def test_separated_edits_recompute_envelope(engine):
    old = engine.run((1, 2, 3, 4, 5)).state
    result = engine.run((1, 9, 3, 8, 5), "naive", old)
    assert result.recomputed == [(1, 4), (4, 5)]
    assert result.tokens_recomputed == 4
    assert logit_metrics(engine.run((1, 9, 3, 8, 5)).state.logits, result.state.logits)["within_tolerance"]


def test_foreign_model_snapshot_rejected(engine):
    other = CacheEngine(load(toy=True)[0])
    with pytest.raises(ValueError, match="different model"):
        engine.run((1, 2), "prefix", other.run((1, 2)).state)
