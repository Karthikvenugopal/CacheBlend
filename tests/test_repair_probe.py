import pytest
import torch

from kvrepair.engine import CacheEngine, Edit
from kvrepair.models import configure, load
from kvrepair.repair_probe import collect, changed_output, direct_attention, repair, fresh_question, scores, select_regions


@pytest.fixture
def engine():
    configure(17, 1)
    return CacheEngine(load(toy=True)[0])


@pytest.mark.parametrize('position', [None, 0, 5, 20, 47])
def test_delta_matches_native_first_attention(engine, position):
    ids = list(range(1, 49))
    old, metadata, _ = collect(engine, ids)
    new = ids.copy()
    if position is not None:
        new[position] = 100
    updated, info = changed_output(engine, old, metadata, new)
    reference = direct_attention(engine, new)[..., info['suffix_start']:, :]
    torch.testing.assert_close(updated, reference, atol=2e-5, rtol=2e-5)
    torch.testing.assert_close(old.kv[0][0], engine.run(ids).state.kv[0][0], atol=0, rtol=0)


def test_fallback_and_metadata_binding(engine):
    ids = list(range(1, 33))
    old, metadata, _ = collect(engine, ids)
    new = ids.copy(); new[5] = 99
    updated, info = changed_output(engine, old, metadata, new, fallback_threshold=1.1)
    assert info['fallback_token_rows'] == len(ids)-6
    torch.testing.assert_close(updated, direct_attention(engine, new)[..., 6:, :], atol=2e-5, rtol=2e-5)
    other = CacheEngine(load(toy=True)[0])
    with pytest.raises(ValueError):
        changed_output(other, old, metadata, new)


def test_full_endpoint_fresh_question_and_independent_ownership(engine):
    ids = list(range(1, 49)); new = ids.copy(); new[6] = 99
    original = engine.run(ids).state; backup = original.clone()
    kv, _ = repair(engine, original, new, list(range(len(ids))))
    reference = engine.run(new).state
    for actual_pair, ref_pair in zip(kv, reference.kv):
        for actual, ref in zip(actual_pair, ref_pair):
            torch.testing.assert_close(actual, ref, atol=2e-5, rtol=2e-5)
    joined = fresh_question(engine, new, kv, [90, 91])
    full = engine.run(new+[90, 91]).state
    torch.testing.assert_close(joined.logits, full.logits, atol=2e-5, rtol=2e-5)
    for (a, b), (x, y) in zip(original.kv, backup.kv):
        torch.testing.assert_close(a, x, atol=0, rtol=0)
        torch.testing.assert_close(b, y, atol=0, rtol=0)
    assert kv[0][0].data_ptr() != original.kv[0][0].data_ptr()


def test_matched_budget_and_unselected_states(engine):
    ids = list(range(1, 81)); new = ids.copy(); new[4] = 99
    old, meta, _ = collect(engine, ids)
    edit = Edit.between(tuple(ids), tuple(new))
    sets = {}
    for mode in ['delta', 'attention', 'local', 'random']:
        signal, _ = scores(engine, old, meta, new, mode) if mode in ['delta', 'attention'] else (None, {})
        sets[mode] = select_regions(edit, signal, mode)
        assert len(sets[mode]) == 32
    chosen = list(range(edit.prefix, edit.end)) + sets['delta']
    kv, _ = repair(engine, old, new, chosen)
    untouched = [i for i in range(len(ids)) if i not in chosen]
    for (k, v), (ok, ov) in zip(kv, old.kv):
        torch.testing.assert_close(k[..., untouched, :], ok[..., untouched, :], atol=0, rtol=0)
        torch.testing.assert_close(v[..., untouched, :], ov[..., untouched, :], atol=0, rtol=0)
