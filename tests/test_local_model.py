"""Opt-in verification against an already cached pretrained model; no downloads."""
import os

import pytest
import torch

from kvrepair.engine import CacheEngine, Edit, logit_metrics
from kvrepair.models import configure, load

pytestmark = pytest.mark.skipif(os.environ.get("KVREPAIR_LOCAL_TESTS") != "1",
                                reason="set KVREPAIR_LOCAL_TESTS=1 for offline pretrained verification")


@pytest.fixture(scope="module")
def local():
    configure(17, 4)
    model, tokenizer = load()
    return CacheEngine(model), tokenizer


@pytest.mark.parametrize("old_text,new_text", [
    ("Alpha beta gamma delta.", "Alpha beta gamma delta."),
    ("Red teams use gate 12. Blue teams use gate 14.", "Blue teams use gate 12. Blue teams use gate 14."),
    ("Red teams use gate 12. Blue teams use gate 14.", "Red teams use gate 73. Blue teams use gate 14."),
    ("Red teams use gate 12. Blue teams use gate 14.", "Red teams use gate 12. Blue teams use gate 18."),
    ("The code is 1234. Answer:", "The code is 1235. Answer:"),
])
def test_pretrained_exact(local, old_text, new_text):
    engine, tokenizer = local
    old_ids = tuple(tokenizer.encode(old_text))
    new_ids = tuple(tokenizer.encode(new_text))
    edit = Edit.between(old_ids, new_ids)
    old = engine.run(old_ids).state
    backup = old.clone()
    full = engine.run(new_ids).state
    exact = engine.run(new_ids, "prefix", old).state
    metrics = logit_metrics(full.logits, exact.logits)
    assert metrics["within_tolerance"], metrics
    a, b = engine.generate(full, 8), engine.generate(exact, 8)
    assert a == b, {"full":a, "prefix":b, "metrics":metrics}
    engine.generate(engine.run(new_ids, "naive", old).state, 2)
    for (k, v), (bk, bv) in zip(old.kv, backup.kv):
        torch.testing.assert_close(k, bk, atol=0, rtol=0)
        torch.testing.assert_close(v, bv, atol=0, rtol=0)
    print({"prefix": edit.prefix, "end":edit.end, "length":len(new_ids), **metrics})


def test_pretrained_token_boundary(local):
    engine, tokenizer = local
    # Discover an equal-token-length replacement with a token straddling the
    # character seam, rather than hardcoding SentencePiece boundary assumptions.
    found = None
    for old_text, new_text in [("international", "internetional"), ("playing", "played"),
                               ("refund", "refuns"), ("1234", "1235"), ("hello", "hallo")]:
        a = tokenizer(old_text, return_offsets_mapping=True)
        b = tokenizer(new_text, return_offsets_mapping=True)
        if len(a["input_ids"]) != len(b["input_ids"]):
            continue
        edit = Edit.between(tuple(a["input_ids"]), tuple(b["input_ids"]))
        char_prefix = next(i for i, (x,y) in enumerate(zip(old_text,new_text)) if x != y)
        if a["offset_mapping"][edit.prefix][0] < char_prefix:
            found = a["input_ids"], b["input_ids"], old_text, new_text, edit
            break
    assert found is not None, "No suitable tokenizer seam found; extend the explicit fixture list"
    a, b, old_text, new_text, edit = found
    old = engine.run(a).state
    full = engine.run(b).state
    exact = engine.run(b, "prefix", old).state
    assert logit_metrics(full.logits, exact.logits)["within_tolerance"]
    assert engine.generate(full, 8) == engine.generate(exact, 8)
    print({"boundary_old":old_text,"boundary_new":new_text,"edit":edit})
