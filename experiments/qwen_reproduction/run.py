"""Six fixed published decision pairs; offline-only cache interventions, no repair selector."""
import argparse
import hashlib
import importlib.metadata
import json
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, Qwen3Config, Qwen3ForCausalLM
from transformers.cache_utils import DynamicCache

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
MODEL_REVISION = '70d244cc86ccca08cf5af4e1e306ecf908b1ad5e'
SOURCE_REVISION = 'e9085eafcc6c83e60c548de69060c4bd5b210c96'
MAX_NEW = 64


def save(path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix('.next.json')
    temp.write_text(json.dumps(obj, indent=2, allow_nan=False) + '\n')
    temp.replace(path)


def fingerprint():
    return {p.name: hashlib.sha256(p.read_bytes()).hexdigest()
            for p in [HERE/'run.py', HERE/'download.py', HERE/'cases.json', HERE/'protocol.md']}


def clone(cache, end=None):
    return DynamicCache(ddp_cache_data=[
        (layer.keys[:, :, :end].clone(), layer.values[:, :, :end].clone())
        for layer in cache.layers])


def aligned(tok, old, new):
    # Paper align_pair: tokenize complete chat renderings, pad shorter edit with space ID.
    a = tok(old, add_special_tokens=True)['input_ids']
    b = tok(new, add_special_tokens=True)['input_ids']
    p = 0
    while p < min(len(a), len(b)) and a[p] == b[p]:
        p += 1
    s = 0
    while s < min(len(a), len(b))-p and a[-s-1] == b[-s-1]:
        s += 1
    am, bm = a[p:len(a)-s], b[p:len(b)-s]
    n = max(len(am), len(bm))
    space = tok(' ', add_special_tokens=False)['input_ids'][-1]
    tail = a[len(a)-s:]
    old_ids = a[:p] + am + [space]*(n-len(am)) + tail
    new_ids = a[:p] + bm + [space]*(n-len(bm)) + tail
    assert len(old_ids) == len(new_ids) and n > 0
    return old_ids, new_ids, {'start': p, 'end': p+n, 'space_id': space,
                              'old_padding': n-len(am), 'new_padding': n-len(bm)}


@torch.inference_mode()
def forward(model, ids, cache=None, position=None):
    device = next(model.parameters()).device
    args = {'input_ids': torch.tensor([ids], device=device), 'use_cache': True,
            'logits_to_keep': 1}
    if cache is not None:
        args['past_key_values'] = cache
        args['cache_position'] = torch.arange(position, position+len(ids), device=device)
    return model(**args)


def last_step(model, cache, ids):
    # Discard the cached last token and replay it, exactly as the published readout.
    return forward(model, [ids[-1]], clone(cache, len(ids)-1), len(ids)-1)


def restricted(logits, tok, correct, stale):
    c = tok(correct, add_special_tokens=False)['input_ids']
    s = tok(stale, add_special_tokens=False)['input_ids']
    assert c[0] != s[0], 'Published alternatives collapse to the same first token'
    v = logits[0, -1].float().cpu()
    return {'correct_ids': c, 'stale_ids': s, 'correct_logit': float(v[c[0]]),
            'stale_logit': float(v[s[0]]), 'margin': float(v[c[0]]-v[s[0]]),
            'choice': 'correct' if v[c[0]] >= v[s[0]] else 'stale',
            'vocabulary_argmax_id': int(v.argmax()),
            'vocabulary_argmax_piece': tok.decode([int(v.argmax())])}


@torch.inference_mode()
def generate(model, tok, cache, ids):
    out = last_step(model, cache, ids)
    tokens = []
    eos = model.generation_config.eos_token_id
    eos = {eos} if isinstance(eos, int) else set(eos)
    assert eos == {151645, 151643}, 'Candidate native EOS configuration changed'
    for _ in range(MAX_NEW):
        nx = int(out.logits[0, -1].argmax())
        tokens.append(nx)
        if nx in eos or len(tokens) == MAX_NEW:
            break
        out = forward(model, [nx], out.past_key_values, len(ids)+len(tokens)-1)
    return {'tokens': tokens, 'text': tok.decode(tokens, skip_special_tokens=True),
            'hit_limit': len(tokens) == MAX_NEW and tokens[-1] not in eos,
            'utility': 'pending manual review; labels did not restrict generation'}


def render(tok, body):
    return tok.apply_chat_template([{'role': 'user', 'content': body}], tokenize=False,
                                   add_generation_prompt=True, enable_thinking=False)


def states(model, tok, case):
    texts = {k: render(tok, case[k+'_body']) for k in ['old', 'new', 'erratum']}
    assert texts == case['expected_rendered_prompts'], 'Reviewed chat rendering changed'
    a, b, edit = aligned(tok, texts['old'], texts['new'])
    e = tok(texts['erratum'], add_special_tokens=False)['input_ids']
    assert max(len(a), len(e)) <= 2048, 'Stop rather than shorten a published prompt'
    old = forward(model, a).past_key_values
    new = forward(model, b).past_key_values
    field = clone(old)
    for dst, src in zip(field.layers, new.layers):
        dst.keys[:, :, edit['start']:edit['end']] = src.keys[:, :, edit['start']:edit['end']].clone()
        dst.values[:, :, edit['start']:edit['end']] = src.values[:, :, edit['start']:edit['end']].clone()
    # Clean cached endpoint must agree numerically with native edited prefill.
    direct = forward(model, b).logits.float().cpu()
    replay = last_step(model, new, b).logits.float().cpu()
    tolerance = 0.02 if next(model.parameters()).dtype == torch.bfloat16 else 0.0002
    endpoint_error = float((direct-replay).abs().max())
    assert torch.allclose(direct, replay, atol=tolerance, rtol=tolerance), endpoint_error
    assert int(direct.argmax()) == int(replay.argmax()), 'Native/cached first greedy token differs; investigate'
    c = tok(case['correct'], add_special_tokens=False)['input_ids'][0]
    s = tok(case['stale'], add_special_tokens=False)['input_ids'][0]
    assert bool(direct[0, -1, c] >= direct[0, -1, s]) == bool(replay[0, -1, c] >= replay[0, -1, s]), 'Native/cached restricted decision differs; investigate'
    meta = {'rendered_prompts': texts, 'old_ids': a, 'new_ids': b, 'erratum_ids': e,
            'edit': edit, 'endpoint_max_abs_error': endpoint_error,
            'endpoint_atol_rtol': tolerance,
            'affected_tokens': [{'position': i, 'old': tok.decode([a[i]]),
                                 'new': tok.decode([b[i]])} for i in range(edit['start'], edit['end'])]}
    return meta, {'original_full': (old, a), 'full': (new, b), 'field_only': (field, b)}


def preflight(device, dtype):
    # Random tiny Qwen architecture only: no pretrained weights, tokenizer, or factual cases.
    config = Qwen3Config(vocab_size=257, hidden_size=32, intermediate_size=64,
                        num_hidden_layers=2, num_attention_heads=4, num_key_value_heads=2,
                        head_dim=8, max_position_embeddings=128, attention_dropout=0.0)
    config._attn_implementation = 'eager'
    model = Qwen3ForCausalLM(config).to(device=device, dtype=dtype).eval()
    a, b = [1, 3, 4, 5, 6], [1, 7, 4, 5, 6]
    old = forward(model, a).past_key_values
    backup = clone(old)
    new_out = forward(model, b)
    new = new_out.past_key_values
    replay = last_step(model, new, b)
    error = float((new_out.logits.float().cpu()-replay.logits.float().cpu()).abs().max())
    tolerance = 0.02 if dtype == torch.bfloat16 else 0.0002
    assert torch.allclose(new_out.logits.float().cpu(), replay.logits.float().cpu(),
                          atol=tolerance, rtol=tolerance)
    mixed = clone(old)
    for dest, src in zip(mixed.layers, new.layers):
        dest.keys[:, :, 1:2] = src.keys[:, :, 1:2].clone()
        dest.values[:, :, 1:2] = src.values[:, :, 1:2].clone()
    branch = last_step(model, mixed, b)
    forward(model, [int(branch.logits.argmax())], branch.past_key_values, len(b))
    for x, y in zip(old.layers, backup.layers):
        assert torch.equal(x.keys, y.keys) and torch.equal(x.values, y.values)
    return {'device': device, 'dtype': str(dtype), 'native_vs_cached_max_error': error,
            'cache_branch_ownership_passed': True, 'pretrained_model_loaded': False}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('stage', choices=['preflight', 'restricted', 'generation'])
    p.add_argument('--device', choices=['cpu', 'mps'], default='mps')
    p.add_argument('--dtype', choices=['bfloat16', 'float32'], default='bfloat16')
    args = p.parse_args()
    torch.manual_seed(17)
    torch.set_num_threads(4)
    torch.use_deterministic_algorithms(True)
    dtype = getattr(torch, args.dtype)
    if args.stage == 'preflight':
        print(json.dumps(preflight(args.device, dtype), indent=2))
        return
    path = ROOT/'artifacts/qwen_reproduction'/f'{args.stage}.json'
    assert not path.exists(), 'Retain prior outputs; do not overwrite'
    frozen = json.loads((HERE/'prepared.json').read_text())
    assert frozen['files'] == fingerprint(), 'Prepared protocol/code/cases changed'
    baseline = None
    if args.stage == 'generation':
        baseline = json.loads((path.parent/'restricted.json').read_text())
        assert baseline.get('completed') and len(baseline['records']) == 6
        assert baseline['files'] == fingerprint()
        assert (baseline['device'], baseline['dtype']) == (args.device, args.dtype)
    model_path = ROOT/'.models/qwen3-1.7b'
    download = json.loads((model_path/'download-manifest.json').read_text())
    assert download['revision'] == MODEL_REVISION
    for name, recorded in download['files'].items():
        h = hashlib.sha256()
        with (model_path/name).open('rb') as stream:
            for block in iter(lambda: stream.read(8*1024*1024), b''):
                h.update(block)
        assert h.hexdigest() == recorded['sha256'], name
    tok = AutoTokenizer.from_pretrained(model_path, local_files_only=True, trust_remote_code=False)
    model = AutoModelForCausalLM.from_pretrained(model_path, local_files_only=True,
                trust_remote_code=False, dtype=dtype, attn_implementation='eager').eval()
    assert model.config.model_type == 'qwen3'
    separate_identical_weights = model.lm_head.weight is not model.model.embed_tokens.weight
    if separate_identical_weights:
        assert model.config.tie_word_embeddings
        assert torch.equal(model.lm_head.weight, model.model.embed_tokens.weight), 'Checkpoint embedding/output values differ; stop'
        # Transformers 5.2 leaves duplicate checkpoint entries untied. Values were
        # verified identical; restore the declared alias before device transfer.
        model.lm_head.weight = model.model.embed_tokens.weight
    model.to(args.device)
    assert model.lm_head.weight is model.model.embed_tokens.weight, 'Unexpected embedding/output tying'
    report = {'stage': args.stage, 'model_revision_provenance': 'candidate pin; author revision unreported',
              'candidate_model_revision': MODEL_REVISION, 'source_revision': SOURCE_REVISION,
              'device': args.device, 'dtype': args.dtype, 'files': fingerprint(),
              'generation': {'max_new_tokens': MAX_NEW, 'do_sample': False, 'num_beams': 1,
                             'penalties': 'none', 'enable_thinking': False, 'seed': 17,
                             'eos_ids': [151645, 151643],
                             'stop': 'native generation-config EOS or fixed token limit', 'vocabulary': 'unrestricted'},
              'torch_version': torch.__version__,
              'versions': {x: importlib.metadata.version(x) for x in ['torch', 'transformers', 'tokenizers', 'safetensors']},
              'unique_model_parameters': sum(p.numel() for p in model.parameters()),
              'identical_checkpoint_weights_aliased': separate_identical_weights,
              'download_manifest': download,
              'records': []}
    cases = json.loads((HERE/'cases.json').read_text())
    assert len(cases) == 6
    report['records'] = [{'id': c['id'], 'status': 'not_run'} for c in cases]
    save(path, report)
    for case in cases:
        item = next(r for r in report['records'] if r['id'] == case['id'])
        item['status'] = 'started'
        save(path, report)
        try:
            meta, branches = states(model, tok, case)
        except Exception as exc:
            item.update(status='failed', error=type(exc).__name__+': '+str(exc))
            save(path, report)
            raise
        item.update(meta)
        item['outcomes'] = {}
        if args.stage == 'restricted':
            err = forward(model, meta['erratum_ids']).past_key_values
            branches['erratum'] = (err, meta['erratum_ids'])
        for name, (cache, ids) in branches.items():
            score = restricted(last_step(model, cache, ids).logits, tok, case['correct'], case['stale'])
            if args.stage == 'generation':
                previous = next(r for r in baseline['records'] if r['id'] == case['id'])
                assert previous['new_ids'] == meta['new_ids'] and previous['old_ids'] == meta['old_ids']
                assert score['choice'] == previous['outcomes'][name]['choice'], 'Scoring changed on replay'
                item['outcomes'][name] = {'restricted': score, **generate(model, tok, cache, ids)}
            else:
                score['published_choice'] = case['published'].get(name)
                score['matches_published'] = (score['choice'] == score['published_choice']
                                               if score['published_choice'] else None)
                item['outcomes'][name] = score
        save(path, report)
        item['status'] = 'completed'
        print(case['id'], 'saved', flush=True)
        del branches
    report['completed'] = True
    save(path, report)


if __name__ == '__main__':
    main()
