"""Frozen 399-support, two-intervention oracle; no deployable selector or label-guided search."""
import hashlib
import importlib.metadata
import importlib.util
import json
from itertools import combinations
from pathlib import Path
from time import perf_counter

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from kvrepair.engine import logit_metrics

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
OUT = ROOT/'artifacts/qwen_headroom'


def module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    result = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(result)
    return result


r = module('preserved_reproduction', ROOT/'experiments/qwen_reproduction/run.py')
op = module('qwen_sparse_operator', HERE/'sparse_qwen.py')


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def cache_digest(cache):
    h = hashlib.sha256()
    for layer in cache.layers:
        for tensor in [layer.keys, layer.values]:
            h.update(tensor.detach().cpu().contiguous().view(torch.uint8).numpy().tobytes())
    return h.hexdigest()


def main():
    frozen = json.loads((HERE/'frozen.json').read_text())
    assert all(digest(ROOT/p) == h for p, h in frozen['files'].items())
    assert not (OUT/'run.json').exists(), 'Retain previous attempts'
    fixed = json.loads((HERE/'fixed_cases.json').read_text())
    assert fixed['actual_oracle_supports'] == 399 and fixed['max_oracle_supports'] == 400
    report = {'status': 'started', 'frozen_sha256': digest(HERE/'frozen.json'),
              'device': 'mps', 'dtype': 'float32', 'attention': 'eager',
              'author_model_revision': 'unknown', 'candidate_model_revision': r.MODEL_REVISION,
              'versions': {x: importlib.metadata.version(x) for x in ['torch', 'transformers', 'tokenizers']},
              'cases': [{'id': x['id'], 'status': 'not_run'} for x in fixed['cases']],
              'oracle_attempts': 0, 'control_attempts': 0}
    r.save(OUT/'run.json', report)
    run_start = perf_counter()
    torch.manual_seed(17)
    torch.set_num_threads(4)
    torch.use_deterministic_algorithms(True)
    try:
        load_start = perf_counter()
        path = ROOT/'.models/qwen3-1.7b'
        manifest = json.loads((path/'download-manifest.json').read_text())
        assert manifest['revision'] == r.MODEL_REVISION
        for name, info in manifest['files'].items():
            h = hashlib.sha256()
            with (path/name).open('rb') as f:
                for block in iter(lambda: f.read(8*1024*1024), b''):
                    h.update(block)
            assert h.hexdigest() == info['sha256'], name
        tok = AutoTokenizer.from_pretrained(path, local_files_only=True, trust_remote_code=False)
        model = AutoModelForCausalLM.from_pretrained(path, local_files_only=True,
                    trust_remote_code=False, dtype=torch.float32, attn_implementation='eager').eval()
        assert torch.equal(model.lm_head.weight, model.model.embed_tokens.weight)
        model.lm_head.weight = model.model.embed_tokens.weight
        model.to('mps')
        op.sync(model)
        report['model_load_and_weight_verification_ms'] = (perf_counter()-load_start)*1000
        source = json.loads((ROOT/'artifacts/qwen_reproduction/float32_resume/continued/generation.json').read_text())
        cpu = json.loads((ROOT/'artifacts/qwen_reproduction/float32_resume/cpu_reference.json').read_text())
        for fixed_case in fixed['cases']:
            case_start = perf_counter()
            item = next(x for x in report['cases'] if x['id'] == fixed_case['id'])
            item.update(status='started', controls=[], trials=[])
            r.save(OUT/'run.json', report)
            metadata_start = perf_counter()
            old_ids, new_ids = fixed_case['old_ids'], fixed_case['new_ids']
            edit = fixed_case['edit']
            mandatory = list(range(edit['start'], edit['end']))
            assert old_ids == r.aligned(tok, fixed_case['raw_old_rendered_prompt'], fixed_case['raw_new_rendered_prompt'])[0]
            assert new_ids == r.aligned(tok, fixed_case['raw_old_rendered_prompt'], fixed_case['raw_new_rendered_prompt'])[1]
            region_pairs = [tuple(x) for x in fixed_case['regions']]
            supports = [(budget, c) for budget in [16, 32]
                        for c in combinations(range(len(region_pairs)), budget//16)
                        if len(c) == 1 or region_pairs[c[0]][1] <= region_pairs[c[1]][0]]
            assert len(supports) == fixed_case['support_count']
            item['metadata_and_enumeration_ms'] = (perf_counter()-metadata_start)*1000
            old_start = perf_counter()
            old = r.forward(model, old_ids).past_key_values
            op.sync(model)
            item['old_prefill_ms'] = (perf_counter()-old_start)*1000
            old_before = cache_digest(old)
            full_start = perf_counter()
            full_out = r.forward(model, new_ids)
            op.sync(model)
            full_prefill_ms = (perf_counter()-full_start)*1000
            clean = full_out.past_key_values
            native_logits = full_out.logits.flatten().float().cpu()
            source_case = next(x for x in source['records'] if x['id'] == fixed_case['id'])
            cpu_case = next(x for x in cpu['records'] if x['id'] == fixed_case['id'])
            assert source_case['new_ids'] == new_ids == cpu_case['new_ids']
            cpu_metrics = logit_metrics(torch.tensor(cpu_case['full_logits']), native_logits)
            assert cpu_metrics['within_tolerance'] and cpu_metrics['top1_equal'], cpu_metrics
            item['native_full_vs_cpu_reference'] = cpu_metrics
            item['full_prefill_ms'] = full_prefill_ms
            case_path = OUT/fixed_case['id']/'outcomes.json'
            full_replay = None
            full_tokens = None

            def attempt(kind, positions=None, budget=0, region_indices=None):
                nonlocal full_replay, full_tokens
                record = {'kind': kind, 'status': 'started', 'budget': budget,
                          'region_indices': list(region_indices or []), 'timings_ms': {}}
                collection = item['trials'] if kind == 'oracle' else item['controls']
                collection.append(record)
                if kind == 'oracle':
                    report['oracle_attempts'] += 1
                    assert report['oracle_attempts'] <= 400
                else:
                    report['control_attempts'] += 1
                    assert report['control_attempts'] <= 10
                op.sync(model)
                started = perf_counter()
                if kind == 'full':
                    cache = clean
                    work = {'cache_copy_ms': 0, 'sparse_forward_ms': 0,
                            'repair_ms_including_copy': 0, 'token_layers_recomputed': len(new_ids)*28,
                            'positions': list(range(len(new_ids)))}
                    record['native_prefill_ms'] = full_prefill_ms
                elif kind == 'published_unrepaired':
                    copied = perf_counter()
                    cache = r.clone(old)
                    op.sync(model)
                    copy_ms = (perf_counter()-copied)*1000
                    with torch.inference_mode():
                        for dst, src in zip(cache.layers, clean.layers):
                            dst.keys[:, :, edit['start']:edit['end']] = src.keys[:, :, edit['start']:edit['end']].clone()
                            dst.values[:, :, edit['start']:edit['end']] = src.values[:, :, edit['start']:edit['end']].clone()
                    op.sync(model)
                    work = {'cache_copy_ms': copy_ms, 'sparse_forward_ms': 0,
                            'repair_ms_including_copy': (perf_counter()-copied)*1000,
                            'token_layers_recomputed': 0, 'positions': [],
                            'transplanted_field_rows': len(mandatory)*28,
                            'requires_clean_prefill_dependency_ms': full_prefill_ms}
                else:
                    cache, work = op.repair(model, old, new_ids, positions)
                record['work'] = work
                if kind != 'full':
                    assert all(a.keys.data_ptr() != b.keys.data_ptr() and a.values.data_ptr() != b.values.data_ptr()
                               for a, b in zip(cache.layers, old.layers))
                if kind == 'all_positions_endpoint':
                    record['all_layer_KV_max_abs'] = max(float((getattr(a, name)-getattr(b, name)).abs().max().cpu())
                        for a, b in zip(cache.layers, clean.layers) for name in ['keys', 'values'])
                readout_start = perf_counter()
                logits = r.last_step(model, cache, new_ids).logits.flatten().float().cpu()
                op.sync(model)
                record['timings_ms']['diagnostic_readout_including_copy'] = (perf_counter()-readout_start)*1000
                if kind == 'full':
                    full_replay = logits.clone()
                    record['native_full_cached_check'] = logit_metrics(native_logits, logits)
                    assert record['native_full_cached_check']['within_tolerance'] and record['native_full_cached_check']['top1_equal']
                metrics_start = perf_counter()
                record['logits_vs_full'] = logit_metrics(full_replay, logits)
                record['logits_vs_full']['relative_l2'] = float((full_replay-logits).norm()/full_replay.norm())
                record['timings_ms']['generic_metrics'] = (perf_counter()-metrics_start)*1000
                generation_start = perf_counter()
                record.update(r.generate(model, tok, cache, new_ids))
                op.sync(model)
                record['timings_ms']['generation_including_readout_and_copy'] = (perf_counter()-generation_start)*1000
                record['diagnostic_plus_generation_readout_tokens'] = 2
                record['decode_forward_tokens_after_first_readout'] = len(record['tokens'])-1
                if kind == 'full':
                    full_tokens = record['tokens']
                record['full_generated_ids_equal'] = record['tokens'] == full_tokens
                record['status'] = 'completed'
                record['timings_ms']['trial_compute_wall'] = (perf_counter()-started)*1000
                io_start = perf_counter()
                r.save(case_path, item)
                record['timings_ms']['first_checkpoint_write'] = (perf_counter()-io_start)*1000
                record['timings_ms']['trial_wall_including_first_checkpoint'] = (perf_counter()-started)*1000
                return record, logits

            full_record, _ = attempt('full')
            assert full_record['tokens'] == source_case['outcomes']['full']['tokens'], 'Full baseline output changed'
            stale_record, stale_logits = attempt('published_unrepaired')
            assert stale_record['tokens'] == source_case['outcomes']['field_only']['tokens'], 'Published unrepaired output changed'
            edit_record, edit_logits = attempt('actual_edit_only', mandatory)
            item['actual_edit_vs_transplant'] = logit_metrics(stale_logits, edit_logits)
            assert item['actual_edit_vs_transplant']['within_tolerance'] and edit_record['tokens'] == stale_record['tokens'], item['actual_edit_vs_transplant']
            for mode, positions in [('all_positions_endpoint', list(range(len(new_ids)))),
                                    ('full_suffix_endpoint', list(range(edit['start'], len(new_ids)-1)))]:
                endpoint, _ = attempt(mode, positions)
                assert endpoint['logits_vs_full']['within_tolerance'] and endpoint['tokens'] == full_tokens, mode
            assert cache_digest(old) == old_before, 'Control mutated original cache'
            print(fixed_case['id'], 'all baseline/operator gates passed;', len(supports), 'supports', flush=True)
            for ordinal, (budget, chosen) in enumerate(supports, 1):
                additional = [i for j in chosen for i in range(*region_pairs[j])]
                assert len(set(additional)) == budget
                attempt('oracle', mandatory+additional, budget, chosen)
                if ordinal % 25 == 0:
                    print(fixed_case['id'], ordinal, '/', len(supports), 'attempts saved', flush=True)
            assert cache_digest(old) == old_before, 'Search mutated original cache'
            item.update(status='completed', base_cache_unchanged=True,
                        wall_ms_including_setup_checks_and_checkpoints=(perf_counter()-case_start)*1000,
                        mps_memory_sample={'allocated_bytes': torch.mps.current_allocated_memory(),
                                           'driver_bytes': torch.mps.driver_allocated_memory()})
            r.save(case_path, item)
            r.save(OUT/'run.json', report)
            del old, clean, full_out, full_replay, stale_logits, edit_logits
        assert report['oracle_attempts'] == 399 and report['control_attempts'] == 10
        assert all(digest(ROOT/p) == h for p, h in frozen['files'].items())
        report.update(status='completed', completed=True,
                      wall_seconds_including_load_setup_and_all_attempts=perf_counter()-run_start)
        r.save(OUT/'run.json', report)
    except Exception as exc:
        report.update(status='failed', completed=False, error=type(exc).__name__+': '+str(exc),
                      wall_seconds_to_stop=perf_counter()-run_start)
        for item in report['cases']:
            if item['status'] == 'started':
                item.update(status='failed', error=report['error'])
                for record in item.get('controls', [])+item.get('trials', []):
                    if record['status'] == 'started':
                        record.update(status='failed', error=report['error'])
                r.save(OUT/item['id']/'outcomes.json', item)
        r.save(OUT/'run.json', report)
        raise


if __name__ == '__main__':
    main()
