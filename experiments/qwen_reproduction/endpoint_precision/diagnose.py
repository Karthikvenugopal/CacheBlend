"""First-case-only native cache diagnosis, one backend/precision per process."""
import argparse
import hashlib
import importlib.metadata
import importlib.util
import json
import os
from pathlib import Path
from time import perf_counter

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

ROOT = Path(__file__).resolve().parents[3]
OUT = ROOT/'artifacts/qwen_reproduction/endpoint_precision'
MODEL = ROOT/'.models/qwen3-1.7b'


def write(path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, allow_nan=False)+'\n')


def cache_digest(cache):
    h = hashlib.sha256()
    for layer in cache.layers:
        for tensor in [layer.keys, layer.values]:
            h.update(tensor.detach().cpu().contiguous().view(torch.uint8).numpy().tobytes())
    return h.hexdigest()


def independent(a, b):
    return all(x.keys.data_ptr() != y.keys.data_ptr() and
               x.values.data_ptr() != y.values.data_ptr()
               for x, y in zip(a.layers, b.layers))


def errors(a, b, atol):
    a, b = a.float().cpu(), b.float().cpu()
    delta = (a-b).abs()
    return {'max_absolute': float(delta.max()),
            'max_relative_elementwise_floor_1e-6': float((delta/a.abs().clamp_min(1e-6)).max()),
            'relative_linf': float(delta.max()/a.abs().max().clamp_min(1e-20)),
            'relative_l2': float((a-b).norm()/a.norm().clamp_min(1e-20)),
            'allclose_atol_rtol': atol,
            'within_tolerance': bool(torch.allclose(a, b, atol=atol, rtol=atol)),
            'exact_equal': bool(torch.equal(a, b))}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('configuration', choices=['cpu-float32', 'mps-float32', 'mps-bfloat16'])
    args = p.parse_args()
    device, dtype_name = args.configuration.split('-')
    dtype = getattr(torch, dtype_name)
    dest = OUT/args.configuration
    assert not dest.exists(), 'Preserve prior diagnostic outputs'
    dest.mkdir(parents=True)
    torch.manual_seed(17)
    torch.set_num_threads(4)
    torch.use_deterministic_algorithms(True)
    report = {'configuration': args.configuration, 'status': 'started',
              'scope': 'retail_refund edited prompt only; no generation or repair',
              'versions': {x: importlib.metadata.version(x) for x in ['torch', 'transformers', 'tokenizers']},
              'environment': {x: os.environ.get(x) for x in ['PYTORCH_MPS_FAST_MATH', 'PYTORCH_MPS_PREFER_METAL', 'PYTORCH_ENABLE_MPS_FALLBACK']}}
    write(dest/'result.json', report)
    started = perf_counter()
    try:
        spec = importlib.util.spec_from_file_location('existing_reproduction', ROOT/'experiments/qwen_reproduction/run.py')
        previous = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(previous)
        case = json.loads((ROOT/'experiments/qwen_reproduction/cases.json').read_text())[0]
        assert case['id'] == 'retail_refund'
        tokenizer = AutoTokenizer.from_pretrained(MODEL, local_files_only=True, trust_remote_code=False)
        old, text = previous.render(tokenizer, case['old_body']), previous.render(tokenizer, case['new_body'])
        assert text == case['expected_rendered_prompts']['new']
        _, ids, edit = previous.aligned(tokenizer, old, text)
        assert ids == tokenizer(text, add_special_tokens=True)['input_ids']  # First case has no artificial padding.
        ci = tokenizer(case['correct'], add_special_tokens=False)['input_ids'][0]
        si = tokenizer(case['stale'], add_special_tokens=False)['input_ids'][0]
        model = AutoModelForCausalLM.from_pretrained(MODEL, local_files_only=True,
            trust_remote_code=False, dtype=dtype, attn_implementation='eager').eval()
        assert torch.equal(model.lm_head.weight, model.model.embed_tokens.weight)
        model.lm_head.weight = model.model.embed_tokens.weight
        model.to(device)
        assert not any(m.training for m in model.modules())
        active = {}
        handles = []
        flags = []

        def capture(name):
            def hook(module, inputs, output):
                value = output[0] if isinstance(output, tuple) else output
                active[name] = value[0, -1].detach().float().cpu().clone().flatten()
                flags.append(torch.is_inference_mode_enabled())
            return hook

        def capture_input(name):
            def hook(module, inputs):
                active[name] = inputs[0][0, -1].detach().float().cpu().clone().flatten()
            return hook

        handles.append(model.model.embed_tokens.register_forward_hook(capture('embedding')))
        handles.append(model.model.norm.register_forward_hook(capture('final_norm')))
        for i, layer in enumerate(model.model.layers):
            handles.append(layer.register_forward_hook(capture(f'layer.{i}.output')))
        layer0 = model.model.layers[0]
        for name in ['input_layernorm', 'self_attn.q_proj', 'self_attn.k_proj', 'self_attn.v_proj',
                     'self_attn.q_norm', 'self_attn.k_norm', 'self_attn.o_proj', 'self_attn',
                     'post_attention_layernorm', 'mlp']:
            handles.append(layer0.get_submodule(name).register_forward_hook(capture('layer.0.'+name)))
        handles.append(layer0.self_attn.o_proj.register_forward_pre_hook(capture_input('layer.0.before_o_proj')))

        calls = []
        def run_native(input_ids, cache=None):
            nonlocal active
            active = {}
            length = cache.get_seq_length() if cache is not None else 0
            calls.append({'input_ids': input_ids[0].cpu().tolist(), 'cache_length_before': length,
                          'automatic_positions': list(range(length, length+input_ids.shape[1]))})
            # Native API: no manual cloning, splicing, positions, mask, or cache construction.
            out = model(input_ids=input_ids, past_key_values=cache, use_cache=True, logits_to_keep=1)
            return out, dict(active)

        def logit_metrics(a, b):
            a, b = a.flatten().float().cpu(), b.flatten().float().cpu()
            return {**errors(a, b, .02 if dtype == torch.bfloat16 else .0002),
                    'top_ids': [int(a.argmax()), int(b.argmax())],
                    'top_pieces': [tokenizer.decode([int(a.argmax())]), tokenizer.decode([int(b.argmax())])],
                    'top_agreement': int(a.argmax()) == int(b.argmax()),
                    'restricted_margins': [float(a[ci]-a[si]), float(b[ci]-b[si])],
                    'restricted_choice_agreement': bool((a[ci]>=a[si]) == (b[ci]>=b[si]))}

        with torch.inference_mode():
            ids_tensor = torch.tensor([ids], device=device)
            full, full_layers = run_native(ids_tensor)
            full_before = cache_digest(full.past_key_values)
            prefix, _ = run_native(ids_tensor[:, :-1])
            prefix_length = prefix.past_key_values.get_seq_length()
            owns = independent(full.past_key_values, prefix.past_key_values)
            # Hand the exact native returned cache straight back to the model.
            incremental, incremental_layers = run_native(ids_tensor[:, -1:], prefix.past_key_values)
            prefix_length_after = prefix.past_key_values.get_seq_length()
            assert torch.equal(torch.cat([ids_tensor[:, :-1], ids_tensor[:, -1:]], dim=1), ids_tensor)
            assert prefix_length == len(ids)-1 and prefix_length_after == len(ids)
            assert incremental.past_key_values is prefix.past_key_values
            full_unchanged_after_native = cache_digest(full.past_key_values) == full_before
            # Secondary control only: compare the existing wrapper, independent of the native test.
            active = {}
            wrapper = previous.last_step(model, full.past_key_values, ids)
            full_unchanged_after_wrapper = cache_digest(full.past_key_values) == full_before
            wrapper_owns = independent(full.past_key_values, wrapper.past_key_values)
            full_logits = full.logits.detach().float().cpu().clone()
            incremental_logits = incremental.logits.detach().float().cpu().clone()
            wrapper_logits = wrapper.logits.detach().float().cpu().clone()

        for handle in handles:
            handle.remove()
        tolerance = .02 if dtype == torch.bfloat16 else .0002
        layers = {name: errors(full_layers[name], incremental_layers[name], tolerance)
                  for name in full_layers}
        bitwise = [i for i in range(28) if not layers[f'layer.{i}.output']['exact_equal']]
        outside = [i for i in range(28) if not layers[f'layer.{i}.output']['within_tolerance']]
        report.update(status='completed', token_ids=ids, token_count=len(ids), token_sha256=hashlib.sha256(bytes(text, 'utf8')).hexdigest(),
                      token_ids_sha256=hashlib.sha256(json.dumps(ids).encode()).hexdigest(), edit=edit,
                      full_vs_native_incremental=logit_metrics(full_logits, incremental_logits),
                      full_vs_existing_wrapper=logit_metrics(full_logits, wrapper_logits),
                      native_incremental_vs_wrapper=logit_metrics(incremental_logits, wrapper_logits),
                      first_layer_with_any_bitwise_output_difference=bitwise[0] if bitwise else None,
                      first_layer_output_outside_frozen_tolerance=outside[0] if outside else None,
                      layer_and_component_errors=layers, native_calls=calls,
                      invariants={'all_modules_eval': not any(m.training for m in model.modules()),
                        'all_observed_forwards_inference_mode': all(flags),
                        'same_ids_no_duplicate_last_token': True,
                        'prefix_length_before': prefix_length, 'length_after': prefix_length_after,
                        'native_cache_returned_directly_and_mutated_in_its_branch': True,
                        'native_full_and_prefix_storage_independent': owns,
                        'full_cache_unchanged_after_native_continuation': full_unchanged_after_native,
                        'full_cache_unchanged_after_wrapper': full_unchanged_after_wrapper,
                        'wrapper_storage_independent': wrapper_owns},
                      restricted_ids={'correct': ci, 'stale': si}, elapsed_seconds_including_loading=perf_counter()-started)
        assert all(value for key, value in report['invariants'].items() if isinstance(value, bool))
        if args.configuration != 'cpu-float32':
            ref = torch.load(OUT/'cpu-float32/tensors.pt', weights_only=True, map_location='cpu')
            report['cpu_float32_reference'] = {
                'full_vs_full': logit_metrics(ref['full_logits'], full_logits),
                'incremental_vs_incremental': logit_metrics(ref['incremental_logits'], incremental_logits),
                'full_layer_output_errors': {name: errors(ref['full_layers'][name], full_layers[name], .0002)
                                             for name in full_layers}}
        torch.save({'full_logits': full_logits, 'incremental_logits': incremental_logits,
                    'wrapper_logits': wrapper_logits, 'full_layers': full_layers,
                    'incremental_layers': incremental_layers}, dest/'tensors.pt')
        write(dest/'result.json', report)
        print(json.dumps({k: report[k] for k in ['configuration', 'status', 'full_vs_native_incremental',
              'native_incremental_vs_wrapper', 'first_layer_with_any_bitwise_output_difference',
              'first_layer_output_outside_frozen_tolerance', 'invariants']}, indent=2))
    except Exception as exc:
        report.update(status='failed', error=type(exc).__name__+': '+str(exc))
        write(dest/'result.json', report)
        raise


if __name__ == '__main__':
    main()
