"""Random-model functional checks only; no factual cases/checkpoint."""
import argparse
import json
import torch
from transformers import Qwen3Config, Qwen3ForCausalLM
from transformers.cache_utils import DynamicCache
import importlib.util
from pathlib import Path

spec = importlib.util.spec_from_file_location('indexed_qwen_operator', Path(__file__).with_name('sparse_qwen.py'))
op = importlib.util.module_from_spec(spec)
spec.loader.exec_module(op)


def clone(cache, end=None):
    return DynamicCache(ddp_cache_data=[(x.keys[:, :, :end].clone(), x.values[:, :, :end].clone()) for x in cache.layers])


def main():
    p = argparse.ArgumentParser()
    p.add_argument('device', choices=['cpu', 'mps'])
    device = p.parse_args().device
    torch.manual_seed(17)
    torch.set_num_threads(4)
    torch.use_deterministic_algorithms(True)
    config = Qwen3Config(vocab_size=257, hidden_size=32, intermediate_size=64,
                        num_hidden_layers=3, num_attention_heads=4, num_key_value_heads=2,
                        head_dim=8, max_position_embeddings=128, attention_dropout=0.0)
    config._attn_implementation = 'eager'
    m = Qwen3ForCausalLM(config).to(device).eval()
    a, b = [1, 2, 3, 4, 5, 6, 7, 8], [1, 2, 9, 4, 5, 6, 7, 8]
    with torch.inference_mode():
        def forward(ids, cache=None):
            return m(input_ids=torch.tensor([ids], device=device), past_key_values=cache,
                     use_cache=True, logits_to_keep=1)
        old = forward(a).past_key_values
        backup = clone(old)
        full = forward(b)
        all_rows, _ = op.repair(m, old, b, range(len(b)))
        suffix, _ = op.repair(m, old, b, range(2, len(b)))
        changed, _ = op.repair(m, old, b, [2])
        native_edit = forward([9], clone(old, 2)).past_key_values
        errors = []
        for x, y, z, target, saved in zip(old.layers, all_rows.layers, changed.layers, full.past_key_values.layers, backup.layers):
            assert torch.equal(x.keys, saved.keys) and torch.equal(x.values, saved.values)
            assert x.keys.data_ptr() != y.keys.data_ptr() and x.keys.data_ptr() != z.keys.data_ptr()
            errors.extend([float((y.keys-target.keys).abs().max().cpu()), float((y.values-target.values).abs().max().cpu())])
            torch.testing.assert_close(y.keys, target.keys, atol=.0002, rtol=.0002)
            torch.testing.assert_close(y.values, target.values, atol=.0002, rtol=.0002)
        for x, y in zip(changed.layers, native_edit.layers):
            torch.testing.assert_close(x.keys[:, :, 2:3], y.keys[:, :, 2:3], atol=.0002, rtol=.0002)
            torch.testing.assert_close(x.values[:, :, 2:3], y.values[:, :, 2:3], atol=.0002, rtol=.0002)
        logits = forward(b[-1:], clone(suffix, len(b)-1)).logits
        torch.testing.assert_close(logits, full.logits, atol=.0002, rtol=.0002)
    print(json.dumps({'device': device, 'full_KV_max_abs': max(errors),
          'native_changed_span_equivalence': True, 'full_suffix_endpoint': True,
          'old_cache_unchanged': True, 'independent_storage': True, 'random_model_only': True}, indent=2))


if __name__ == '__main__':
    main()
