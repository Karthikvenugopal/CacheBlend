"""Actual selected-row recomputation through native Qwen3, no clean-cache transplant."""
from time import perf_counter
import torch
from transformers.cache_utils import DynamicCache


def sync(model):
    if next(model.parameters()).device.type == 'mps':
        torch.mps.synchronize()


class IndexedCache(DynamicCache):
    def __init__(self, old, positions):
        super().__init__(ddp_cache_data=[(x.keys.clone(), x.values.clone()) for x in old.layers])
        self.positions = positions

    def update(self, key_states, value_states, layer_idx, cache_kwargs=None):
        layer = self.layers[layer_idx]
        layer.keys.index_copy_(2, self.positions, key_states)
        layer.values.index_copy_(2, self.positions, value_states)
        return layer.keys, layer.values


@torch.inference_mode()
def repair(model, old, new_ids, positions):
    assert model.config.model_type == 'qwen3' and not model.training
    assert next(model.parameters()).dtype == torch.float32
    assert model.config._attn_implementation == 'eager'
    assert not model.config.sliding_window
    positions = sorted(set(positions))
    assert positions and min(positions) >= 0 and max(positions) < len(new_ids)
    assert old.get_seq_length() == len(new_ids)
    device = next(model.parameters()).device
    pos = torch.tensor(positions, device=device)
    sync(model)
    begin = perf_counter()
    cache = IndexedCache(old, pos)
    sync(model)
    copy_ms = (perf_counter()-begin)*1000
    begin_compute = perf_counter()
    key_positions = torch.arange(len(new_ids), device=device)
    mask = torch.zeros((1, 1, len(positions), len(new_ids)), device=device)
    mask.masked_fill_(key_positions[None, :] > pos[:, None], -torch.inf)
    model.model(input_ids=torch.tensor([[new_ids[i] for i in positions]], device=device),
                past_key_values=cache, cache_position=pos, position_ids=pos[None],
                attention_mask={'full_attention': mask}, use_cache=True)
    sync(model)
    assert cache.get_seq_length() == len(new_ids)
    return cache, {'cache_copy_ms': copy_ms,
                   'sparse_forward_ms': (perf_counter()-begin_compute)*1000,
                   'repair_ms_including_copy': (perf_counter()-begin)*1000,
                   'positions': positions, 'token_layers_recomputed': len(positions)*len(old.layers)}
