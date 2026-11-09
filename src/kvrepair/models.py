"""Offline-only model loading and reproducible diagnostic configuration."""
import importlib.metadata
import os
import platform
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, LlamaConfig, LlamaForCausalLM

MODEL = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
REVISION = "fe8a4ea1ffedaf415f4da2f062534de366a451e6"


class ByteTokenizer:
    """Synthetic smoke tests only: no language understanding, no BPE claims."""
    eos_token_id = 0

    def encode(self, text, **kwargs):
        return [b + 1 for b in text.encode("utf-8")]

    def decode(self, ids, **kwargs):
        return bytes(i - 1 for i in ids if 1 <= i <= 256).decode("utf-8", errors="replace")


def configure(seed=17, threads=4):
    torch.manual_seed(seed)
    torch.set_num_threads(threads)
    torch.use_deterministic_algorithms(True)


def load(model_path=None, device="cpu", toy=False):
    if device == "mps" and not torch.backends.mps.is_available():
        raise ValueError("MPS is unavailable in this process")
    if toy:
        cfg = LlamaConfig(vocab_size=257, hidden_size=32, intermediate_size=64,
                          num_hidden_layers=3, num_attention_heads=4,
                          num_key_value_heads=2, max_position_embeddings=2048,
                          attention_dropout=0.0)
        cfg._attn_implementation = "eager"
        return LlamaForCausalLM(cfg).to(device).eval(), ByteTokenizer()
    kwargs = {"local_files_only": True}
    if model_path is None:
        kwargs["revision"] = REVISION
    name = model_path or MODEL
    tokenizer = AutoTokenizer.from_pretrained(name, **kwargs)
    model = AutoModelForCausalLM.from_pretrained(
        name, dtype=torch.float32, attn_implementation="eager", **kwargs,
    ).to(device).eval()
    return model, tokenizer


def _version(package):
    try:
        return importlib.metadata.version(package)
    except importlib.metadata.PackageNotFoundError:
        return None


def diagnostics():
    memory = None
    processor = None
    try:
        import ctypes
        lib = ctypes.CDLL(None, use_errno=True)
        value = ctypes.c_uint64()
        size = ctypes.c_size_t(ctypes.sizeof(value))
        if lib.sysctlbyname(b"hw.memsize", ctypes.byref(value), ctypes.byref(size), None, 0) == 0:
            memory = value.value
        brand = ctypes.create_string_buffer(256)
        brand_size = ctypes.c_size_t(256)
        if lib.sysctlbyname(b"machdep.cpu.brand_string", brand, ctypes.byref(brand_size), None, 0) == 0:
            processor = brand.value.decode()
    except (AttributeError, OSError):
        pass
    return {
        "platform": platform.platform(), "machine": platform.machine(),
        "cpu_count": os.cpu_count(), "processor": processor, "physical_memory_bytes": memory,
        "memory_note": "null means OS query unavailable; no RAM assumption made",
        "mps_built": torch.backends.mps.is_built(), "mps_available": torch.backends.mps.is_available(),
        "cuda_available": torch.cuda.is_available(),
        "versions": {p: _version(p) for p in ["torch", "transformers", "tokenizers", "safetensors", "pytest"]},
        "python": platform.python_version(),
        "cached_default_snapshot": str(Path.home() / ".cache/huggingface/hub" / ("models--" + MODEL.replace("/", "--")) / "snapshots" / REVISION),
    }
