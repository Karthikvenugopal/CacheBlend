"""Run only after explicit authorization: one pinned unquantized checkpoint, no execution."""
import hashlib
import json
import os
from pathlib import Path
import shutil

ROOT = Path(__file__).resolve().parents[2]
REVISION = '70d244cc86ccca08cf5af4e1e306ecf908b1ad5e'
FILES = ['config.json', 'generation_config.json', 'tokenizer.json', 'tokenizer_config.json',
         'vocab.json', 'merges.txt', 'model.safetensors.index.json',
         'model-00001-of-00002.safetensors', 'model-00002-of-00002.safetensors', 'LICENSE']


def main():
    # Keep cache and downloads inside this workspace; avoid a second Xet chunk cache.
    os.environ['HF_HOME'] = str(ROOT/'.models/.hub')
    os.environ['HF_HUB_DISABLE_XET'] = '1'
    from huggingface_hub import HfApi, snapshot_download
    target = ROOT/'.models/qwen3-1.7b'
    assert not (target/'download-manifest.json').exists(), 'Preserve an existing completed download'
    info = HfApi().model_info('Qwen/Qwen3-1.7B', revision=REVISION, files_metadata=True, token=False)
    assert info.sha == REVISION
    sizes = {s.rfilename: s.size for s in info.siblings if s.rfilename in FILES}
    remote_digests = {s.rfilename: getattr(s.lfs, 'sha256', None)
                      for s in info.siblings if s.rfilename in FILES and s.lfs}
    assert set(sizes) == set(FILES) and all(isinstance(v, int) for v in sizes.values())
    total = sum(sizes.values())
    assert total < 4_200_000_000, 'Remote size exceeds the reviewed bound; stop before payload download'
    assert shutil.disk_usage(ROOT).free > total+2*2**30, 'Insufficient reviewed disk reserve'
    print(json.dumps({'revision': REVISION, 'bytes': total, 'files': sizes}, indent=2), flush=True)
    snapshot_download('Qwen/Qwen3-1.7B', revision=REVISION, local_dir=target,
                      allow_patterns=FILES, token=False, max_workers=1)
    checks = {}
    for name in FILES:
        p = target/name
        assert p.stat().st_size == sizes[name]
        h = hashlib.sha256()
        with p.open('rb') as f:
            for block in iter(lambda: f.read(8*1024*1024), b''):
                h.update(block)
        checks[name] = {'bytes': sizes[name], 'sha256': h.hexdigest()}
        if name in remote_digests and remote_digests[name]:
            assert h.hexdigest() == remote_digests[name], 'Remote SHA-256 mismatch: '+name
    (target/'download-manifest.json').write_text(json.dumps(
        {'repository': 'Qwen/Qwen3-1.7B', 'revision': REVISION, 'files': checks}, indent=2)+'\n')


if __name__ == '__main__':
    main()
