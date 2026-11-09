"""Resume only unfinished cases, without changing the frozen inference source.

The two runtime substitutions below filter case enumeration and checkpoint
writing only. Selectors, prompts, inference, scores and timing stay unchanged.
"""
from copy import deepcopy
import json
from pathlib import Path
from unittest.mock import patch

from kvrepair import feasibility as f

D = Path(__file__).resolve().parents[2] / 'artifacts/feasibility'


def main():
    frozen = json.loads((D/'frozen.json').read_text())
    f.require_gate1(D/'gate1.json')
    assert frozen['source_hashes'] == f.hashes(), 'Frozen inference source changed'
    assert frozen['config'] == f.CONFIG, 'Frozen configuration changed'
    all_cases = f.cases('heldout')
    assert frozen['heldout_dataset_sha256'] == f.fingerprint(all_cases), 'Frozen cases changed'
    partial = json.loads((D/'heldout.json').read_text())
    assert partial['source_hashes'] == frozen['source_hashes']
    assert partial['frozen_manifest_sha256'] == f.fingerprint(frozen)
    assert partial['dataset_sha256'] == frozen['heldout_dataset_sha256']
    expected = {c['id']:c for c in all_cases}
    completed = {c['case']['id']:c for c in partial['cases']}
    assert len(completed) == len(partial['cases']), 'Duplicate checkpoint cases'
    methods = {'full','edit-only','local','attention','delta','random-17','random-29','random-43'}
    for case_id, result in completed.items():
        assert result['case'] == expected[case_id]
        if 'rejected' not in result:
            assert set(result['methods']) == methods
            assert all(len(m['timings']) == f.CONFIG['repeats'] for m in result['methods'].values())
    remaining = [c for c in all_cases if c['id'] not in completed]
    if not remaining:
        print('Pilot already complete; no inference repeated.')
        return
    current_diagnostics = f.diagnostics()
    for package in ['torch','transformers','tokenizers','safetensors']:
        assert current_diagnostics['versions'][package] == partial['diagnostics']['versions'][package], package+' changed'
    backup = D/('heldout-paused-'+f.fingerprint(partial)[:12]+'.json')
    if not backup.exists():
        f.write(backup,partial)
    number = 1
    while (D/f'heldout-resume-{number}.json').exists():
        number += 1
    batch_path = D/f'heldout-resume-{number}.json'
    original_write, original_cases = f.write, f.cases
    original_completed_hash = f.fingerprint(partial['cases'])
    resumed_at = f.now()

    def checkpoint(path, report):
        original_write(path,report)
        if Path(path) != batch_path:
            return
        combined = deepcopy(partial)
        new = {c['case']['id']:c for c in report['cases']}
        assert not (set(new) & set(completed)), 'Completed case was repeated'
        combined['cases'] = [deepcopy(completed[c['id']] if c['id'] in completed else new[c['id']])
                             for c in all_cases if c['id'] in completed or c['id'] in new]
        assert f.fingerprint([c for c in combined['cases'] if c['case']['id'] in completed]) == original_completed_hash
        combined['summary'] = f.summarize(combined)
        combined.pop('completed_utc',None)
        if report.get('completed_utc') and len(combined['cases']) == len(all_cases):
            combined['completed_utc'] = report['completed_utc']
        combined['execution_batches'] = partial.get('execution_batches',[
            {'label':'before-user-pause','started_utc':partial['utc'],
             'case_ids':list(completed),'diagnostics':partial['diagnostics']}]) + [
            {'label':batch_path.stem,'started_utc':resumed_at,'case_ids':list(new),
             'diagnostics':report['diagnostics']}]
        combined['resume_note'] = ('Resumed at independent case boundaries after explicit user authorization. '
                                   'Completed records preserved verbatim; frozen method unchanged. '
                                   'Timing observations span two execution sessions.')
        temporary = D/'heldout.next.json'
        original_write(temporary,combined)
        temporary.replace(D/'heldout.json')
    f.configure(f.CONFIG['seed'],f.CONFIG['threads'])
    model,tokenizer = f.load()
    engine = f.CacheEngine(model)
    print(f'Resuming {len(remaining)} cases; retaining {len(completed)} completed cases.',flush=True)
    with patch.object(f,'cases',side_effect=lambda split: remaining if split=='heldout' else original_cases(split)), \
         patch.object(f,'write',side_effect=checkpoint):
        f.evaluate(engine,tokenizer,'heldout',batch_path,frozen)
    f.write(D/'resume-status.json',{'status':'completed','completed_utc':f.now(),
                                 'preserved_checkpoint':backup.name,'batch':batch_path.name,
                                 'original_completed_records_sha256':original_completed_hash,
                                 'frozen_inference_source_unchanged':f.hashes()==frozen['source_hashes']})


if __name__ == '__main__':
    main()
