"""Small static report; raw JSON remains the authoritative experiment record."""
import json
from pathlib import Path


def render(source, destination):
    data = json.loads(Path(source).read_text())
    lines = ["# Development observations", "",
             f"Model: `{data['model']}`; {data['dtype']}; {data['config']['device']}; seed {data['config']['seed']}.", "",
             "These are development examples, not evidence of general repair quality. "
             "`answer_correct` in JSON is strict normalized short-answer matching. "
             "A false score may reflect verbosity rather than a wrong fact. Inspect raw answers below. "
             "Logit drift, continuation agreement and factual correctness measure different things.", "",
             "Ranges below are zero-based, half-open indices in the complete rendered prompt, including chat tokens. "
             "Times are single functional measurements including cache copying; they are not speedup estimates.", ""]
    for c in data['cases']:
        case = c['case']
        lines += [f"## {case['id']}", "", f"Old: {case['old']}", "", f"Edited: {case['new']}", "",
                  f"Question: {case['question']}", "", f"Expected: {case['answers']}; should change: {case['should_change']}.", ""]
        if 'rejected' in c:
            lines += [f"Rejected: {c['rejected']}", ""]
            continue
        lines += [f"Original full-recompute answer: `{c['original_answer']!r}`", "",
                  f"Actual changed envelope: [{c['edit']['prefix']}, {c['edit']['end']}); prompt length {c['edit']['length']}.", "",
                  "| Position | Old ID / decoded piece | New ID / decoded piece |",
                  "|---|---|---|"]
        for t in c['affected_tokens']:
            lines.append(f"| {t['position']} | {t['old_id']} / {t['old_piece']!r} | {t['new_id']} / {t['new_piece']!r} |")
        lines += ["", "| Method | Answer | Reused | Recomputed | Max logit delta | KL(full‖method) | Same greedy tokens | Prefill ms | Through answer ms |",
                  "|---|---|---|---|---|---|---|---|---|"]
        for mode, m in c['methods'].items():
            answer = repr(m['answer']).replace('|', '\\|')
            lines.append(f"| {mode}{' (approximate)' if m['approximate'] else ''} | {answer} | {m['reused_ranges']} | {m['recomputed_ranges']} | {m['logits_vs_full']['max_abs']:.6g} | {m['logits_vs_full']['kl_full_to_candidate']:.6g} | {m['greedy_equal']} | {m['prefill_ms_including_copy']:.1f} | {m['edit_to_answer_ms']:.1f} |")
        lines += [""]
    Path(destination).write_text('\n'.join(lines) + '\n')
