import argparse
from dataclasses import asdict
import json
from pathlib import Path
import re
from time import perf_counter

import torch

from .engine import CacheEngine, Edit, UnsupportedEdit, logit_metrics
from .models import MODEL, REVISION, configure, diagnostics, load


def prompt(document, question, tokenizer, toy=False):
    content = f"Use only this document to answer the question.\nDocument:\n{document}\nQuestion: {question}"
    if not toy:
        return tokenizer.apply_chat_template([{"role": "user", "content": content}], tokenize=False, add_generation_prompt=True)
    return content + "\nAnswer:"


def correct(text, answers):
    # Strict short-answer score; keep raw outputs for manual assessment of formatting failures.
    def norm(s):
        return re.sub(r"[^\w\s]", "", s.lower()).strip()
    return norm(text) in {norm(a) for a in answers}


def agreement(a, b):
    first = next((i for i, (x, y) in enumerate(zip(a, b)) if x != y), None)
    if first is None and len(a) != len(b):
        first = min(len(a), len(b))
    return {"greedy_equal": a == b, "first_divergent_step": first}


def divergence_probe(engine, reference, candidate, tokens):
    """Compare logits under the same full-recompute continuation, avoiding path confounding."""
    a, b = reference.clone(), candidate.clone()
    trajectory = []
    for i, token in enumerate(tokens):
        metrics = logit_metrics(a.logits, b.logits)
        trajectory.append({"step": i, **metrics})
        if i < len(tokens) - 1:
            akv, alog = engine._forward((token,), a.kv)
            bkv, blog = engine._forward((token,), b.kv)
            a = type(a)(a.ids + (token,), akv, alog, a.owner)
            b = type(b)(b.ids + (token,), bkv, blog, b.owner)
    return trajectory


def demo(args):
    configure(args.seed, args.threads)
    model, tokenizer = load(args.model_path, args.device, args.toy)
    engine = CacheEngine(model)
    report = {"config": vars(args), "diagnostics": diagnostics(),
              "answer_scoring": "Strict normalized short-answer equality; inspect factual content separately. Verbose/truncated answers may fail this check without stating a wrong fact.",
              "model": "random-tiny-llama-SMOKE-ONLY" if args.toy else args.model_path or MODEL,
              "revision": None if args.toy or args.model_path else REVISION,
              "dtype": str(next(model.parameters()).dtype), "attention": model.config._attn_implementation,
              "tolerance": {"atol": 2e-4, "rtol": 2e-4},
              "parameter_bytes": sum(p.numel() * p.element_size() for p in model.parameters()),
              "timing_note": "Single CPU/MPS functional measurements, fixed method order; not a speed benchmark. Tokenization excluded; snapshot copy/splice/sync included. Original prefill reported separately.",
              "cases": []}
    cases = json.loads(Path(args.cases).read_text())
    # Warmup is excluded from reported timings.
    engine.run(tokenizer.encode("Warmup.", add_special_tokens=True))
    for case in cases:
        old_text = prompt(case["old"], case["question"], tokenizer, args.toy)
        new_text = prompt(case["new"], case["question"], tokenizer, args.toy)
        # Always tokenize the COMPLETE rendered prompts, independently.
        a = tuple(tokenizer.encode(old_text, add_special_tokens=False))
        b = tuple(tokenizer.encode(new_text, add_special_tokens=False))
        item = {"case": case, "old_prompt": old_text, "new_prompt": new_text,
                "old_ids": a, "new_ids": b}
        try:
            edit = Edit.between(a, b)
        except UnsupportedEdit as error:
            item["rejected"] = str(error)
            report["cases"].append(item)
            print(case["id"], "REJECTED:", error)
            continue
        item["edit"] = asdict(edit)
        item["affected_tokens"] = [{"position": i, "old_id": a[i], "new_id": b[i],
                                    "old_piece": tokenizer.decode([a[i]]), "new_piece": tokenizer.decode([b[i]])}
                                   for i in range(edit.prefix, edit.end)]
        old_run = engine.run(a)
        item["original_prefill_ms"] = old_run.elapsed_ms
        item["kv_bytes_per_snapshot"] = old_run.state.nbytes
        original = engine.generate(old_run.state, args.max_new_tokens, tokenizer.eos_token_id)
        old_answer = tokenizer.decode(original, skip_special_tokens=True)
        item["original_answer"] = old_answer
        item["original_hit_token_limit"] = len(original) == args.max_new_tokens and bool(original) and original[-1] != tokenizer.eos_token_id
        item["original_baseline_correct"] = None if args.toy else correct(old_answer, case["old_answers"])
        full = engine.run(b)
        modes = {"full": full, "prefix": engine.run(b, "prefix", old_run.state)}
        if args.include_naive:
            modes["naive"] = engine.run(b, "naive", old_run.state)
        item["methods"] = {}
        continuations = {}
        for mode, run in modes.items():
            engine.sync()
            begin = perf_counter()
            tokens = engine.generate(run.state, args.max_new_tokens, tokenizer.eos_token_id)
            engine.sync()
            decode_ms = (perf_counter() - begin) * 1000
            continuations[mode] = tokens
            answer = tokenizer.decode(tokens, skip_special_tokens=True)
            item["methods"][mode] = {
                "approximate": mode == "naive", "reused_ranges": run.reused,
                "recomputed_ranges": run.recomputed, "prefill_tokens_recomputed": run.tokens_recomputed,
                "prefill_ms_including_copy": run.elapsed_ms, "continuation_ms_including_copy": decode_ms,
                "edit_to_answer_ms": run.elapsed_ms + decode_ms,
                "answer": answer, "generated_ids": tokens,
                "hit_token_limit": len(tokens) == args.max_new_tokens and bool(tokens) and tokens[-1] != tokenizer.eos_token_id,
                "answer_correct": None if args.toy else correct(answer, case["answers"]),
                "logits_vs_full": logit_metrics(full.state.logits, run.state.logits),
            }
        for mode, result in item["methods"].items():
            result.update(agreement(continuations["full"], continuations[mode]))
            full_ok = item["methods"]["full"]["answer_correct"]
            result["reuse_introduced_failure"] = None if args.toy or not full_ok else not result["answer_correct"]
            if not result["greedy_equal"]:
                result["teacher_forced_divergence_investigation"] = divergence_probe(
                    engine, full.state, modes[mode].state, continuations["full"])
        # Exact baseline divergence fails loudly; approximate divergence is evidence.
        exact = item["methods"]["prefix"]
        item["exact_check_passed"] = exact["logits_vs_full"]["within_tolerance"] and exact["greedy_equal"]
        report["cases"].append(item)
        print(case["id"], f"edit [{edit.prefix},{edit.end}) / {len(b)} tokens")
        for mode, result in item["methods"].items():
            print(f"  {mode}: {result['answer']!r}; max logit delta={result['logits_vs_full']['max_abs']:.3g}; recomputed={result['prefill_tokens_recomputed']}")
    accepted = [c for c in report["cases"] if "rejected" not in c]
    report["accepted_cases"] = len(accepted)
    report["rejected_cases"] = len(report["cases"]) - len(accepted)
    report["all_exact_checks_passed"] = all(c["exact_check_passed"] for c in accepted) if accepted else None
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2) + "\n")
    print("Report:", out)
    return 2 if not accepted else (0 if report["all_exact_checks_passed"] else 1)


def main():
    parser = argparse.ArgumentParser(description="Offline, equal-token-length KV edit experiments")
    subs = parser.add_subparsers(dest="command", required=True)
    subs.add_parser("diagnose")
    report_parser = subs.add_parser("report")
    report_parser.add_argument("source")
    report_parser.add_argument("destination")
    p = subs.add_parser("demo")
    p.add_argument("--cases", default="examples/development.json")
    p.add_argument("--output", default="artifacts/demo.json")
    p.add_argument("--model-path", help="Existing local unquantized Llama checkpoint; never downloaded")
    p.add_argument("--device", choices=["cpu", "mps"], default="cpu")
    p.add_argument("--toy", action="store_true", help="Random tiny model: functional smoke only")
    p.add_argument("--include-naive", action="store_true", help="Explicit opt-in to incorrect approximate suffix reuse")
    p.add_argument("--seed", type=int, default=17)
    p.add_argument("--threads", type=int, default=4)
    p.add_argument("--max-new-tokens", type=int, default=12)
    args = parser.parse_args()
    if args.command == "report":
        from .report import render
        render(args.source, args.destination)
        return 0
    if args.command == "diagnose":
        print(json.dumps(diagnostics(), indent=2))
        return 0
    return demo(args)
