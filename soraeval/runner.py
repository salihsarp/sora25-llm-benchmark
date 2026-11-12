"""Command-line runner.

  python -m soraeval.runner generate --n 3000 --seed 2026 --out data/scenarios.jsonl
  python -m soraeval.runner run   --model claude-haiku-5-5 --n 60 --out results/pilot.jsonl --budget 2
  python -m soraeval.runner batch-submit  --model claude-sonnet-5-5 --n 600 --out results/sonnet.jsonl
  python -m soraeval.runner batch-collect --out results/sonnet.jsonl
  python -m soraeval.runner run --provider openai --base-url http://localhost:11434/v1 --model qwen3:8b ...
  python -m soraeval.runner run --provider ollama --model gpt-oss:20b --think medium ...   (Amendment 2)

Results are appended as JSON lines; finished episodes are skipped, so every command is resumable.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

from . import prompts
from .generator import generate, save_jsonl, load_jsonl
from .parse import extract_json, normalise_answer, calculator_answer
from .providers import Anthropic, Ollama, OpenAICompat, cost_usd, NO_TEMPERATURE

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_SCEN = os.path.join(HERE, "data", "scenarios.jsonl")


def ep_key(sid, lang, cond, model, rep):
    return f"{sid}|{lang}|{cond}|{model}|{rep}"


def done_keys(path):
    keys = set()
    if os.path.exists(path):
        with open(path, encoding="utf-8") as fh:
            for line in fh:
                try:
                    o = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if not o.get("error"):
                    keys.add(o["key"])
    return keys


def spent(path):
    tot = 0.0
    if os.path.exists(path):
        with open(path, encoding="utf-8") as fh:
            for line in fh:
                try:
                    tot += json.loads(line).get("cost_usd", 0.0)
                except json.JSONDecodeError:
                    pass
    return tot


def make_record(sc, lang, cond, model, rep, resp, batch=False, settings=None):
    rec = {"key": ep_key(sc["id"], lang, cond, model, rep), "sid": sc["id"], "lang": lang, "condition": cond,
           "model": model, "rep": rep, "raw": resp["text"], "usage": resp["usage"],
           "stop_reason": resp["stop_reason"], "latency_s": resp["latency_s"], "error": resp["error"],
           "cost_usd": round(cost_usd(model, resp["usage"], batch=batch), 6), "batch": batch, "ts": time.time(),
           "settings": dict(settings or {})}
    if "thinking" in resp:
        rec["thinking"] = resp["thinking"]
    if "temperature" in rec["settings"]:
        rec["settings"]["temperature"] = resp.get("temperature_used", None if model in NO_TEMPERATURE else
                                                  rec["settings"]["temperature"])
    obj = extract_json(resp["text"])
    if cond == "extract":
        rec["params"] = obj if isinstance(obj, dict) else None
        rec["answer"] = calculator_answer(obj)
    else:
        rec["answer"] = normalise_answer(obj)
    t = sc["truth"]
    a = rec["answer"]
    rec["sail_correct"] = a.get("sail") == t["sail"]
    return rec


def plan(scs, langs, conds, model, reps, out):
    done = done_keys(out)
    todo = []
    for sc in scs:
        for rep in range(reps):
            for lang in langs:
                for cond in conds:
                    if ep_key(sc["id"], lang, cond, model, rep) not in done:
                        todo.append((sc, lang, cond, rep))
    return todo


def parse_think(s):
    """--think: '' = do not send; true/false; or a level (low/medium/high, gpt-oss)."""
    return {"": None, "true": True, "false": False}.get(s.lower(), s)


def client_for(args):
    if args.provider == "anthropic":
        return Anthropic(args.model, temperature=args.temperature, max_tokens=args.max_tokens, effort=args.effort or None)
    if args.provider == "ollama":
        return Ollama(args.model, args.base_url or "http://127.0.0.1:11437", max_tokens=args.max_tokens,
                      num_ctx=args.num_ctx, seed=args.seed, think=parse_think(args.think))
    return OpenAICompat(args.model, args.base_url or "http://localhost:11434/v1", temperature=args.temperature,
                        max_tokens=args.max_tokens)


def run_settings(args, cli):
    if args.provider == "ollama":
        return {"provider": "ollama", "max_tokens": args.max_tokens, "num_ctx": args.num_ctx, "seed": args.seed,
                "think": parse_think(args.think), "temperature": None, "sampling": "model default",
                "digest": cli.digest}
    return {"effort": args.effort, "max_tokens": args.max_tokens, "temperature": args.temperature}


def load_subset(args):
    scs = load_jsonl(args.scenarios)
    return scs[args.offset: args.offset + args.n]


def cmd_generate(args):
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    scs = generate(args.n, seed=args.seed)
    save_jsonl(scs, args.out)
    print(f"wrote {len(scs)} scenarios to {args.out}")


def cmd_run(args):
    scs = load_subset(args)
    langs, conds = args.langs.split(","), args.conditions.split(",")
    todo = plan(scs, langs, conds, args.model, args.reps, args.out)
    already = spent(args.out)
    print(f"{len(todo)} episodes to run; already spent ${already:.3f} in {args.out}; budget ${args.budget}")
    if not todo:
        return
    cli = client_for(args)
    settings = run_settings(args, cli)
    lock = threading.Lock()
    state = {"cost": already, "n": 0, "ok": 0, "err": 0, "stop": False}
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    fh = open(args.out, "a", encoding="utf-8")

    def work(item):
        sc, lang, cond, rep = item
        if state["stop"]:
            return
        system, user = prompts.build(cond, lang, sc[f"text_{lang}"])
        resp = cli.complete(system, user)
        rec = make_record(sc, lang, cond, args.model, rep, resp, settings=settings)
        with lock:
            fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
            fh.flush()
            state["cost"] += rec["cost_usd"]
            state["n"] += 1
            state["ok"] += int(rec["sail_correct"])
            state["err"] += int(bool(rec["error"]))
            if state["cost"] >= args.budget:
                state["stop"] = True
            if state["n"] % 25 == 0 or state["n"] == len(todo):
                print(f"  {state['n']}/{len(todo)}  SAIL correct {state['ok']}/{state['n']}  errors {state['err']}"
                      f"  spent ${state['cost']:.3f}", flush=True)

    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        futs = [ex.submit(work, it) for it in todo]
        for _ in as_completed(futs):
            pass
    fh.close()
    if state["stop"]:
        print(f"STOPPED at budget cap ${args.budget}")
    print(f"done: {state['n']} episodes, SAIL correct {state['ok']}, errors {state['err']}, total spent ${state['cost']:.3f}")


def _batch_file(out):
    return out + ".batches.json"


def cmd_batch_submit(args):
    scs = load_subset(args)
    langs, conds = args.langs.split(","), args.conditions.split(",")
    todo = plan(scs, langs, conds, args.model, args.reps, args.out)
    bf = _batch_file(args.out)
    pending = json.load(open(bf)) if os.path.exists(bf) else []
    inflight = {cid for b in pending if not b.get("collected") for cid in b["ids"]}
    items, ids = [], []
    for sc, lang, cond, rep in todo:
        cid = f"{sc['id']}-{lang}-{cond}-{rep}"
        if cid in inflight:
            continue
        system, user = prompts.build(cond, lang, sc[f"text_{lang}"])
        items.append((cid, system, user))
        ids.append(cid)
    if not items:
        print("nothing to submit")
        return
    est = len(items) * args.est_cost_per_call
    print(f"{len(items)} requests; rough estimate ${est:.2f} (batch price)")
    if est > args.budget:
        print(f"estimate exceeds budget ${args.budget}; use --n/--conditions to shrink or raise --budget")
        return
    cli = client_for(args)
    chunk = 5000
    for i in range(0, len(items), chunk):
        bid = cli.batch_submit(items[i:i + chunk])
        pending.append({"batch_id": bid, "model": args.model, "ids": ids[i:i + chunk], "submitted": time.time(),
                        "settings": {"effort": args.effort, "max_tokens": args.max_tokens, "temperature": args.temperature},
                        "scenarios": args.scenarios, "collected": False})
        print(f"submitted batch {bid} ({len(items[i:i + chunk])} requests)")
        json.dump(pending, open(bf, "w"), indent=1)


def cmd_batch_collect(args):
    bf = _batch_file(args.out)
    if not os.path.exists(bf):
        print("no batches recorded")
        return
    pending = json.load(open(bf))
    scs = {s["id"]: s for s in load_jsonl(args.scenarios)}
    while True:
        open_batches = [b for b in pending if not b["collected"]]
        if not open_batches:
            print("all batches collected")
            return
        for b in open_batches:
            cli = Anthropic(b["model"], max_tokens=args.max_tokens, effort=args.effort or None)
            st = cli.batch_status(b["batch_id"])
            print(f"{b['batch_id']}: {st.get('processing_status')} {st.get('request_counts')}")
            if st.get("processing_status") != "ended":
                continue
            res = cli.batch_results(b["batch_id"])
            with open(args.out, "a", encoding="utf-8") as fh:
                for cid, resp in res.items():
                    sid, lang, rest = cid.split("-", 2)
                    cond, rep = rest.rsplit("-", 1)
                    rec = make_record(scs[sid], lang, cond, b["model"], int(rep), resp, batch=True,
                                      settings=b.get("settings", {}))
                    fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
            b["collected"] = True
            json.dump(pending, open(bf, "w"), indent=1)
            print(f"collected {len(res)} results; total spent in {args.out}: ${spent(args.out):.3f}")
        if not args.wait:
            return
        time.sleep(60)


def main(argv=None):
    ap = argparse.ArgumentParser(prog="soraeval")
    sub = ap.add_subparsers(dest="cmd", required=True)
    g = sub.add_parser("generate")
    g.add_argument("--n", type=int, default=3000)
    g.add_argument("--seed", type=int, default=2026)
    g.add_argument("--out", default=DEFAULT_SCEN)
    for name in ("run", "batch-submit", "batch-collect"):
        p = sub.add_parser(name)
        p.add_argument("--scenarios", default=DEFAULT_SCEN)
        p.add_argument("--out", required=True)
        p.add_argument("--model", default="claude-haiku-5-5")
        p.add_argument("--provider", choices=["anthropic", "openai", "ollama"], default="anthropic")
        p.add_argument("--base-url", default=None, help="default: 127.0.0.1:11437 (ollama), localhost:11434/v1 (openai)")
        p.add_argument("--think", default="", help="ollama: '' = not sent, true/false, or low/medium/high")
        p.add_argument("--num-ctx", type=int, default=40960)
        p.add_argument("--seed", type=int, default=2026)
        p.add_argument("--n", type=int, default=60)
        p.add_argument("--offset", type=int, default=0)
        p.add_argument("--langs", default="en,tr")
        p.add_argument("--conditions", default=",".join(prompts.CONDITIONS))
        p.add_argument("--reps", type=int, default=1)
        p.add_argument("--temperature", type=float, default=0.0)
        p.add_argument("--max-tokens", type=int, default=32000)
        p.add_argument("--effort", default="medium", help="output_config.effort (low/medium/high/xhigh/max); '' = model default")
        p.add_argument("--budget", type=float, default=5.0)
        p.add_argument("--workers", type=int, default=4)
        p.add_argument("--est-cost-per-call", type=float, default=0.01)
        p.add_argument("--wait", action="store_true")
    args = ap.parse_args(argv)
    {"generate": cmd_generate, "run": cmd_run, "batch-submit": cmd_batch_submit,
     "batch-collect": cmd_batch_collect}[args.cmd](args)


if __name__ == "__main__":
    main()
