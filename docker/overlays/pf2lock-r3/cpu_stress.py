#!/usr/bin/env python3
"""
pf2lock CPU stress test for the n-gram staging code (exllamav3/modules/ngram_embedding.py), as installed in the image
it runs in (or the file given with --module). CPU only; no GPU, no network. Shims: pf2lock_harness.py.

The race (EXL3_NGRAM_PREFETCH2=1, served daily flag): the prefetch worker's deferred stage (_stage_deferred ->
_acquire_pin) and the issuing thread's forward miss (forward -> _acquire_pin) both choose a free staging set before
either marks it held, so the miss can return rows the worker is overwriting. Related hazards in the same code: the
unlocked _pending list (a retire on one thread vs a hit/append on the other) and entries made visible before their
future exists. Deterministic coverage of the ownership rules is in pf2lock_tests.py; this is regression stress.

Scenarios (fresh module per iteration, created and always shut down by the runner):
  share   prefetch_ids(a) (deferred), forward(b) (miss), forward(a) (hit); asserts exactly 1 hit and 1 miss
  steady  20 decode-like steps: prefetch_ids(h_k), prefetch_ids(spurious_k) (never consumed: retire path),
          forward(miss_k), forward(h_k); hits are not guaranteed by design (a stale-but-finished entry may be
          retired first), so the hit/miss counts are reported, not asserted
  long    PREFETCH2 long prefetch() (deferred, >= PREFETCH_MIN_TOKENS), forward(miss long), forward(hit long);
          asserts 1 hit and 1 miss
  offpath EXL3_NGRAM_PREFETCH2 unset semantics: prefetch(h) (worker stage, issuing-thread acquire), forward(h)
          hit, forward(miss); asserts 1 hit and 1 miss
share, steady and long need EXL3_NGRAM_PREFETCH2=1 at import; offpath needs it unset/0. An explicit scenario that
does not match the flag is refused (exit 2); `all` runs the scenarios that match it.

An iteration fails on: wrong rows, an exception, a hit/miss count other than asserted, a leftover pending entry or
held set after the forwards, more than MAX_PIN_SETS sets, or a hang (watchdog, per-iteration timeout). Exit 0 only
when every iteration passed.

  python3 cpu_stress.py --iterations 5000 [--scenario share|steady|long|offpath|all] [--module PATH]
                        [--grow-sleep-us 0] [--json out]
"""
import argparse
import json
import os
import sys
import threading
import time
import traceback

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import pf2lock_harness as H  # noqa: E402
import torch  # noqa: E402

ne = None
h = H.h


def check_rows(m, history, tag, fails, want = None):
    """forward(history) must return the reference rows; want = "hit"/"miss" also asserts how it was served."""
    before = dict(m.prefetch_stats)
    out = m.forward(history, {})
    if not torch.equal(out.to(torch.float16), H.expected(history)):
        got = out.reshape(-1, ne.ROW_DIM)[:, 0].tolist()
        fails.append(f"wrong rows on {tag}: want {history[0, 1:].tolist()[:8]} got {[int(g) for g in got][:8]}")
    if want is not None and m.prefetch_stats[want] != before[want] + 1:
        fails.append(f"counter: {tag} was not a {want} (stats {before} -> {m.prefetch_stats})")


def check_end(m, fails):
    if m._pending:
        fails.append(f"pending left: {len(m._pending)}")
    held = [p for p in m._pins if p.held]
    if held:
        fails.append(f"held sets left: {len(held)}")
    if len(m._pins) > ne.MAX_PIN_SETS:
        fails.append(f"{len(m._pins)} staging sets > MAX_PIN_SETS {ne.MAX_PIN_SETS}")


def it_share(m, i, fails):
    a, b = h(1, 2, 3, 4), h(5, 6, 7, 8)
    m.prefetch_ids(a)
    check_rows(m, b, "miss b", fails, "miss")
    check_rows(m, a, "hit a", fails, "hit")
    check_end(m, fails)


def it_steady(m, i, fails):
    for k in range(20):
        base = (k * 37 + i) % (H.ROWS - 40)
        hk = h(base, base + 1, base + 2, base + 3)
        spurious = h(base + 10, base + 11, base + 12, base + 13)
        miss = h(base + 20, base + 21, base + 22, base + 23)
        m.prefetch_ids(hk)
        m.prefetch_ids(spurious)
        check_rows(m, miss, f"step {k} miss", fails)
        check_rows(m, hk, f"step {k} hk", fails)
    # whatever is still queued (the spurious prefetches) must retire cleanly
    m._drain_prefetch()
    check_end(m, fails)


def _long(i, shift):
    n = ne.PREFETCH_MIN_TOKENS + 8                      # prefetch() declines decode-sized inputs
    base = (i + shift) % (H.ROWS - n - 2)
    return torch.arange(base, base + n + 1, dtype = torch.int64).view(1, -1)


def it_long(m, i, fails):
    hist, miss = _long(i, 0), _long(i, 97)
    m.prefetch(hist)
    check_rows(m, miss, "long miss", fails, "miss")
    check_rows(m, hist, "long hit", fails, "hit")
    check_end(m, fails)


def it_offpath(m, i, fails):
    hist, miss = _long(i, 0), _long(i, 97)
    m.prefetch(hist)
    check_rows(m, miss, "off miss", fails, "miss")
    check_rows(m, hist, "off hit", fails, "hit")
    check_end(m, fails)


SCENARIOS = {"share": it_share, "steady": it_steady, "long": it_long, "offpath": it_offpath}
NEEDS_PREFETCH2 = {"share": True, "steady": True, "long": True, "offpath": False}


def run(name, fn, iterations, timeout_s):
    counts = {"iterations": 0, "failed": 0, "wrong_rows": 0, "exceptions": 0, "counter": 0, "leftover": 0,
              "hang": 0, "hits": 0, "misses": 0, "retired": 0}
    examples = []
    state = {"i": -1, "t": time.monotonic()}
    stop = threading.Event()

    def watchdog():
        while not stop.wait(1.0):
            if time.monotonic() - state["t"] > timeout_s:
                counts["hang"] += 1
                counts["failed"] += 1
                print(json.dumps({"scenario": name, "hang_at_iteration": state["i"], "counts": counts}), flush = True)
                os._exit(3)

    threading.Thread(target = watchdog, daemon = True).start()
    for i in range(iterations):
        state["i"], state["t"] = i, time.monotonic()
        m = H.make_module(ne)
        fails = []
        try:
            fn(m, i, fails)
        except BaseException as e:  # noqa: BLE001
            fails.append(f"exception {type(e).__name__}: {e}")
            if len(examples) < 3:
                fails.append(traceback.format_exc(limit = 4))
        finally:
            # the module exists before the scenario runs, so its worker is always shut down; a failed iteration
            # may have stranded a set, so never block the run on its worker
            if fails:
                H.shutdown(m)
            elif m._executor is not None:
                m._executor.shutdown(wait = True)
                m._executor = None
        counts["iterations"] += 1
        counts["hits"] += m.prefetch_stats["hit"]
        counts["misses"] += m.prefetch_stats["miss"]
        counts["retired"] += m.prefetch_stats["retired"]
        if fails:
            counts["failed"] += 1
            for key, prefixes in (("wrong_rows", ("wrong rows",)), ("exceptions", ("exception",)),
                                  ("counter", ("counter",)), ("leftover", ("pending", "held", "staging sets"))):
                if any(f.startswith(prefixes) for f in fails):
                    counts[key] += 1
            if len(examples) < 5:
                examples.append({"iteration": i, "fails": fails[:4]})
    stop.set()
    counts["rate"] = counts["failed"] / max(1, counts["iterations"])
    return {"scenario": name, "counts": counts, "examples": examples}


def main():
    global ne
    p = argparse.ArgumentParser(description = __doc__, formatter_class = argparse.RawDescriptionHelpFormatter)
    p.add_argument("--iterations", type = int, default = 5000)
    p.add_argument("--scenario", choices = list(SCENARIOS) + ["all"], default = "all")
    p.add_argument("--module", help = "module file to test instead of the installed one (loaded as "
                                      "exllamav3.modules.ngram_embedding)")
    p.add_argument("--grow-sleep-us", type = int, default = 0,
                   help = "test-side widening of the select->held window (default 0 = natural)")
    p.add_argument("--timeout-s", type = float, default = 30.0, help = "per-iteration watchdog")
    p.add_argument("--json", help = "write the result here too")
    a = p.parse_args()
    ne = H.load_module(a.module)
    H.install_shims(ne, a.grow_sleep_us)
    if a.scenario == "all":
        names = [n for n in SCENARIOS if NEEDS_PREFETCH2[n] == ne.PREFETCH2]
    elif NEEDS_PREFETCH2[a.scenario] != ne.PREFETCH2:
        print(f"cpu_stress: scenario {a.scenario} needs EXL3_NGRAM_PREFETCH2="
              f"{1 if NEEDS_PREFETCH2[a.scenario] else 0} at import; the module has PREFETCH2={ne.PREFETCH2}",
              file = sys.stderr)
        sys.exit(2)
    else:
        names = [a.scenario]
    import hashlib
    with open(ne.__file__, "rb") as f:
        sha = hashlib.sha256(f.read()).hexdigest()
    res = {"module": ne.__file__, "module_sha256": sha, "prefetch2": ne.PREFETCH2, "stubbed": H.STUBBED,
           "grow_sleep_us": a.grow_sleep_us, "has_lock": hasattr(H.make_module(ne), "_lock"),
           "torch": torch.__version__, "scenarios": names, "results": []}
    t0 = time.time()
    for n in names:
        r = run(n, SCENARIOS[n], a.iterations, a.timeout_s)
        res["results"].append(r)
        print(json.dumps(r), flush = True)
    res["seconds"] = round(time.time() - t0, 1)
    res["pass"] = bool(names) and all(r["counts"]["failed"] == 0 for r in res["results"])
    print(json.dumps({k: v for k, v in res.items() if k != "results"}), flush = True)
    if a.json:
        with open(a.json, "w") as f:
            json.dump(res, f, indent = 2)
    sys.stdout.flush()
    os._exit(0 if res["pass"] else 1)    # a leaked worker of a failed iteration must not hold the exit


if __name__ == "__main__":
    main()
