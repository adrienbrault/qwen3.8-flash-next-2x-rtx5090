#!/usr/bin/env python3
"""Id identity: the patched exllamav3 Tokenizer (encode_batch path, EXL3_TOKENIZE_OFFLOOP unset) returns exactly the
ids of the served tokenizer.py for every corpus string, for every combination TabbyAPI and exllamav3 use.

  test_identity.py [--tokenizer-dir DIR ...]

Always runs on the synthetic Qwen-shaped tokenizer (tests/_load.py: special + non-special added tokens, a config-only
non-special token -> the unspecial split loop, a config-only special token -> the missing-special split); each
--tokenizer-dir (a model dir with tokenizer.json [+ tokenizer_config.json, config.json]) runs the same checks on a real
tokenizer (run_offline.sh passes the Qwen3.8-27B tokenizer from the HF cache when present; on flan,
probes/tokenize_offloop_check.py runs them inside the served container against the Flash-Next checkpoint).

Checks, per tokenizer:
  raw     HF encode(t).ids == encode_batch([t])[0].ids with encode_special_tokens False and True
  wrapper patched Tokenizer.encode(t, add_bos, encode_special_tokens) == served Tokenizer.encode(...) for
          encode_special_tokens in (True, False) x add_bos in (False, True), the tensor's dtype and shape included;
          the knob-off path (EXL3_TOKENIZE_OFFLOOP=0) of the patched file too; list input (padded batch + offsets)
  flag    the HF flag state after a call sequence (True, False, True, False ...) matches the served file's, in both
          orders, and ids stay identical across mode switches
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import _load  # noqa: E402

FAILS = []
CHECKS = [0]


def check(cond, what):
    CHECKS[0] += 1
    if not cond:
        FAILS.append(what)
        print("FAIL:", what)


def run(model_dir, label):
    os.environ.pop("EXL3_TOKENIZE_OFFLOOP", None)
    src = _load.load_tokenizer_module("src")
    base = _load.load_tokenizer_module("base")
    cfg_s, cfg_b = _load.make_config(model_dir), _load.make_config(model_dir)
    ts, tb = src.Tokenizer(cfg_s), base.Tokenizer(cfg_b)
    check(src.TOKENIZE_OFFLOOP_REVISION == "tokenize-offloop-r1", f"{label}: revision")
    check(not hasattr(base, "TOKENIZE_OFFLOOP_REVISION"), f"{label}: base module is the served file")
    check(ts.unspecial_piece_to_id == tb.unspecial_piece_to_id
          and ts.missing_special_piece_to_id == tb.missing_special_piece_to_id
          and ts.extended_piece_to_id == tb.extended_piece_to_id, f"{label}: wrapper state after __init__")
    texts = _load.corpus(seed=7)
    hf = ts.tokenizer
    n_raw = 0
    for flag in (False, True):
        hf.encode_special_tokens = flag
        for t in texts:
            a = hf.encode(t, add_special_tokens=False).ids
            b = hf.encode_batch([t], add_special_tokens=False)[0].ids
            n_raw += 1
            check(a == b, f"{label}: raw encode != encode_batch (flag {flag}) for {t[:40]!r}")
    hf.encode_special_tokens = False
    n_wrap = 0
    total_ids = 0
    for knob in (None, "0"):
        if knob is None:
            os.environ.pop("EXL3_TOKENIZE_OFFLOOP", None)
        else:
            os.environ["EXL3_TOKENIZE_OFFLOOP"] = knob
        for special in (True, False):
            for add_bos in (False, True):
                for t in texts:
                    a = ts.encode(t, add_bos=add_bos, encode_special_tokens=special)
                    b = tb.encode(t, add_bos=add_bos, encode_special_tokens=special)
                    n_wrap += 1
                    total_ids += a.shape[-1]
                    check(a.dtype == b.dtype and tuple(a.shape) == tuple(b.shape) and bool((a == b).all()),
                          f"{label}: wrapper ids differ (knob {knob}, special {special}, bos {add_bos}) for {t[:40]!r}")
                    check(ts.tokenizer.encode_special_tokens == tb.tokenizer.encode_special_tokens,
                          f"{label}: flag state differs after encode (special {special})")
        a, oa = ts.encode(texts[:12], encode_special_tokens=True, return_offsets=True)
        b, ob = tb.encode(texts[:12], encode_special_tokens=True, return_offsets=True)
        check(tuple(a.shape) == tuple(b.shape) and bool((a == b).all()) and bool((oa == ob).all()),
              f"{label}: list input (padded batch) differs (knob {knob})")
    os.environ.pop("EXL3_TOKENIZE_OFFLOOP", None)
    # flag persistence across alternating modes, both orders, on fresh instances
    for order in ((True, False) * 3, (False, True) * 3):
        ts2, tb2 = src.Tokenizer(_load.make_config(model_dir)), base.Tokenizer(_load.make_config(model_dir))
        for special in order:
            for t in texts[15:23]:
                a = ts2.encode(t, encode_special_tokens=special)
                b = tb2.encode(t, encode_special_tokens=special)
                check(tuple(a.shape) == tuple(b.shape) and bool((a == b).all()), f"{label}: alternating modes differ")
            check(ts2.tokenizer.encode_special_tokens == tb2.tokenizer.encode_special_tokens == (not special),
                  f"{label}: flag after special={special} is not `not special`")
        # a direct HF call after the sequence sees the same flag as on the served file (num_tokens uses it)
        check(ts2.num_tokens(texts[16]) == tb2.num_tokens(texts[16]), f"{label}: num_tokens after the sequence")
    print(f"{label}: raw {n_raw} pairs, wrapper {n_wrap} pairs ({total_ids} ids), vocab {ts.tokenizer.get_vocab_size()}, "
          f"unspecial {len(ts.unspecial_piece_to_id)}, missing-special {len(ts.missing_special_piece_to_id)}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tokenizer-dir", action="append", default=[])
    a = ap.parse_args()
    synth = _load.build_synth_tokenizer()
    run(synth, "synthetic")
    for d in a.tokenizer_dir:
        run(d, os.path.basename(os.path.normpath(d)) or d)
    print(f"test_identity: {CHECKS[0] - len(FAILS)} passed, {len(FAILS)} failed")
    return 1 if FAILS else 0


if __name__ == "__main__":
    sys.exit(main())
