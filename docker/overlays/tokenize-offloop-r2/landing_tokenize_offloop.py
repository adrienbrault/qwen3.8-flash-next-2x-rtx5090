#!/usr/bin/env python3
"""Import-time proof that tokenize-offloop r2 landed in the image (CPU; EXL3_* and TABBY_ENCODE_ONCE stripped).

Asserts: the installed exllamav3 tokenizer.py and the /app files hash to SHA256SUMS.exl3.src / SHA256SUMS.app.src;
the REAL packages import (the image's exllamav3 with its extension, /app's TabbyAPI) and carry the changes: the
Tokenizer's per-instance encode lock and encode_batch path, both knobs ON by default and OFF at =0, generate_gen takes
its ids from _encode_prompt before AsyncJob and has no direct prompt encode left, validate_context_length stores the
ids, BaseSamplerRequest declares _prompt_ids (and still loop-think's _loop_backstop), the router awaits both
check_context_length calls through run_tokenize and has no bare one, the reuse is keyed on the tokenizer (weakref), the
installed HF tokenizers exposes the encode_special_tokens getter the conditional write reads and is 0.23.2 (the version
the lock design was measured on). Prints one `tokenize-offloop-r2 landed:` line; exits
non-zero otherwise.
"""
import ast
import hashlib
import importlib.util
import inspect
import os
import sys

for k in list(os.environ):
    if k.startswith("EXL3_") or k == "TABBY_ENCODE_ONCE":
        del os.environ[k]

HERE = os.path.dirname(os.path.abspath(__file__))
APP = os.environ.get("TOK_APP", "/app")
site = os.path.dirname(importlib.util.find_spec("exllamav3").origin)
bad = []
for sums, root in (("SHA256SUMS.exl3.src", site), ("SHA256SUMS.app.src", APP)):
    for line in open(os.path.join(HERE, sums)):
        h, f = line.split()
        got = hashlib.sha256(open(os.path.join(root, f), "rb").read()).hexdigest()
        if got != h:
            bad.append(f"{root}/{f} {got[:12]} != {h[:12]}")
if bad:
    sys.exit("tokenize-offloop-r2 LANDING FAILED: installed files differ from src/: " + "; ".join(bad))

import tokenizers  # noqa: E402
from tokenizers import Tokenizer as HFTokenizer, models  # noqa: E402
import exllamav3.tokenizer.tokenizer as ET  # noqa: E402

sys.path.insert(0, APP)
os.chdir(APP)
import common.tokenize_offloop as TO  # noqa: E402
import backends.exllamav3.model as M  # noqa: E402
from common.sampling import BaseSamplerRequest  # noqa: E402

epb = inspect.getsource(ET.Tokenizer.encode_part_base)
gen = inspect.getsource(M.ExllamaV3Container.generate_gen)
val = inspect.getsource(M.ExllamaV3Container.validate_context_length)
enc = inspect.getsource(M.ExllamaV3Container._encode_prompt)
job_at = gen.index("AsyncJob(")
router = ast.parse(open(os.path.join(APP, "endpoints", "OAI", "router.py")).read())
wrapped = sum(1 for n in ast.walk(router) if isinstance(n, ast.Await) and isinstance(n.value, ast.Call)
              and isinstance(n.value.func, ast.Name) and n.value.func.id == "run_tokenize" and len(n.value.args) >= 3
              and isinstance(n.value.args[0], ast.Call) and getattr(n.value.args[0].func, "id", None) == "prompt_chars"
              and isinstance(n.value.args[1], ast.Attribute) and n.value.args[1].attr == "check_context_length")
bare = sum(1 for n in ast.walk(router) if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute)
           and n.func.attr == "check_context_length")


def knob(fn, key, val):
    os.environ[key] = val
    try:
        return fn()
    finally:
        del os.environ[key]


checks = {
    "exllamav3 revision": ET.TOKENIZE_OFFLOOP_REVISION == "tokenize-offloop-r1",
    "exllamav3 encode lock": "self._encode_lock = threading.Lock()" in inspect.getsource(ET.Tokenizer.__init__)
                             and "with self._encode_lock:" in epb,
    "exllamav3 encode_batch path": "encode_batch([t], add_special_tokens = False)[0].ids" in epb
                                   and "if self.tokenizer.encode_special_tokens != want:" in epb,
    "exllamav3 knob": ET.tokenize_offloop_enabled() and not knob(ET.tokenize_offloop_enabled, "EXL3_TOKENIZE_OFFLOOP", "0"),
    "tabby revision": TO.REVISION == "tokenize-offloop-r2",
    "hop threshold": TO.DEFAULT_MIN_CHARS == 12000 and TO.offload_min_chars() == 12000
                     and knob(TO.offload_min_chars, "EXL3_TOKENIZE_OFFLOOP_MIN_CHARS", "0") == 0
                     and not TO.should_offload(12000) and TO.should_offload(12001) and TO._pending == 0,
    "tabby knobs": TO.offloop_enabled() and TO.encode_once_enabled()
                   and not knob(TO.offloop_enabled, "EXL3_TOKENIZE_OFFLOOP", "0")
                   and not knob(TO.encode_once_enabled, "TABBY_ENCODE_ONCE", "0"),
    "generate_gen ids before AsyncJob": 0 <= gen.find("await self._encode_prompt(") < job_at
                                        and "self.tokenizer.encode(" not in gen[:job_at],
    "_encode_prompt reuse + worker": "run_tokenize(" in enc and "len(prompt)," in enc and "_prompt_ids" in enc and ".clone()" in enc
                                     and "entry[0]() is self.tokenizer" in enc,
    "validate stores ids + tokenizer weakref": "encode_once_enabled()" in val and "_prompt_ids" in val
                                               and "weakref.ref(self.tokenizer)" in val,
    "request private attrs": "_prompt_ids" in BaseSamplerRequest.__private_attributes__
                             and "_loop_backstop" in BaseSamplerRequest.__private_attributes__,
    "router wrapped": wrapped == 2 and bare == 0,
    "hf flag getter": HFTokenizer(models.BPE()).encode_special_tokens is False,
    # the setter-waits-for-encode behaviour the lock design rests on was measured on this version only; another PyO3
    # build could raise "Already borrowed" instead of waiting
    "tokenizers == 0.23.2": tokenizers.__version__ == "0.23.2",
}
failed = [k for k, v in checks.items() if not v]
if failed:
    sys.exit("tokenize-offloop-r2 LANDING FAILED: " + ", ".join(failed))
print(f"tokenize-offloop-r2 landed: {len(checks)} checks; exllamav3 at {site}, /app at {APP}; tokenizers "
      f"{tokenizers.__version__}; knobs TABBY_ENCODE_ONCE / EXL3_TOKENIZE_OFFLOOP default ON, "
      f"EXL3_TOKENIZE_OFFLOOP_MIN_CHARS default {TO.DEFAULT_MIN_CHARS}")
