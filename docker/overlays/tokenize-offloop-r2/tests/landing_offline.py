#!/usr/bin/env python3
"""Runs landing_tokenize_offloop.py outside the image: the stub exllamav3 package of tests/_load.py stands in for the
installed one (its origin = TOK_EXL3, so the hash check reads the patched scratch tree) and the patched tokenizer
module is registered as exllamav3.tokenizer.tokenizer. Everything else the landing checks is the real code."""
import os
import runpy
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import _load  # noqa: E402

pkg = _load.install_exl3_stubs()
pkg.__spec__.origin = os.path.join(_load.exl3_dir(), "__init__.py")
tm = _load.load_tokenizer_module("src")
sys.modules["exllamav3.tokenizer.tokenizer"] = tm
sys.modules["exllamav3.tokenizer"].tokenizer = tm
pkg.Tokenizer = tm.Tokenizer
os.environ.setdefault("TOK_APP", _load.app_dir())
runpy.run_path(os.path.join(_load.OVERLAY, "landing_tokenize_offloop.py"), run_name="__main__")
