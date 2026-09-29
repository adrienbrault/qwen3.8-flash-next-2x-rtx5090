"""Shared helpers for the tokenize-offloop r2 tests (CPU only: no GPU, no exllamav3 extension, no model).

Environment (tests/run_offline.sh sets these on the build host; the Dockerfile sets them in the image):
  TOK_EXL3      directory that IS the patched exllamav3 package (site-packages/exllamav3 in the image)
  TOK_EXL3_BASE the served (unpatched) tokenizer.py, for identity comparisons (base/exllamav3/tokenizer/tokenizer.py)
  TOK_APP       the patched TabbyAPI tree (/app in the image)

The exllamav3 tokenizer module is loaded from its file under a stub `exllamav3` package: exllamav3.util (pure Python +
torch) is the real one, every other exllamav3.* import resolves to an inert stub, so nothing needs the CUDA extension.
The base and the patched tokenizer.py load side by side as exllamav3.tokenizer.tokenizer_base / _src.
"""
import importlib.abc
import importlib.machinery
import importlib.util
import json
import os
import random
import sys
import tempfile
import types

sys.dont_write_bytecode = True
os.environ.setdefault("KMP_WARNINGS", "0")

HERE = os.path.dirname(os.path.abspath(__file__))
OVERLAY = os.path.dirname(HERE)


def _env_path(key, default):
    p = os.environ.get(key) or default
    if not p or not os.path.exists(p):
        sys.exit(f"{key}: {p!r} does not exist (see tests/_load.py)")
    return os.path.abspath(p)


def exl3_dir():
    return _env_path("TOK_EXL3", None)


def exl3_base_file():
    return _env_path("TOK_EXL3_BASE", os.path.join(OVERLAY, "base", "exllamav3", "tokenizer", "tokenizer.py"))


def app_dir():
    return _env_path("TOK_APP", None)


# ------------------------------------------------------------------------------------------------ exllamav3 stubs
class _Dummy:
    """Inert stand-in for any exllamav3 class the TabbyAPI imports name (never called on the tested path)."""

    def __init__(self, *a, **k):
        pass

    def __getattr__(self, name):
        return _Dummy()

    def __call__(self, *a, **k):
        return _Dummy()


def _stub_module(name, path=None):
    m = types.ModuleType(name)
    m.__path__ = list(path or [])
    m.__file__ = f"<stub {name}>"
    m.__spec__ = importlib.machinery.ModuleSpec(name, None, is_package=True)
    m.__spec__.submodule_search_locations = m.__path__
    cache = {}

    def __getattr__(attr):
        if attr.startswith("__"):
            raise AttributeError(attr)
        if attr not in cache:
            cache[attr] = type(attr, (_Dummy,), {})
        return cache[attr]

    m.__getattr__ = __getattr__
    return m


class _StubFinder(importlib.abc.MetaPathFinder, importlib.abc.Loader):
    """Claims every exllamav3.* module except exllamav3.util[.*] (loaded from the real package)."""

    def find_spec(self, fullname, path=None, target=None):
        if not fullname.startswith("exllamav3."):
            return None
        if fullname == "exllamav3.util" or fullname.startswith("exllamav3.util."):
            return None
        return importlib.machinery.ModuleSpec(fullname, self, is_package=True)

    def create_module(self, spec):
        return _stub_module(spec.name)

    def exec_module(self, module):
        pass


_installed = False


def install_exl3_stubs():
    global _installed
    if _installed:
        return sys.modules["exllamav3"]
    pkg_dir = exl3_dir()
    sys.meta_path.insert(0, _StubFinder())
    pkg = _stub_module("exllamav3", [pkg_dir])
    sys.modules["exllamav3"] = pkg
    tok_pkg = _stub_module("exllamav3.tokenizer", [os.path.join(pkg_dir, "tokenizer")])
    sys.modules["exllamav3.tokenizer"] = tok_pkg
    pkg.tokenizer = tok_pkg
    cfg = _stub_module("exllamav3.model.config")
    cfg.Config = object
    sys.modules["exllamav3.model"] = _stub_module("exllamav3.model")
    sys.modules["exllamav3.model.config"] = cfg
    _installed = True
    return pkg


def load_tokenizer_module(which):
    """which = 'src' (the patched package's tokenizer.py) or 'base' (the served file)."""
    install_exl3_stubs()
    name = f"exllamav3.tokenizer.tokenizer_{which}"
    if name in sys.modules:
        return sys.modules[name]
    path = os.path.join(exl3_dir(), "tokenizer", "tokenizer.py") if which == "src" else exl3_base_file()
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def make_config(model_dir):
    """The attributes exllamav3's Tokenizer reads from a Config (config.json's token ids)."""
    c = json.load(open(os.path.join(model_dir, "config.json"))) if os.path.exists(os.path.join(model_dir, "config.json")) else {}
    tc = c.get("text_config", {})

    def pick(k):
        v = c.get(k, tc.get(k))
        return v

    eos = pick("eos_token_id")
    eos_list = list(eos) if isinstance(eos, list) else ([eos] if eos is not None else [])
    return types.SimpleNamespace(
        directory=model_dir,
        eos_token_id=eos_list[0] if eos_list else None,
        eos_token_id_list=eos_list,
        bos_token_id=pick("bos_token_id"),
        pad_token_id=pick("pad_token_id"),
    )


# ------------------------------------------------------------------------------------------------ synthetic tokenizer
QWEN_SPLIT = (r"(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?[\p{L}\p{M}]+|\p{N}| ?[^\s\p{L}\p{M}\p{N}]+[\r\n]*"
              r"|\s*[\r\n]+|\s+(?!\S)|\s+")
SPECIALS = ["<|endoftext|>", "<|im_start|>", "<|im_end|>", "<|vision_start|>"]
UNSPECIAL_KNOWN = ["<think>", "</think>", "<tool_call>", "</tool_call>"]
UNSPECIAL_UNKNOWN = "<|unspecial_x|>"     # in tokenizer_config.json only: exllamav3's unspecial split path
MISSING_SPECIAL = "<|missing_special|>"  # special in tokenizer_config.json only: exllamav3's missing-special path


def build_synth_tokenizer(out_dir=None, vocab_size=3000):
    """A Qwen-shaped byte-level BPE (Qwen's split regex + ByteLevel, NFC) trained on the test corpus, with special
    added tokens, non-special added tokens known to tokenizer.json, one non-special and one special token declared only
    in tokenizer_config.json. Returns the model dir (tokenizer.json, tokenizer_config.json, config.json)."""
    from tokenizers import AddedToken, Regex, Tokenizer, decoders, models, normalizers, pre_tokenizers, trainers
    d = out_dir or tempfile.mkdtemp(prefix="tok-offloop-synth-")
    tok = Tokenizer(models.BPE())
    tok.normalizer = normalizers.NFC()
    tok.pre_tokenizer = pre_tokenizers.Sequence([
        pre_tokenizers.Split(Regex(QWEN_SPLIT), behavior="isolated"),
        pre_tokenizers.ByteLevel(add_prefix_space=False, use_regex=False),
    ])
    tok.decoder = decoders.ByteLevel()
    trainer = trainers.BpeTrainer(vocab_size=vocab_size, special_tokens=SPECIALS,
                                  initial_alphabet=pre_tokenizers.ByteLevel.alphabet(), show_progress=False)
    tok.train_from_iterator(corpus(seed=1) * 4, trainer=trainer)
    tok.add_tokens([AddedToken(t, special=False, normalized=False) for t in UNSPECIAL_KNOWN])
    tok.save(os.path.join(d, "tokenizer.json"))
    n = tok.get_vocab_size()
    atd = {}
    for t in SPECIALS + UNSPECIAL_KNOWN:
        atd[str(tok.token_to_id(t))] = {"content": t, "special": t in SPECIALS}
    atd[str(n + 3)] = {"content": UNSPECIAL_UNKNOWN, "special": False}
    atd[str(n + 4)] = {"content": MISSING_SPECIAL, "special": True}
    json.dump({"added_tokens_decoder": atd, "eos_token": "<|im_end|>", "bos_token": None,
               "pad_token": "<|endoftext|>"}, open(os.path.join(d, "tokenizer_config.json"), "w"))
    im_end = tok.token_to_id("<|im_end|>")
    json.dump({"eos_token_id": im_end, "bos_token_id": None, "pad_token_id": None}, open(os.path.join(d, "config.json"), "w"))
    return d


# ------------------------------------------------------------------------------------------------ corpus
def corpus(seed=0):
    """Varied strings: ASCII prose, unicode (CJK, RTL, combining marks, NFC/NFD pairs), emoji (ZWJ sequences, flags),
    code, whitespace runs, control characters, chat-template text with special and non-special added tokens (whole,
    adjacent, split across words), the synthetic tokenizer's config-only tokens, the empty string, and a 100k-char
    mixed document."""
    rng = random.Random(seed)
    base = [
        "",
        " ",
        "\n",
        "Hello, world! This is a plain ASCII sentence.",
        "  leading and trailing spaces  ",
        "tabs\tand\ttabs\t\t\tthen newlines\n\n\n\nand CRLF\r\n\r\nend",
        " " * 257 + "x" + "\n" * 64 + "\t" * 33,
        "Ünïcödé façade naïve coöperate — “quotes” ‘single’ … ellipsis",
        "日本語のテキストと中文文本以及한국어 텍스트",
        "العربية والعبرית עברית mixed with English",
        "é vs é (NFD vs NFC), äöü, Å vs Å",
        "emoji: 😀😃😄 👩‍👩‍👧‍👦 🏳️‍🌈 🇫🇷🇯🇵 👍🏽 ✔️",
        "zero​width‍joiner﻿bom \u0000nul \u0007bell \x7f",
        "def f(x):\n    return {'a': [1, 2, 3], \"b\": x ** 2}  # comment\n\nclass A(B):\n\tpass\n",
        "SELECT * FROM t WHERE a <> 'x' AND b >= 3; -- sql\n<div class=\"a\">&amp;&lt;</div>",
        "1234567890 3.14159 1e-10 0xDEADBEEF 1,000,000",
        "<|im_start|>system\nYou are helpful.<|im_end|>\n<|im_start|>user\nHi<|im_end|>\n<|im_start|>assistant\n",
        "<think>\nreasoning here\n</think>\n\nanswer<tool_call>\n{\"name\": \"f\", \"arguments\": {}}\n</tool_call>",
        "<|im_start|><|im_end|><think></think><|endoftext|>",
        "text<|im_start|>glued<|im_end|>text <think>x</think>y",
        "<|im_start <|im_end| |im_start|> <think <thin k>",
        "<|vision_start|><|image_pad|><|vision_end|>",
        f"a {UNSPECIAL_UNKNOWN} b {MISSING_SPECIAL} c{UNSPECIAL_UNKNOWN}{MISSING_SPECIAL}d",
        f"{UNSPECIAL_UNKNOWN}",
        f"{MISSING_SPECIAL}",
        "'s 't 're 've 'm 'll 'd 'S 'T don't won't y'all",
        "a" * 5000,
        "ab" * 3000 + "\n",
        "🙂" * 2000,
    ]
    words = ["the", "model", "token", "stream", "décodé", "流", "🚀", "def", "{", "}", "\n", "  ", "<think>", "</think>",
             "<|im_start|>", "<|im_end|>", "x_y", "CamelCase", "12", ".", ",", "\t", "Ω", "é"]
    for _ in range(40):
        base.append("".join(rng.choice(words) + rng.choice(["", " ", " ", "\n"]) for _ in range(rng.randint(1, 60))))
    long_doc = []
    while sum(len(s) for s in long_doc) < 100_000:
        long_doc.append(rng.choice(base[3:26]))
    base.append("".join(long_doc)[:100_000])
    return base


def long_text(chars):
    """A long agent-shaped prompt (code + prose + chat markup) of about `chars` characters."""
    block = ("<|im_start|>user\nPlease look at this function and explain it.\n```python\n"
             "def f(x):\n    return {'a': [1, 2, 3], 'b': x ** 2}  # comment\n```\n"
             "Ünïcödé, 日本語, emoji 🚀 and numbers 1234567890.<|im_end|>\n"
             "<|im_start|>assistant\n<think>\nLet me think about it.\n</think>\n\nIt squares x.<|im_end|>\n")
    return (block * (chars // len(block) + 1))[:chars]
