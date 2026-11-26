"""rebase165: tokenizer. 1.6.5's _encode_texts (newline-run chunking) carries the served tokenize-offloop r1
short-input path: encode_batch (GIL released) while EXL3_TOKENIZE_OFFLOOP != 0, encode() otherwise. Same ids either
way. The encode_special_tokens write and every backend encode run under _encode_lock. No torch (AST extraction)."""
import ast
import types
import unittest

from _iso import functions, text

TOK = "exllamav3/tokenizer/tokenizer.py"


class FakeBackend:
    def __init__(self):
        self.calls = []
        self.encode_special_tokens = False

    def encode(self, t, add_special_tokens = False):
        self.calls.append(("encode", t))
        return types.SimpleNamespace(ids = [ord(c) for c in t])

    def encode_batch(self, ts, add_special_tokens = False):
        self.calls.append(("encode_batch", tuple(ts)))
        return [types.SimpleNamespace(ids = [ord(c) for c in t]) for t in ts]


def make(environ):
    os_ns = types.SimpleNamespace(environ = environ)
    ns = functions(TOK, ["tokenize_offloop_enabled"], ns = {"os": os_ns})
    tree = ast.parse(text(TOK))
    consts = {n.targets[0].id: ast.literal_eval(n.value) for n in tree.body
              if isinstance(n, ast.Assign) and isinstance(n.targets[0], ast.Name)
              and n.targets[0].id == "_CHUNK_MIN_CHARS"}
    ns.update(consts)
    ns = functions(TOK, ["_encode_texts"], cls = "Tokenizer", ns = ns)
    self = types.SimpleNamespace(tokenizer = FakeBackend(), _chunking_ok = lambda: False)
    self._encode_texts = types.MethodType(ns["_encode_texts"], self)
    return self, ns


class Offloop(unittest.TestCase):

    def test_default_on_uses_encode_batch(self):
        tok, ns = make({})
        self.assertEqual(tok._encode_texts(["ab", "c"]), [[97, 98], [99]])
        self.assertEqual(tok.tokenizer.calls, [("encode_batch", ("ab", "c"))])
        self.assertEqual(tok._encode_texts([]), [])

    def test_off_uses_encode(self):
        tok, ns = make({"EXL3_TOKENIZE_OFFLOOP": "0"})
        self.assertEqual(tok._encode_texts(["ab", "c"]), [[97, 98], [99]])
        self.assertEqual([c[0] for c in tok.tokenizer.calls], ["encode", "encode"])

    def test_long_input_keeps_upstream_chunked_batch(self):
        for env in ({}, {"EXL3_TOKENIZE_OFFLOOP": "0"}):
            tok, ns = make(env)
            long = "x" * (ns["_CHUNK_MIN_CHARS"] + 1)
            self.assertEqual(tok._encode_texts([long]), [[120] * len(long)])
            self.assertEqual([c[0] for c in tok.tokenizer.calls], ["encode_batch"])

    def test_flag_write_and_encodes_under_one_lock(self):
        tree = ast.parse(text(TOK))
        cls = [n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "Tokenizer"][0]
        fn = [n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == "encode_part_base"][0]
        withs = [n for n in fn.body if isinstance(n, ast.With)]
        self.assertEqual(len(withs), 1)
        self.assertEqual(ast.unparse(withs[0].items[0].context_expr), "self._encode_lock")
        inner = ast.unparse(withs[0])
        self.assertIn("self.tokenizer.encode_special_tokens = want", inner)
        outside = [ast.unparse(n) for n in fn.body if n is not withs[0]]
        self.assertFalse(any("_encode_texts" in s or "encode_special_tokens =" in s for s in outside))


if __name__ == "__main__":
    unittest.main()
