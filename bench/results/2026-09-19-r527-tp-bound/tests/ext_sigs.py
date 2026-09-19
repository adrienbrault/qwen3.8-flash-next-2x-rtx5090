"""Signature database of the compiled extension, parsed from the C++ sources in src/ (no compiler needed).

    bindings.cpp      m.def("name", &symbol, [py::arg(...)...])   -> free functions
    libtorch/*_bc.h   py::class_<C>(m, "Name").def(py::init<T...>(), py::arg...).def("meth", &C::meth, ...)
    headers           the C++ declarations of every bound symbol / method (parameter counts)

pybind11 only honours a default value when it is spelled as py::arg("x") = value; a C++ default argument without one
is still a required positional parameter from Python. That is the rule `Sig.bind` applies.

`StubExt` (below) is a stand-in for `exllamav3.ext.exllamav3_ext` that accepts ONLY names the real module binds and
raises on any call whose arity or keyword names the real binding would reject. It records every call.
"""
from __future__ import annotations

import os
import re
from dataclasses import dataclass, field

_ID = r"[A-Za-z_]\w*"


@dataclass
class Sig:
    name: str
    n_params: int                  # C++ parameter count
    arg_names: list = field(default_factory=list)   # py::arg names (may be empty)
    n_defaults: int = 0            # trailing py::arg(...) = default count
    source: str = ""

    def bind(self, args, kwargs, what=""):
        n = len(args) + len(kwargs)
        if kwargs:
            if not self.arg_names:
                raise TypeError(f"{what}{self.name}: keyword arguments {sorted(kwargs)} but the binding declares no "
                                f"py::arg names ({self.source})")
            pos_names = self.arg_names[:len(args)]
            for k in kwargs:
                if k not in self.arg_names:
                    raise TypeError(f"{what}{self.name}: unknown keyword '{k}' (py::args {self.arg_names})")
                if k in pos_names:
                    raise TypeError(f"{what}{self.name}: '{k}' given twice")
        required = self.n_params - self.n_defaults
        if n < required or n > self.n_params:
            raise TypeError(f"{what}{self.name}: {n} arguments given, binding takes {required}..{self.n_params} "
                            f"({self.source})")
        if kwargs:
            given = set(self.arg_names[:len(args)]) | set(kwargs)
            missing = [a for a in self.arg_names[:required] if a not in given]
            if missing:
                raise TypeError(f"{what}{self.name}: missing required {missing}")
        return True


class SigSet:
    """Overloads: binding succeeds if any overload accepts the call."""

    def __init__(self, sigs):
        self.sigs = list(sigs)
        self.name = self.sigs[0].name

    def bind(self, args, kwargs, what=""):
        errs = []
        for sg in self.sigs:
            try:
                return sg.bind(args, kwargs, what)
            except TypeError as e:
                errs.append(str(e))
        raise TypeError(" | ".join(errs))

    @property
    def n_params(self):
        return [sg.n_params for sg in self.sigs] if len(self.sigs) > 1 else self.sigs[0].n_params

    def __repr__(self):
        return f"SigSet({self.sigs})"


def _strip_comments(s: str) -> str:
    s = re.sub(r"/\*.*?\*/", " ", s, flags=re.S)
    return re.sub(r"//[^\n]*", " ", s)


def _balanced(s: str, i: int, open_c="(", close_c=")") -> int:
    """s[i] == open_c; return index of the matching close."""
    depth = 0
    for j in range(i, len(s)):
        c = s[j]
        if c == open_c:
            depth += 1
        elif c == close_c:
            depth -= 1
            if depth == 0:
                return j
    raise ValueError("unbalanced")


def _split_top(s: str) -> list:
    out, depth, cur = [], 0, []
    for c in s:
        if c in "(<[{":
            depth += 1
        elif c in ")>]}":
            depth -= 1
        if c == "," and depth == 0:
            out.append("".join(cur))
            cur = []
        else:
            cur.append(c)
    if "".join(cur).strip():
        out.append("".join(cur))
    return [x.strip() for x in out if x.strip()]


def _param_count(plist: str) -> int:
    plist = plist.strip()
    if not plist or plist == "void":
        return 0
    return len(_split_top(plist))


def _decls(text: str, name: str) -> list:
    """Parameter counts of C++ declarations/definitions of `name` in text (not calls)."""
    res = []
    for m in re.finditer(r"(?<![\w.:>])" + re.escape(name) + r"\s*\(", text):
        start = m.start()
        # preceding token must look like a return type (identifier, '*', '&', '>'), not an operator/call context
        pre = text[:start].rstrip()
        if not pre or pre[-1] in "=(,!?:+-/%|^~" or pre.endswith("return") or pre.endswith("."):
            continue
        prev_tok = re.search(r"(\w+|[*&>])\s*$", pre)
        if not prev_tok:
            continue
        if prev_tok.group(1) in ("return", "else", "new", "case", "if", "while", "for", "switch", "sizeof"):
            continue
        p0 = text.index("(", m.start())
        try:
            p1 = _balanced(text, p0)
        except ValueError:
            continue
        after = text[p1 + 1:p1 + 80].lstrip()
        after = re.sub(r"^(const|override|noexcept|final)\b\s*", "", after)
        if not after or after[0] not in ";{":
            continue
        res.append(_param_count(text[p0 + 1:p1]))
    return res


def _parse_pyargs(s: str):
    names, n_def = [], 0
    for m in re.finditer(r"py::arg\(\s*\"(\w+)\"\s*\)(\s*=\s*)?", s):
        names.append(m.group(1))
        if m.group(2):
            n_def += 1
    return names, n_def


class ExtSigs:
    def __init__(self, ext_dir: str):
        self.ext_dir = ext_dir
        self.headers = {}
        for root, _, files in os.walk(ext_dir):
            for f in files:
                if f.endswith((".h", ".cuh", ".hpp")):
                    p = os.path.join(root, f)
                    with open(p, errors="replace") as fh:
                        self.headers[os.path.relpath(p, ext_dir)] = _strip_comments(fh.read())
        with open(os.path.join(ext_dir, "bindings.cpp")) as fh:
            self.bindings = _strip_comments(fh.read())
        self.funcs: dict[str, Sig] = {}
        self.classes: dict[str, dict] = {}
        self.unresolved: list = []
        self.attrs = {m.group(1): m.group(2).strip() for m in
                      re.finditer(r"m\.attr\(\s*\"(\w+)\"\s*\)\s*=\s*([^;]+);", self.bindings)}
        self._parse_funcs()
        self._parse_classes()

    # -- free functions ----------------------------------------------------------------------------------------------
    def _parse_funcs(self):
        b = self.bindings
        for m in re.finditer(r"m\.def\(\s*\"(\w+)\"\s*,\s*&\s*(" + _ID + r"(?:::" + _ID + r")*)", b):
            pyname, sym = m.group(1), m.group(2)
            p0 = b.index("(", m.start())
            p1 = _balanced(b, p0)
            names, n_def = _parse_pyargs(b[p0:p1])
            counts = []
            src = ""
            short = sym.split("::")[-1]
            for hp, text in self.headers.items():
                c = _decls(text, short)
                if c:
                    counts += c
                    src = src or hp
            if not counts:
                self.unresolved.append(pyname)
                continue
            if names:
                if len(names) not in counts:
                    # py::arg list must cover every parameter; trust it but record the mismatch
                    self.unresolved.append(f"{pyname} (py::args {len(names)} vs decl {counts})")
                self.funcs[pyname] = SigSet([Sig(pyname, len(names), names, n_def, src)])
            else:
                self.funcs[pyname] = SigSet([Sig(pyname, c, [], 0, src) for c in sorted(set(counts))])

    # -- BC classes ------------------------------------------------------------------------------------------------
    def _class_body(self, cls: str):
        for hp, text in self.headers.items():
            for m in re.finditer(r"\b(struct|class)\s+" + re.escape(cls) + r"\b[^;{]*\{", text):
                i = text.index("{", m.start())
                j = _balanced(text, i, "{", "}")
                return hp, text[i + 1:j]
        return None, None

    def _parse_classes(self):
        texts = {hp: t for hp, t in self.headers.items() if hp.endswith("_bc.h")}
        for hp, t in texts.items():
            for m in re.finditer(r"py::class_<\s*(" + _ID + r")[^>]*>+\s*\(\s*m\s*,\s*\"(\w+)\"\s*\)", t):
                cls, pyname = m.group(1), m.group(2)
                # the chain runs until the terminating ';'
                end = t.index(";", m.end())
                chain = t[m.end():end]
                info = {"init": None, "methods": {}, "source": hp}
                inits = []
                for im in re.finditer(r"py::init<(.*?)>\s*\(\s*\)", chain, flags=re.S):
                    types = _split_top(im.group(1))
                    rest = chain[im.end():]
                    stop = rest.find(".def(")
                    names, n_def = _parse_pyargs(rest if stop < 0 else rest[:stop])
                    inits.append(Sig(pyname, len(types), names, n_def, hp))
                if inits:
                    info["init"] = SigSet(inits)
                chp, body = self._class_body(cls)
                for dm in re.finditer(r"\.def\(\s*\"(\w+)\"\s*,\s*&\s*" + re.escape(cls) + r"::(\w+)", chain):
                    meth, sym = dm.group(1), dm.group(2)
                    p0 = chain.index("(", dm.start())
                    p1 = _balanced(chain, p0)
                    names, n_def = _parse_pyargs(chain[p0:p1])
                    counts = _decls(body, sym) if body else []
                    if not counts:
                        self.unresolved.append(f"{pyname}.{meth}")
                        continue
                    if names:
                        new = [Sig(f"{pyname}.{meth}", len(names), names, n_def, chp or hp)]
                    else:
                        new = [Sig(f"{pyname}.{meth}", c, [], 0, chp or hp) for c in sorted(set(counts))]
                    old = info["methods"].get(meth)
                    info["methods"][meth] = SigSet((old.sigs if old else []) + new)
                self.classes[pyname] = info


# ----------------------------------------------------------------------------------------------------------------------
# Stub extension module
# ----------------------------------------------------------------------------------------------------------------------

class StubBC:
    def __init__(self, stub, pyname, info, args, kwargs):
        self._stub, self._pyname, self._info = stub, pyname, info
        self._args, self._kwargs = args, kwargs

    def __getattr__(self, name):
        if name.startswith("_"):
            raise AttributeError(name)
        meths = self._info["methods"]
        if name not in meths:
            raise AttributeError(f"{self._pyname} has no bound method '{name}' (binds {sorted(meths)})")
        sig = meths[name]

        def call(*args, **kwargs):
            sig.bind(args, kwargs)
            self._stub.calls.append((f"{self._pyname}.{name}", args, kwargs))
            return self._stub.returns.get(f"{self._pyname}.{name}", lambda *a, **k: None)(*args, **kwargs)
        return call


class StubExt:
    """Drop-in for exllamav3_ext that validates names and arities against ExtSigs."""

    def __init__(self, sigs: ExtSigs, returns: dict | None = None):
        self._sigs = sigs
        self.calls = []
        self.returns = dict(returns or {})

    def __getattr__(self, name):
        if name.startswith("__"):
            raise AttributeError(name)
        s = self.__dict__["_sigs"]
        if name in s.classes:
            info = s.classes[name]

            def ctor(*args, **kwargs):
                if info["init"] is None:
                    raise TypeError(f"{name} has no py::init")
                info["init"].bind(args, kwargs, "BC ")
                self.calls.append((name, args, kwargs))
                return StubBC(self, name, info, args, kwargs)
            return ctor
        if name in s.funcs:
            sig = s.funcs[name]

            def fn(*args, **kwargs):
                sig.bind(args, kwargs)
                self.calls.append((name, args, kwargs))
                r = self.returns.get(name)
                return r(*args, **kwargs) if r else None
            return fn
        if name in s.attrs:
            v = s.attrs[name]
            return int(v) if v.isdigit() else 0
        raise AttributeError(f"exllamav3_ext has no binding '{name}' (not in bindings.cpp)")

    def names_called(self):
        return [c[0] for c in self.calls]
