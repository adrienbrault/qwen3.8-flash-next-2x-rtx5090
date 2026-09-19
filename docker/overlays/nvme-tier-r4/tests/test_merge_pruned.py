"""Round 4: the generator.py for the mtp-pruned-r1 base (overlay-pruned/) is mtp-pruned-r1's generator.py plus exactly the
tier's hunks, placed exactly as they were placed in the stack generator.py (overlay/), so with EXL3_NVME_TIER unset it
behaves as mtp-pruned-r1's generator.py.

  1. text: the stack port and the pruned port are insert-only diffs with the same inserted blocks between the same
     context lines, in the same order
  2. three-way: `git merge-file` of (pruned, stack, stack + tier) is conflict-free and byte-identical to overlay-pruned
  3. AST: removing the tier's statements from overlay-pruned gives mtp-pruned-r1's module AST exactly, and nothing else
     in the module names the tier
  4. behaviour: mtp-pruned-r1's own CPU suite (drafting chain, pruned mirror, keep ids, embedding module) passes with
     the merged generator.py in place of its own
"""
import ast
import difflib
import pathlib
import shutil
import subprocess
import sys

import pytest

import harness

DELIV = harness.DELIV
ROOT = harness.ROOT
GEN = "exllamav3/generator/generator.py"
PRUNEDDIR = ROOT / "ref" / "mtp-pruned-r1"
STACK = ROOT / "src" / "exllamav3" / "generator" / "generator.py"
STACK_TIER = DELIV / "overlay" / GEN
PRUNED = PRUNEDDIR / "overlay" / GEN
MERGED = DELIV / "overlay-pruned" / GEN


def insertions(base: pathlib.Path, ported: pathlib.Path):
    """[(line before, inserted block, line after)] of an insert-only diff; fails on any other edit."""
    a = base.read_text().splitlines()
    b = ported.read_text().splitlines()
    out = []
    for op, i1, i2, j1, j2 in difflib.SequenceMatcher(a = a, b = b, autojunk = False).get_opcodes():
        assert op in ("equal", "insert"), (ported, op, a[i1:i2], b[j1:j2])
        if op == "insert":
            out.append((a[i1 - 1] if i1 else None, tuple(b[j1:j2]), a[i1] if i1 < len(a) else None))
    return out


def test_pruned_port_has_the_stack_ports_hunks_only():
    stack = insertions(STACK, STACK_TIER)
    pruned = insertions(PRUNED, MERGED)
    assert len(stack) == len(pruned) == 5
    # same inserted blocks, each right before the same line
    assert [(blk, after) for _, blk, after in stack] == [(blk, after) for _, blk, after in pruned]
    # and right after the same line, except the flag block: its anchor is "class Generator:", and mtp-pruned-r1 put
    # its own flag (_EMBED_GPU_PRUNED) right before that class, so the tier flag now follows the pruned flag
    assert [before for before, _, _ in stack[1:]] == [before for before, _, _ in pruned[1:]]
    assert stack[0][0].startswith("_MTP_DEVICE_DRAFT = ") and pruned[0][0].startswith("_EMBED_GPU_PRUNED = ")


def test_three_way_merge_equals_overlay_pruned(tmp_path):
    if shutil.which("git") is None:
        pytest.skip("git not installed")
    cur = tmp_path / "pruned.py"
    shutil.copyfile(PRUNED, cur)
    r = subprocess.run(["git", "merge-file", "-p", str(cur), str(STACK), str(STACK_TIER)], capture_output = True)
    assert r.returncode == 0, r.stderr        # exit status = number of conflicts
    assert r.stdout == MERGED.read_bytes()


def _is_tier_node(node) -> bool:
    """A statement the tier port adds: the module flag, `self.disk_page_cache = None`, the `if _NVME_TIER:` install
    block, or an `if self.disk_page_cache is not None:` block that only calls methods of self.disk_page_cache."""
    src = ast.unparse(node)
    if isinstance(node, ast.Assign):
        return src in ('_NVME_TIER = os.environ.get(\'EXL3_NVME_TIER\', \'\')', "self.disk_page_cache = None")
    if isinstance(node, ast.If) and not node.orelse:
        test = ast.unparse(node.test)
        if test == "_NVME_TIER":
            return [ast.unparse(s) for s in node.body] == [
                "from .disk_cache import DiskPageCache",
                "self.disk_page_cache = DiskPageCache.install(self, cache, draft_cache, _NVME_TIER)",
            ]
        if test == "self.disk_page_cache is not None":
            return all(isinstance(s, ast.Expr) and isinstance(s.value, ast.Call) and not s.value.args and
                       not s.value.keywords and ast.unparse(s.value.func).startswith("self.disk_page_cache.")
                       for s in node.body)
    return False


class _StripTier(ast.NodeTransformer):
    def __init__(self):
        self.removed = 0

    def generic_visit(self, node):
        super().generic_visit(node)
        for field in ("body", "orelse", "finalbody"):
            stmts = getattr(node, field, None)
            if isinstance(stmts, list):
                keep = [s for s in stmts if not (isinstance(s, ast.stmt) and _is_tier_node(s))]
                self.removed += len(stmts) - len(keep)
                setattr(node, field, keep)
        return node


def test_tier_off_ast_equals_pruned():
    merged = ast.parse(MERGED.read_text())
    strip = _StripTier()
    strip.visit(merged)
    assert strip.removed == 6        # flag, attribute init, install block, pump / on_busy / on_idle guards
    assert ast.dump(merged) == ast.dump(ast.parse(PRUNED.read_text()))
    # every remaining mention of the tier would be a tier-off behaviour change: there is none
    names = {n.id for n in ast.walk(merged) if isinstance(n, ast.Name)} | \
            {n.attr for n in ast.walk(merged) if isinstance(n, ast.Attribute)}
    assert not names & {"_NVME_TIER", "disk_page_cache", "DiskPageCache"}


def test_pruned_rounds_own_cpu_suite_on_the_merged_generator(tmp_path):
    """mtp-pruned-r1's tests resolve their workspace as <deliverable>/../.. and load the generator from the
    deliverable's overlay/: lay out a root with src/ and a copy of the pruned deliverable whose generator.py is the
    merged one. test_packaging_cpu.py pins the pruned manifest's own hashes, so it is not run."""
    (tmp_path / "ref").mkdir()
    (tmp_path / "src").symlink_to(ROOT / "src", target_is_directory = True)
    dst = tmp_path / "ref" / "mtp-pruned-r1"
    shutil.copytree(PRUNEDDIR, dst, ignore = shutil.ignore_patterns("__pycache__", ".pytest_cache"))
    shutil.copyfile(MERGED, dst / "overlay" / GEN)
    tests = sorted(p.name for p in (dst / "tests").glob("test_*.py") if p.name != "test_packaging_cpu.py")
    r = subprocess.run([sys.executable, "-m", "pytest", "-q", "-p", "no:cacheprovider", *tests],
                       cwd = dst / "tests", capture_output = True, text = True,
                       env = {"PATH": "/usr/bin:/bin", "PYTHONDONTWRITEBYTECODE": "1"})
    assert r.returncode == 0, r.stdout[-3000:] + r.stderr[-3000:]
    last = r.stdout.strip().splitlines()[-1]
    assert " passed" in last and "failed" not in last and "error" not in last, last
    print(last)
