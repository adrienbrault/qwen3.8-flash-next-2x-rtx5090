#!/usr/bin/env python3
"""tokcount r1 landing check: run inside the built image (python3, no GPU needed).

Re-executes once with every EXL3_* key stripped (the base image bakes EXL3_* keys into its ENV), then:
  - exllamav3.generator.job imports (CPU; torch only warns that no CUDA runtime is present);
  - Job.prepare_for_requeue builds rq_state["rq_new_tokens"] as `self.rq_new_tokens + self.new_tokens` (AST, not grep);
  - the unfixed form `"rq_new_tokens": self.new_tokens,` is gone and the tokcount-r1 marker is present exactly once;
  - rq_new_tokens is still read in exactly one place outside __init__/prepare_for_requeue: the emit_eos result's
    "new_tokens" (so the fix changes reporting and nothing else; ANALYSIS.md has the full consumer list).
Last line "tokcount r1 landed: ..." and exit 0, or an AssertionError.
"""
import ast, os, sys

REEXEC = "TOKCOUNT_LANDING_REEXEC"


def main():
    leaked = sorted(k for k in os.environ if k.startswith("EXL3_"))
    if leaked:
        assert os.environ.get(REEXEC) != "1", f"EXL3_* keys survived the re-exec: {leaked}"
        env = {k: v for k, v in os.environ.items() if not k.startswith("EXL3_")}
        env[REEXEC] = "1"
        os.execve(sys.executable, [sys.executable] + sys.argv, env)

    import exllamav3.generator.job as jm  # must import on CPU
    path = jm.__file__
    src = open(path).read()
    assert src.count("# tokcount-r1: cumulative across requeues") == 1, "tokcount-r1 marker missing or duplicated"
    assert '"rq_new_tokens": self.new_tokens,' not in src, "unfixed rq_new_tokens line still present"

    tree = ast.parse(src)
    job = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "Job")
    prq = next(n for n in job.body if isinstance(n, ast.FunctionDef) and n.name == "prepare_for_requeue")
    vals = []
    for node in ast.walk(prq):
        if isinstance(node, ast.Dict):
            for k, v in zip(node.keys, node.values):
                if isinstance(k, ast.Constant) and k.value == "rq_new_tokens":
                    vals.append(ast.unparse(v))
    assert vals == ["self.rq_new_tokens + self.new_tokens"], f"rq_state['rq_new_tokens'] = {vals}"

    # every read of self.rq_new_tokens outside __init__ and prepare_for_requeue
    reads = []
    for fn in (n for n in job.body if isinstance(n, ast.FunctionDef) and n.name not in ("__init__", "prepare_for_requeue")):
        for node in ast.walk(fn):
            if isinstance(node, ast.Attribute) and node.attr == "rq_new_tokens" and isinstance(node.ctx, ast.Load):
                reads.append((fn.name, node.lineno))
    assert len(reads) == 1, f"rq_new_tokens read at {reads} (expected only the emit_eos report)"
    line = src.splitlines()[reads[0][1] - 1]
    assert '"new_tokens": self.rq_new_tokens + self.new_tokens' in line, f"unexpected reader: {line.strip()}"

    print(f"tokcount r1 landed: {path} imports on CPU; rq_state carries self.rq_new_tokens + self.new_tokens; "
          f"unfixed line absent; marker x1; sole reader {reads[0][0]}:{reads[0][1]} (the reported new_tokens)")


if __name__ == "__main__":
    main()
