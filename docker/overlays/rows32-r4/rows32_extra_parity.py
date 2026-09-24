#!/usr/bin/env python3
"""rows32 r4: stack-r3's own GPU parity tests at the row counts the rows32 policy serves, without copying them.

The rows32 B policy ([[4, 3], [8, 2]]) serves c6 / c7 / c8 at depth 2: 18 / 21 / 24 verify rows, i.e. (bsz, q_len) =
(6, 3), (7, 3), (8, 3) in the per-sequence kernels. stack-r3's tests stop short of that:
  densegemm-r2  ROWS = 1..16, 17, 24, 32              -> adds 18 and 21 (R714 checked 17, 24, 32 only)
  latchain      GDN (bsz, seqlen) up to (8, 4) without (6, 3) / (7, 3) / (8, 3); QSA only the served shapes
                                                      -> both lists get (6, 3), (7, 3), (8, 3) (+ the served controls)
  hcfast r2     ROWS up to 16, 17, 24, 32              -> adds 18 and 21
This wrapper imports the test module FROM THE IMAGE (/opt/stack-r3/tests/..., the file R716b ran), overrides its
module-level shape list, and calls its main() with the given arguments. The comparison code is theirs, unchanged.

  rows32_extra_parity.py dense    --model /models/<ckpt> --mode 1 --json /results/parity-dense-rows32.json
  rows32_extra_parity.py latchain --json /results/parity-latchain-rows32.json
  rows32_extra_parity.py hcfast   --model /models/<ckpt> --json /results/parity-hcfast-rows32.json
Exit code and the "PARITY PASS" / "PARITY FAIL" line are the wrapped test's.
"""
import importlib.util
import os
import sys

TESTS = os.environ.get("STACK_R3_TESTS", "/opt/stack-r3/tests")
SUITES = {
    # suite: (path under TESTS, {module attribute: new value})
    "dense": ("densegemm-r2/test_densegemm_parity.py",
              {"ROWS": [1, 4, 8, 16, 17, 18, 21, 24, 32]}),
    "latchain": ("latchain/test_latchain_parity.py",
                 {"GDN_SHAPES": [(1, 4), (4, 4), (8, 2), (6, 3), (7, 3), (8, 3), (6, 1), (7, 1), (8, 1)],
                  "QSA_SHAPES": [(1, 4), (4, 4), (8, 2), (6, 3), (7, 3), (8, 3)],
                  # the served QSA options and the stack-r3 QT line (split 2, combine 1, div16); the full 24-variant
                  # sweep ran in stack-r3's own parity
                  "VARIANTS": [(2, 1, False), (2, 1, True)]}),
    "hcfast": ("hcfast/test_hcfast_parity.py",
               {"ROWS": (1, 4, 8, 16, 17, 18, 21, 24, 32)}),
}


def main():
    if len(sys.argv) < 2 or sys.argv[1] not in SUITES:
        print(__doc__)
        return 2
    suite = sys.argv[1]
    rel, over = SUITES[suite]
    path = os.path.join(TESTS, rel)
    spec = importlib.util.spec_from_file_location(f"rows32_extra_{suite}", path)
    mod = importlib.util.module_from_spec(spec)
    sys.argv = [path] + sys.argv[2:]            # the wrapped test parses these; children it spawns re-run ITS file
    spec.loader.exec_module(mod)
    for k, v in over.items():
        assert hasattr(mod, k), f"{rel} has no {k}: the wrapped test changed, update rows32_extra_parity.py"
        setattr(mod, k, v)
    print(f"[rows32-extra] {suite}: {rel} with " + "; ".join(f"{k} = {v}" for k, v in over.items()), flush=True)
    rc = mod.main()
    return rc if isinstance(rc, int) else 0


if __name__ == "__main__":
    sys.exit(main())
