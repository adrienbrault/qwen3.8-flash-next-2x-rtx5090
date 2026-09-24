#!/usr/bin/env python3
"""rows32 r4 landing check: run inside tabbyapi:stack-r3-rows32 (python3, no GPU needed).

1. The base image bakes EXL3_* keys into its ENV (R716: EXL3_DECODE_FUSE, EXL3_MOE_COOP_V2, ...). The checks are about
   defaults, so this script re-executes itself once with every EXL3_* key stripped, before anything imports exllamav3
   (stack-r3's tools/landing.py pattern; R716's first build failed on exactly this).
2. stack-r3's own landing (/opt/stack-r3/tools/landing.py --include <the base's INCLUDE>): every stack-r3 marker (hcfast
   r2, latchain 1, moefast 3, densegemm 2 + lcguard), every stack-r3 flag default off, HC2 tables parse.
3. rows32 r4, torch imported BEFORE exllamav3_ext (the extension links libc10.so, R700):
   - moe_rows32_revision == 2; the C++ cap is 16 unset, 32 with EXL3_MOE_COOP_ROWS32=1, and "true" is refused;
   - module constants: modules/mlp.py and modules/block_sparse_mlp.py MAX_BSZN 16 unset, 32 / 32 with both flags;
     EXL3_MOE_COOP_ROWS32=1 alone is refused (the shared expert runs the same rows);
   - densegemm-r2's EXL3_DENSE_ROWS32 parses: dense_v2_modes() == (0, 1) with it, (1, 1) with EXL3_DENSE_V2=1 too
     (the A union's mode 1 + rows32 = the B env's dense part).

  landing_rows32.py --include "mf3 dg2"   -> last line "rows32 r4 landed: ..." and exit 0, or an AssertionError
"""
import argparse
import os
import subprocess
import sys

REEXEC = "ROWS32_R4_LANDING_REEXEC"


def child(code, env_add):
    env = {k: v for k, v in os.environ.items() if not k.startswith("EXL3_")}
    env.update(env_add)
    r = subprocess.run([sys.executable, "-c", code], env=env, capture_output=True, text=True)
    return r.returncode, (r.stdout + r.stderr).strip()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--include", required=True, help="the base image's stack-r3 INCLUDE (label local.stack.include)")
    ap.add_argument("--s3-landing", default="/opt/stack-r3/tools/landing.py")
    a = ap.parse_args()
    leaked = sorted(k for k in os.environ if k.startswith("EXL3_"))
    if leaked:
        assert os.environ.get(REEXEC) != "1", f"EXL3_* keys survived the re-exec: {leaked}"
        print(f"landing: stripping image EXL3_* keys and re-executing: {leaked}", flush=True)
        env = {k: v for k, v in os.environ.items() if not k.startswith("EXL3_")}
        env[REEXEC] = "1"
        os.execve(sys.executable, [sys.executable] + sys.argv, env)

    inc = set(a.include.replace(",", " ").split())
    assert "dg2" in inc, f"base INCLUDE {sorted(inc)} lacks dg2: EXL3_DENSE_ROWS32 needs densegemm-r2's rows32 twins"
    r = subprocess.run([sys.executable, a.s3_landing, "--include", a.include], capture_output=True, text=True)
    print((r.stdout + r.stderr).strip().splitlines()[-1] if (r.stdout + r.stderr).strip() else "(no output)")
    assert r.returncode == 0, f"stack-r3 landing failed:\n{(r.stdout + r.stderr)[-1500:]}"

    import torch  # noqa: F401  (before the extension)
    import importlib.util as u
    import exllamav3_ext as e
    print("ext:", u.find_spec("exllamav3_ext").origin)
    assert getattr(e, "moe_rows32_revision", None) == 2, f"moe_rows32_revision {getattr(e, 'moe_rows32_revision', None)}"
    assert e.exl3_moe_coop_rows_cap() == 16, "the MoE rows cap must default to 16"
    assert tuple(e.dense_v2_modes()) == (0, 0), f"densegemm must default off: {e.dense_v2_modes()}"
    probe = ("import torch, exllamav3_ext as e\n"
             "import exllamav3.modules.mlp as m, exllamav3.modules.block_sparse_mlp as b\n"
             "print('RES', m.MAX_BSZN, b.MAX_BSZN, e.exl3_moe_coop_rows_cap(), tuple(e.dense_v2_modes()))\n")
    cases = [({}, "RES 16 16 16 (0, 0)", True),
             ({"EXL3_SHARED_EXPERT_ROWS32": "1", "EXL3_MOE_COOP_ROWS32": "1", "EXL3_DENSE_ROWS32": "1"},
              "RES 32 32 32 (0, 1)", True),
             ({"EXL3_SHARED_EXPERT_ROWS32": "1", "EXL3_MOE_COOP_ROWS32": "1", "EXL3_DENSE_ROWS32": "1",
               "EXL3_DENSE_V2": "1"}, "RES 32 32 32 (1, 1)", True),
             ({"EXL3_MOE_COOP_ROWS32": "1"}, "requires EXL3_SHARED_EXPERT_ROWS32=1", False),
             ({"EXL3_SHARED_EXPERT_ROWS32": "true"}, "must be 0 or 1", False)]
    for env_add, want, ok in cases:
        rc, out = child(probe, env_add)
        assert want in out and (rc == 0) == ok, f"{env_add}: expected {want!r} (rc {'0' if ok else '!=0'}), got rc {rc}: {out[-800:]}"
    rc, out = child("import torch, exllamav3_ext as e; e.exl3_moe_coop_rows_cap()", {"EXL3_MOE_COOP_ROWS32": "true"})
    assert rc != 0 and "must be 0 or 1" in out, f"EXL3_MOE_COOP_ROWS32=true must be refused: rc {rc} {out[-400:]}"
    print(f"rows32 r4 landed: moe_rows32_revision 2 on stack-r3 (INCLUDE '{a.include}'); flags default off (caps 16, "
          f"dense (0, 0)); with the three flags MAX_BSZN 32/32, coop cap 32, dense (0, 1); 4b-without-4c and non-0/1 "
          f"values refused")


if __name__ == "__main__":
    main()
