#!/usr/bin/env python3
"""stack-r3 landing check: run inside the image (python3, no GPU needed, no EXL3_* flags in the environment).
torch is imported BEFORE exllamav3_ext (the extension links libc10.so, which only torch's import puts on the loader
path: R700's failure). Every included patch's revision marker is asserted, every flag must default off, and the
pre-registered HC2 tables must parse to the values R702 measured.

  landing.py --include "mf3 dg1"      -> last line "stack-r3 landed: ..." and exit 0, or an AssertionError
"""
import argparse
import os
import subprocess
import sys

HC2 = ("EXL3_HC_MIX_V3=2 EXL3_HC_MIX_V3_DOTS_B=1:1,4:2,32:4 EXL3_HC_MIX_V3_UP_B=1:1,8:4,32:8 "
       "EXL3_HC_MIX_V3_DOTS_J=1:4,32:8 EXL3_HC_MIX_V3_DOTS_PF=1:1,32:0 EXL3_HC_MIX_V3_UP_Q=1:4,8:2,32:4 "
       "EXL3_HC_MIX_V3_PDL=0")


def child(code, env_add):
    env = {k: v for k, v in os.environ.items() if not k.startswith("EXL3_")}
    env.update(env_add)
    r = subprocess.run([sys.executable, "-c", code], env=env, capture_output=True, text=True)
    assert r.returncode == 0, f"child check failed with {env_add}:\n{r.stderr[-1500:]}"
    return r.stdout.strip()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--include", default="")
    ap.add_argument("--hc2", default=HC2, help="the HC2 env string to parse-check")
    a = ap.parse_args()
    inc = set(a.include.replace(",", " ").split())
    assert not (inc - {"mf3", "dg1", "dg2"}), f"unknown INCLUDE {inc}"
    assert not {"dg1", "dg2"} <= inc, "dg1 and dg2 exclude each other"
    leaked = sorted(k for k in os.environ if k.startswith("EXL3_"))
    if leaked:
        # The base image bakes EXL3_* keys into its ENV (R716: EXL3_DECODE_FUSE, EXL3_MOE_COOP_V2, ...). The checks
        # below are about defaults, so re-exec once with them stripped, before anything imports exllamav3.
        assert os.environ.get("STACK_R3_LANDING_REEXEC") != "1", f"EXL3_* keys survived the re-exec: {leaked}"
        print(f"landing: stripping image EXL3_* keys and re-executing: {leaked}", flush=True)
        env = {k: v for k, v in os.environ.items() if not k.startswith("EXL3_")}
        env["STACK_R3_LANDING_REEXEC"] = "1"
        os.execve(sys.executable, [sys.executable] + sys.argv, env)

    import torch  # noqa: F401  (before the extension)
    import importlib.util as u
    import exllamav3_ext as e
    print("ext:", u.find_spec("exllamav3_ext").origin)
    # core: hcfast r2
    assert getattr(e, "hc_mix_v3_revision", None) == 2, f"hc_mix_v3_revision {getattr(e, 'hc_mix_v3_revision', None)}"
    assert hasattr(e, "gr_mix_v2_int8_v3") and hasattr(e, "gr_mix_v2_int8_statein")
    import exllamav3.modules.hyperconnections as h
    assert h._HC_MIX_V3_BUILD == "r2", h._HC_MIX_V3_BUILD
    G = h.GatedResidual
    assert G.MIX_V3 is False and G.MIX_V3_LEVEL == 0, "EXL3_HC_MIX_V3 must default off"
    # core: latchain r1 (r1b = same code, re-anchored bindings line)
    assert getattr(e, "latchain_revision", None) == 1, f"latchain_revision {getattr(e, 'latchain_revision', None)}"
    assert hasattr(e.TritonKernel, "launch_py")
    import exllamav3.modules.attention_fn.bc_attn as b
    assert b.LC_BUILD == "latchain-r1", b.LC_BUILD
    assert b._LC_QSA_SPLIT_STAGES == 0 and b._LC_QSA_COMBINE_STAGES == 0 and not b._LC_QSA_DIV16, "QT must default off"
    # moefast: r1 (stack-r2) or r3
    want_mf = 3 if "mf3" in inc else 1
    assert e.moe_coop_v3_revision == want_mf, f"moe_coop_v3_revision {e.moe_coop_v3_revision} != {want_mf}"
    import exllamav3.modules.block_sparse_mlp as bsm
    if want_mf == 3:
        assert hasattr(e, "exl3_moe_coop_ev")
        for f in ("start_shared", "shared_early", "shared_prio", "run_bszN"):
            assert hasattr(e.BC_BlockSparseMLP, f), f
        assert hasattr(bsm.BlockSparseMLP, "_shared_early_ok")
    else:
        assert not hasattr(e, "exl3_moe_coop_ev") and not hasattr(bsm.BlockSparseMLP, "_shared_early_ok")
    # densegemm + guard
    dg = 2 if "dg2" in inc else 1 if "dg1" in inc else 0
    if dg:
        assert e.dense_v2_revision == dg, f"dense_v2_revision {e.dense_v2_revision} != {dg}"
        assert getattr(e, "dense_v2_lc_guard", None) == 1, "densegemm without the stack-r3 side-branch guard"
        assert tuple(e.dense_v2_lc_side_counts()) == (0, 0), e.dense_v2_lc_side_counts()
        assert tuple(e.dense_v2_modes()) == (0, 0), f"EXL3_DENSE_V2 must default off: {e.dense_v2_modes()}"
        out = child("import torch, exllamav3_ext as e; print(tuple(e.dense_v2_modes()))", {"EXL3_DENSE_V2": "3"})
        assert out.endswith("(3, 0)"), out
    else:
        assert not hasattr(e, "dense_v2_revision"), "densegemm present but not included"
    # HC2 tables (the pre-registered R702 RECOMMEND string) parse to R702's per-R values
    env = dict(kv.split("=", 1) for kv in a.hc2.split())
    code = ("import torch, exllamav3.modules.hyperconnections as h; G = h.GatedResidual; "
            "print(G.MIX_V3_LEVEL, [G.MIX_V3_DOTS_B[r] for r in (1, 4, 8, 16)], [G.MIX_V3_UP_B[r] for r in (1, 4, 8, 16)], "
            "[G.MIX_V3_DOTS_J[r] for r in (1, 4, 16)], [G.MIX_V3_DOTS_PF[r] for r in (1, 16)], "
            "[G.MIX_V3_UP_Q[r] for r in (1, 4, 8, 16)], G.MIX_V3_PDL)")
    got = child(code, env).splitlines()[-1]
    want = "2 [1, 2, 4, 4] [1, 4, 4, 8] [4, 8, 8] [1, 0] [4, 2, 2, 4] False"
    assert got == want, f"HC2 tables parse to {got!r}, expected {want!r}"
    print(f"stack-r3 landed: hcfast r2 (hc_mix_v3_revision 2, build r2), latchain 1 ({b.LC_BUILD}), moefast {want_mf}, "
          f"densegemm {dg or 'absent'}{' + lcguard' if dg else ''}; every flag default off; HC2 tables parse ({got})")


if __name__ == "__main__":
    main()
