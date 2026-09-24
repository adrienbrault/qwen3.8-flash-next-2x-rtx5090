#!/usr/bin/env python3
"""hcfast r2 CPU flow test (no GPU): the Python dispatch in GatedResidual._mix.

The extension is replaced by a recording fake. Checks:
  * flag off: the served selection is untouched (gr_mix_v2_int8_statein with the served argument list),
    and gr_mix_v2_int8_v3 is never called;
  * EXL3_HC_MIX_V3=1 (r1, unchanged): gr_mix_v2_int8_v3 gets exactly the served 11 arguments (same tensor
    objects/storage, same order) plus mode 3, the two caps and the r2 knobs at their inert values (4, 0, 2);
  * EXL3_HC_MIX_V3=2 (r2): the same 11 arguments plus mode 15 (+16 with EXL3_HC_MIX_V3_PDL=1) and the
    per-R knob values from the tables; site form (post) and final-mixer form (post None);
  * the flag does not change the R > 32 GEMM path or the fp16 V2 path, and needs STATE_IN_UP;
  * the knobs parse (a value or a 'maxR:value,...' table) and reject anything else at import.

Runs in the built image without --gpus:
  docker run --rm --entrypoint python3 tabbyapi:hcfast-r2 /opt/hcfast-r2/test_hcfast_flow_cpu.py
"""
import importlib
import os
import subprocess
import sys

SERVED = {"EXL3_HC_MIX_V2": "1", "EXL3_HC_MIX_V2_MIN_R": "1", "EXL3_HC_MIX_V2_INT8": "1",
          "EXL3_GR_STATE_REGRID": "1", "EXL3_GR_STATE_IN_UP": "1", "EXL3_HC_APPLY_WARP1": "1"}
for k, v in SERVED.items():
    os.environ[k] = v
for k in ("EXL3_HC_MIX_V3", "EXL3_HC_MIX_V3_DOTS_B", "EXL3_HC_MIX_V3_UP_B", "EXL3_HC_MIX_V3_DOTS_J",
          "EXL3_HC_MIX_V3_DOTS_PF", "EXL3_HC_MIX_V3_UP_Q", "EXL3_HC_MIX_V3_PDL"):
    os.environ.pop(k, None)

import torch
import exllamav3.modules.hyperconnections as hcm
from exllamav3.modules.hyperconnections import GatedResidual

H, D, LR = 4, 16, 8
CALLS = []


class FakeExt:
    def __getattr__(self, name):
        def rec(*args):
            CALLS.append((name, args))
            if name in ("gr_mix_v2_int8_statein", "gr_mix_v2_int8_v3", "gr_mix_v2_int8_regrid", "gr_mix_v2_int8"):
                s3, mixed, post = args[0], args[10], args[9]
                mixed.copy_(s3.sum(1).half())
                if post is not None:
                    post.copy_(s3.mean(-1))
                return 3 if name == "gr_mix_v2_int8_v3" else None
            if name == "rms_norm":
                args[2].copy_(args[0].half())
                return None
            raise AssertionError(f"unexpected ext call {name}")
        return rec


def make_site(use_combine: bool, seed: int):
    g = torch.Generator().manual_seed(seed)
    m = GatedResidual(None, f"site{seed}", H, D, 1e-6, use_combine = use_combine)
    m.norm_w_raw = torch.randn(H * D, generator = g) * 0.05
    inject = (torch.randn(H, H * D, generator = g) / 10).half() if use_combine else None
    m._prepare((torch.randn(LR, H * D, generator = g) / 10).half(),
               (torch.randn(H * D, LR, generator = g) / 10).half(), inject)
    assert m.fn_q is not None and m.upx_q is not None
    return m


def set_level(level: int):
    GatedResidual.MIX_V3_LEVEL = level
    GatedResidual.MIX_V3 = level > 0


def run(m, R, level: int):
    set_level(level)
    CALLS.clear()
    s = torch.randn((1, R, H, D), generator = torch.Generator().manual_seed(R))
    post, mixed = m._mix(s, cached = False)
    return list(CALLS), post, mixed


def main():
    hcm.ext = FakeExt()
    assert hcm._HC_MIX_V3_BUILD == "r2"
    G = GatedResidual
    assert G.MIX_V3 is False and G.MIX_V3_LEVEL == 0, "EXL3_HC_MIX_V3 must default off"
    assert G.MIX_V3_DOTS_B == (8,) * 33 and G.MIX_V3_UP_B == (8,) * 33
    assert G.MIX_V3_DOTS_J == (4,) * 33 and G.MIX_V3_DOTS_PF == (0,) * 33 and G.MIX_V3_UP_Q == (2,) * 33
    assert G.MIX_V3_PDL is False
    n = 0
    for use_combine in (True, False):
        m = make_site(use_combine, 7 if use_combine else 8)
        for R in (1, 2, 3, 4, 5, 8, 9, 16, 17, 32):
            off, p0, x0 = run(m, R, 0)
            for level in (1, 2):
                # level 2 with a table: the per-R values must reach the call
                G.MIX_V3_DOTS_B = tuple(4 if r <= 4 else 2 for r in range(33)) if level == 2 else (8,) * 33
                G.MIX_V3_DOTS_J = tuple(8 if r > 8 else 4 for r in range(33))
                G.MIX_V3_DOTS_PF = tuple(r % 3 for r in range(33))
                G.MIX_V3_UP_Q = tuple(4 if r >= 16 else 2 for r in range(33))
                G.MIX_V3_PDL = (R % 2 == 0)
                on, p1, x1 = run(m, R, level)
                assert [c[0] for c in off] == ["gr_mix_v2_int8_statein"], (R, off)
                assert [c[0] for c in on] == ["gr_mix_v2_int8_v3"], (R, on)
                a_off, a_on = off[0][1], on[0][1]
                assert len(a_off) == 11 and len(a_on) == 17, (len(a_off), len(a_on))
                for i, (a, b) in enumerate(zip(a_off, a_on[:11])):
                    if isinstance(a, torch.Tensor):
                        assert isinstance(b, torch.Tensor) and a.shape == b.shape and a.dtype == b.dtype, (R, i)
                        if 1 <= i <= 5:                # weights/scales: the same module tensors
                            assert a is b, (R, i)
                    else:
                        assert a == b, (R, i, a, b)
                if level == 1:
                    assert a_on[11:] == (3, 8, 8, 4, 0, 2), a_on[11:]
                else:
                    exp = (15 | (16 if R % 2 == 0 else 0), 4 if R <= 4 else 2, 8,
                           8 if R > 8 else 4, R % 3, 4 if R >= 16 else 2)
                    assert a_on[11:] == exp, (R, a_on[11:], exp)
                assert (a_off[9] is None) == (not use_combine) and (a_on[9] is None) == (not use_combine)
                assert torch.equal(x0, x1) and (p0 is None) == (p1 is None) and (p0 is None or torch.equal(p0, p1))
                n += 1
        # R > 32: GEMM path, never the fused kernels, flag or not
        for level in (1, 2):
            on, _, _ = run(m, 33, level)
            assert "gr_mix_v2_int8_v3" not in [c[0] for c in on] and "gr_mix_v2_int8_statein" not in [c[0] for c in on]
    G.MIX_V3_DOTS_B = G.MIX_V3_UP_B = (8,) * 33
    G.MIX_V3_DOTS_J, G.MIX_V3_DOTS_PF, G.MIX_V3_UP_Q, G.MIX_V3_PDL = (4,) * 33, (0,) * 33, (2,) * 33, False
    # V3 requires the state-in-up selection; without it the served selection stands
    G.STATE_IN_UP = False
    m = make_site(True, 9)
    for level in (1, 2):
        on, _, _ = run(m, 4, level)
        assert [c[0] for c in on] == ["gr_mix_v2_int8_regrid"], on
    G.STATE_IN_UP = True
    set_level(0)
    print(f"dispatch: {n} (site form x rows x level) cases OK; R > 32 and STATE_IN_UP=0 untouched")

    # knob parsing happens at import: run fresh interpreters
    code = ("import exllamav3.modules.hyperconnections as h; G = h.GatedResidual; "
            "print(G.MIX_V3_LEVEL, G.MIX_V3_DOTS_B[4], G.MIX_V3_DOTS_B[16], G.MIX_V3_UP_B[16], "
            "G.MIX_V3_DOTS_J[16], G.MIX_V3_DOTS_PF[4], G.MIX_V3_DOTS_PF[16], G.MIX_V3_UP_Q[16], G.MIX_V3_PDL)")
    for env, expect in (({"EXL3_HC_MIX_V3": "1", "EXL3_HC_MIX_V3_DOTS_B": "2", "EXL3_HC_MIX_V3_UP_B": "4"},
                         "1 2 2 4 4 0 0 2 False"),
                        ({"EXL3_HC_MIX_V3": "1"}, "1 8 8 8 4 0 0 2 False"),          # r1 defaults unchanged
                        ({"EXL3_HC_MIX_V3": "2"}, "2 2 2 8 4 1 1 4 False"),          # r2 defaults DOTS_B = 2, PF = 1, UP_Q = 4
                        ({"EXL3_HC_MIX_V3": "2", "EXL3_HC_MIX_V3_DOTS_B": "4:4,32:2", "EXL3_HC_MIX_V3_DOTS_J": "8",
                          "EXL3_HC_MIX_V3_DOTS_PF": "4:2,32:1", "EXL3_HC_MIX_V3_UP_Q": "4", "EXL3_HC_MIX_V3_PDL": "1"},
                         "2 4 2 8 8 2 1 4 True"),
                        ({"EXL3_HC_MIX_V3": "2", "EXL3_HC_MIX_V3_DOTS_B": "4:4,8:8"}, "2 4 8 8 4 1 1 4 False"),
                        ({"EXL3_HC_MIX_V3": "0"}, "0 8 8 8 4 0 0 2 False"),
                        ({"EXL3_HC_MIX_V3": "true"}, "0 8 8 8 4 0 0 2 False")):
        out = subprocess.run([sys.executable, "-c", code], env = {**os.environ, **env},
                             capture_output = True, text = True)
        assert out.returncode == 0 and out.stdout.strip().endswith(expect), (env, out.stdout, out.stderr[-500:])
    for env in ({"EXL3_HC_MIX_V3_DOTS_B": "3"}, {"EXL3_HC_MIX_V3_DOTS_J": "6"}, {"EXL3_HC_MIX_V3_DOTS_PF": "3"},
                {"EXL3_HC_MIX_V3_UP_Q": "8"}, {"EXL3_HC_MIX_V3_DOTS_B": "8:2,4:4"}, {"EXL3_HC_MIX_V3_UP_B": "x"}):
        out = subprocess.run([sys.executable, "-c", code], env = {**os.environ, **env},
                             capture_output = True, text = True)
        assert out.returncode != 0 and "must be one of" in out.stderr, (env, out.stderr[-500:])
    print("env parsing OK")
    print("FLOW PASS")
    return 0


if __name__ == "__main__":
    sys.exit(main())
