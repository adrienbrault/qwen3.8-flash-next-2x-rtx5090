#!/usr/bin/env python3
"""latchain r1 CPU test (no GPU, no torch needed for parts A and B).

A. Data-flow model of the GDN recurrent kernel, served (cuda_recurrent_gated_delta_rule_kernel_128)
   vs EXL3_LC_GDN_RR (..._128_rr), per thread (t, bt), in float32 with bf16 or fp32 state storage.
   The served kernel reads the state from the slot/history buffer before each token (twice) and
   writes it after; the rr kernel reads it once, carries it in registers and reloads the value it
   stored (store rounding applied). Both models run the same arithmetic function per element, so
   this checks the DATA FLOW (which bits each token reads, which buffer slots get written, history
   indexing, first/last token handling, unwritten history never read), not fp contraction; that is
   the GPU parity test's job. History slots start as garbage, so reading an unwritten slot fails.

B. Source lints on the patched tree (default: the installed exllamav3 package; --src PATH):
   * gdn.cu: the rr kernel's value-producing statements are the served kernel's statements with
     the state load replaced by the register (list below); the dispatch picks _rr only under lc_rr,
     and lc_rr requires !channelwise and EXL3_LC_GDN_RR;
   * attention.cpp: every QSA-branch launch between the fork and the join uses qstream, the gather
     and combine use stream, the host call order of launches/record_param/_gr calls is the served
     one plus exactly one extra (the forked branch GEMM, same arguments except xh) — compared with
     --served PATH when given;
   * gated_delta_net.cpp: the b/a GEMV is the only launch on the side stream;
   * every flag defaults off.

C. (torch importable) bc_attn flag parsing: unset -> served compile options; values parse/reject.

Usage:
  python3 test_latchain_cpu.py [--src .../site-packages/exllamav3] [--served <unpatched stack-r2 exllamav3>]
Exit 0 and "CPU PASS" only if every check passes.
"""
import argparse
import importlib.util
import os
import random
import re
import struct
import subprocess
import sys
from pathlib import Path

FAILS = []


def check(cond, msg):
    if not cond:
        FAILS.append(msg)
        print("FAIL:", msg)
    return cond


# ---------------------------------------------------------------------------------------------
# A. data-flow model

def f32(x):
    return struct.unpack("f", struct.pack("f", x))[0]


def bf16_rn(x):
    """float32 -> bf16 (round to nearest even) -> float32, as __float2bfloat16_rn/__bfloat162float."""
    u = struct.unpack("I", struct.pack("f", f32(x)))[0]
    u = (u + 0x7FFF + ((u >> 16) & 1)) & 0xFFFF0000
    return struct.unpack("f", struct.pack("I", u))[0]


def model(kind, HD, SUBK, V_SPLIT, heads, seqlen, bf16_state, history, inputs, state0, garbage):
    """One (head, v column) at a time, all SUBK k-slices, token by token, in float32 with the served
    reduction order (per-slice partials, then slices in bt order). kind: 'served' re-reads the state
    from the buffer twice per token (index s, or the final state for token 0); 'rr' loads it once
    from the final state and carries the value it stored (store rounding applied); 'rr_noreload' is
    a deliberately broken rr (carries the unrounded value) that the test must tell apart for bf16.
    Returns (outputs, buffer); buffer keys (hist_idx, head, row, col), hist_idx 0 = final state."""
    store = bf16_rn if bf16_state else f32
    buf = dict(state0)
    buf.update(garbage)
    V_CHUNK = HD // V_SPLIT
    BTS = HD // SUBK
    outs = {}
    for head in range(heads):
        for col in range(V_SPLIT * V_CHUNK):
            regs = {bt: [buf[(0, head, bt * BTS + e, col)] for e in range(BTS)] for bt in range(SUBK)}
            for s in range(seqlen):
                q, k, v_raw, g_h, beta_h = inputs[(s, head, col)]
                if history:
                    r_idx = 0 if s == 0 else s
                    w_idx = 0 if s == seqlen - 1 else s + 1
                else:
                    r_idx = w_idx = 0

                def cur(bt):
                    if kind == "served":
                        return [buf[(r_idx, head, bt * BTS + e, col)] for e in range(BTS)]
                    return regs[bt]
                parts = []
                for bt in range(SUBK):
                    c = cur(bt)
                    part = 0.0
                    for e in range(BTS):
                        part = f32(part + f32(k[bt * BTS + e] * c[e]))
                    parts.append(part)
                dot1 = 0.0
                for part in parts:
                    dot1 = f32(dot1 + part)
                v = f32(v_raw - f32(dot1 * g_h))
                v_out_total = 0.0
                for bt in range(SUBK):
                    c = cur(bt)
                    v_out = 0.0
                    newreg = []
                    for e in range(BTS):
                        r = bt * BTS + e
                        state = f32(f32(c[e] * g_h) + f32(f32(k[r] * v) * beta_h))
                        buf[(w_idx, head, r, col)] = store(state)
                        newreg.append(state if kind == "rr_noreload" else store(state))   # lc_state_reload
                        v_out = f32(v_out + f32(q[r] * state))
                    regs[bt] = newreg
                    v_out_total = f32(v_out_total + v_out)
                outs[(s, head, col)] = v_out_total
    return outs, buf


def part_a():
    rng = random.Random(1234)
    HD, SUBK = 8, 4
    n = mut = nmut = 0
    for bf16_state in (True, False):
        for history in (True, False):
            for V_SPLIT in (1, 4):
                for seqlen in (1, 2, 3, 4):
                    heads = 2
                    inputs = {}
                    for s in range(seqlen):
                        for head in range(heads):
                            q = [f32(rng.uniform(-1, 1)) for _ in range(HD)]
                            k = [f32(rng.uniform(-1, 1)) for _ in range(HD)]
                            for col in range(HD):
                                inputs[(s, head, col)] = (q, k, f32(rng.uniform(-1, 1)),
                                                          f32(rng.uniform(0.2, 1.0)), f32(rng.uniform(0, 1)))
                    rnd = bf16_rn if bf16_state else f32
                    state0 = {(0, h, r, c): rnd(rng.uniform(-2, 2)) for h in range(heads) for r in range(HD) for c in range(HD)}
                    garbage = {(i, h, r, c): rnd(rng.uniform(-50, 50))
                               for i in range(1, max(seqlen, 1)) for h in range(heads) for r in range(HD) for c in range(HD)}
                    o1, b1 = model("served", HD, SUBK, V_SPLIT, heads, seqlen, bf16_state, history, inputs, state0, garbage)
                    o2, b2 = model("rr", HD, SUBK, V_SPLIT, heads, seqlen, bf16_state, history, inputs, state0, garbage)
                    ok = o1 == o2 and b1 == b2
                    check(ok, f"A: served vs rr differ (bf16_state={bf16_state} history={history} V_SPLIT={V_SPLIT} seqlen={seqlen})")
                    # the served model must never have read garbage: rerun it with different garbage
                    garbage2 = {kk: rnd(v + 7.0) for kk, v in garbage.items()}
                    o3, _ = model("served", HD, SUBK, V_SPLIT, heads, seqlen, bf16_state, history, inputs, state0, garbage2)
                    check(o3 == o1, f"A: served model read an unwritten history slot (seqlen={seqlen})")
                    if bf16_state and seqlen > 1:
                        o4, b4 = model("rr_noreload", HD, SUBK, V_SPLIT, heads, seqlen, bf16_state, history, inputs, state0, garbage)
                        mut += (o4, b4) != (o1, b1)
                        nmut += 1
                    n += 1
    check(nmut > 0 and mut == nmut, f"A: the no-reload mutant was caught in {mut}/{nmut} cases (the model is not sensitive)")
    print(f"[A] rr data-flow model == served model in {n} cases (bf16/fp32 state, history on/off, V_SPLIT 1/4, "
          f"seqlen 1-4); no-reload mutant caught in {mut}/{nmut}")


# ---------------------------------------------------------------------------------------------
# B. source lints

def body_of(src, signature_regex):
    m = re.search(signature_regex, src)
    if not m:
        return None
    i = src.index("{", m.end())
    depth = 0
    for j in range(i, len(src)):
        if src[j] == "{":
            depth += 1
        elif src[j] == "}":
            depth -= 1
            if depth == 0:
                return src[i:j + 1]
    return None


def norm_ws(s):
    return " ".join(s.split())


def part_b(pkg: Path, served: Path | None):
    ext = pkg / "exllamav3_ext"
    gdn = (ext / "gdn.cu").read_text()
    served_k = body_of(gdn, r"void cuda_recurrent_gated_delta_rule_kernel_128\s*\(")
    rr_k = body_of(gdn, r"void cuda_recurrent_gated_delta_rule_kernel_128_rr\s*\(")
    if check(served_k and rr_k, "B: gdn.cu kernels not found"):
        S, Rr = norm_ws(served_k), norm_ws(rr_k)
        # (served statement, rr statement): the value-producing arithmetic
        pairs = [
            ("float sumq = q * q;", "float sumq = q * q;"),
            ("float sumk = k * k;", "float sumk = k * k;"),
            ("sumq += __shfl_xor_sync(0xffffffff, sumq, offset);", "sumq += __shfl_xor_sync(0xffffffff, sumq, offset);"),
            ("q = q * rsqrtf(sumq + 1e-6f);", "q = q * rsqrtf(sumq + 1e-6f);"),
            ("k = k * rsqrtf(sumk + 1e-6f);", "k = k * rsqrtf(sumk + 1e-6f);"),
            ("sum = sum + *sh_k_rd * as_float(*rs_rd);", "sum = sum + *sh_k_rd * st[i * 8 + j];"),
            ("for (int s = 0; s < SUBK; ++s) dot1 += sh_dot1[s][t];", "for (int s = 0; s < SUBK; ++s) dot1 += sh_dot1[s][t];"),
            ("float v = __bfloat162float(gl_v[t]) - dot1 * g_h;", "float v = __bfloat162float(cu_v) - dot1 * g_h;"),
            ("state = state * (CHANNELWISE ? *sh_g_rd : g_h) + *sh_k_rd * v * beta_h;", "state = state * g_h + *sh_k_rd * v * beta_h;"),
            ("store_state(rs_w, state);", "store_state(rs_w, state);"),
            ("v_out = v_out + *sh_q_rd * state;", "v_out = v_out + *sh_q_rd * state;"),
            ("for (int s = 0; s < SUBK; ++s) v_out += sh_dot2[s][t];", "for (int s = 0; s < SUBK; ++s) v_out += sh_dot2[s][t];"),
            ("out[t] = __float2bfloat16_rz(v_out * scale);", "out[t] = __float2bfloat16_rz(v_out * scale);"),
            ("float g_h = CHANNELWISE ? 1.0f : __expf(g[head]);", "float g_h = __expf(cu_g);"),
            ("float beta_h = __bfloat162float(beta[head]);", "float beta_h = __bfloat162float(cu_beta);"),
        ]
        for a, b in pairs:
            check(norm_ws(a) in S, f"B: served kernel statement missing: {a}")
            check(norm_ws(b) in Rr, f"B: rr kernel statement missing: {b}")
        check("st[i * 8 + j] = lc_state_reload(rs_w, state);" in Rr, "B: rr kernel does not reload the stored state")
        check("as_float(*rs_r)" not in Rr and "as_float(*rs_rd)" not in Rr, "B: rr kernel still reads the state per token")
        check(norm_ws("return __bfloat162float(__float2bfloat16_rn(x));") in norm_ws(gdn), "B: bf16 reload rounding missing")
        check(norm_ws("*p = __float2bfloat16_rn(x);") in norm_ws(gdn), "B: bf16 store rounding changed")
    disp = body_of(gdn, r"void cuda_recurrent_gated_delta_rule_gr\s*\(")
    if check(disp is not None, "B: dispatcher not found"):
        check("const bool lc_rr = !channelwise && lc_gdn_rr_enabled();" in disp, "B: lc_rr gating changed")
        n_rr = len(re.findall(r"_128_rr<", disp))
        check(n_rr == 8, f"B: expected 8 rr launches in the dispatcher, found {n_rr}")
        check(len(re.findall(r"&& lc_rr\)", disp)) == 4, "B: every rr branch must be guarded by lc_rr")
    check('getenv("EXL3_LC_GDN_RR")' in gdn, "B: EXL3_LC_GDN_RR not read")

    att = (ext / "libtorch" / "attention.cpp").read_text()
    run_gr = body_of(att, r"void BC_Attention::run_gr\s*\(")
    if check(run_gr is not None, "B: BC_Attention::run_gr not found"):
        i_fork = run_gr.find("lcf->fork(stream);")
        i_q = run_gr.find("graph->capture_stream = qstream;")
        i_join = run_gr.find("lc_join();")
        check(0 <= i_fork < i_q < i_join, "B: fork/branch/join order in run_gr")
        branch = run_gr[i_q:i_join]
        for kname in ("k_qsa_stage", "k_qsa_raw_append", "k_qsa_pool_update", "k_qsa_fewq", "k_qsa_expand"):
            m = re.search(kname + r"->launch\(([^;]*)\);", branch)
            check(m is not None and m.group(1).rstrip().endswith("qstream"), f"B: {kname} not launched on qstream")
        tail = run_gr[i_join:]
        for kname in ("k_qsa_split", "k_qsa_combine", "k_split", "k_combine"):
            m = re.search(kname + r"->launch\(([^;]*)\);", tail)
            check(m is not None and m.group(1).rstrip().endswith("stream") and "qstream" not in m.group(1),
                  f"B: {kname} not launched on the main stream after the join")
        check('lc_env_flag("EXL3_LC_QSA_FORK")' in run_gr and "graph && qsa &&" in run_gr, "B: QSA fork gating")
        check(run_gr.count("exl3_gemm_set_locks_offset(LC_LOCKS_ALT_OFFSET);") == 1 and "LocksReset" in run_gr,
              "B: branch GEMM must use the alternate tile locks with a scoped reset")
        check("xh_flat.narrow(0, xh_flat.numel() / 2, (int64_t) R * hs)" in run_gr, "B: branch GEMM must use xh's upper half")
        if served is not None:
            s_att = (served / "exllamav3_ext" / "libtorch" / "attention.cpp").read_text()
            s_run = body_of(s_att, r"void BC_Attention::run_gr\s*\(")
            call_re = re.compile(r"\b(\w+_gr)\(|(\w+)->launch\(|record_param\(\s*([^,]+),\s*(GP_\w+)")

            def seq(txt):
                out = []
                for m in call_re.finditer(txt):
                    if m.group(1):
                        out.append(m.group(1))
                    elif m.group(2):
                        out.append(m.group(2) + "->launch")
                    else:
                        out.append("record:" + m.group(4))
                return out
            ss, ps = seq(s_run), seq(run_gr)
            # the patched function has exactly one extra exl3_gemm_gr (the forked branch's copy of
            # the QSA projection call, in the if/else pair) and nothing else
            extra = list(ps)
            try:
                k = next(i for i in range(len(extra)) if extra[i:i + 2] == ["exl3_gemm_gr", "exl3_gemm_gr"])
                del extra[k]
            except StopIteration:
                pass
            check(extra == ss, "B: run_gr host call order differs from the served one (beyond the forked GEMM copy)")
            print(f"[B] run_gr call sequence: served {len(ss)} calls, patched {len(ps)} (1 forked GEMM copy)")

    gdnc = (ext / "libtorch" / "gated_delta_net.cpp").read_text()
    rb = body_of(gdnc, r"void BC_GatedDeltaNetSplit::run_bszN_gr\s*\(")
    if check(rb is not None, "B: run_bszN_gr not found"):
        i_side = rb.find("graph->capture_stream = lcf->side;")
        i_back = rb.find("graph->capture_stream = lc_main;")
        seg = rb[i_side:i_back]
        check(0 <= i_side < i_back and seg.count("_gr(") == 1 and "gdn_ba_gemv_gr(" in seg,
              "B: only the b/a GEMV may run on the GDN side stream")
        check('lc_env_flag("EXL3_LC_GDN_FORK")' in rb and "graph && !kda" in rb, "B: GDN fork gating")

    lcf_path = ext / "libtorch" / "lc_fork.h"
    lcf = lcf_path.read_text() if lcf_path.is_file() else ""
    check("!e || !*e || !std::strcmp(e, \"0\")" in lcf, "B: lc_env_flag must default off")
    gemm = (ext / "quant" / "exl3_gemm.cu").read_text()
    check("static thread_local int lc_gemm_locks_offset = 0;" in gemm, "B: lock offset must default to 0")
    bca = (pkg / "modules" / "attention_fn" / "bc_attn.py").read_text()
    check("4, _LC_QSA_SPLIT_STAGES or 2)" in bca and "4, _LC_QSA_COMBINE_STAGES or 1)" in bca,
          "B: served compile options must be the fallback")
    print("[B] source lints done")


# ---------------------------------------------------------------------------------------------
# C. bc_attn flag parsing (subprocesses: the flags are read at import)

def part_c():
    if importlib.util.find_spec("torch") is None or importlib.util.find_spec("exllamav3") is None:
        print("[C] skipped (torch/exllamav3 not importable here)")
        return
    probe = ("import exllamav3.modules.attention_fn.bc_attn as b; "
             "print(b.LC_BUILD, b._LC_QSA_SPLIT_STAGES, b._LC_QSA_COMBINE_STAGES, b._LC_QSA_DIV16)")
    base = {k: v for k, v in os.environ.items() if not k.startswith("EXL3_LC_")}

    def run(extra):
        r = subprocess.run([sys.executable, "-c", probe], env={**base, **extra}, capture_output=True, text=True)
        return r.returncode, r.stdout.strip().splitlines()[-1] if r.stdout.strip() else r.stderr.strip()[-200:]
    rc, out = run({})
    if rc != 0 and "latchain" not in out and "Assert" not in out:
        # bc_attn imports triton and the extension; a host without a CUDA driver may not get that far
        print(f"[C] skipped: bc_attn does not import on this host ({out[-160:]!r})")
        return
    check(rc == 0 and out == "latchain-r1 0 0 False", f"C: defaults: {out}")
    rc, out = run({"EXL3_LC_QSA_SPLIT_STAGES": "4", "EXL3_LC_QSA_COMBINE_STAGES": "3", "EXL3_LC_QSA_DIV16": "1"})
    check(rc == 0 and out == "latchain-r1 4 3 True", f"C: set: {out}")
    rc, _ = run({"EXL3_LC_QSA_SPLIT_STAGES": "9"})
    check(rc != 0, "C: out-of-range stages must be rejected")
    rc, _ = run({"EXL3_LC_QSA_DIV16": "2"})
    check(rc != 0, "C: DIV16=2 must be rejected")
    import exllamav3.modules.attention_fn.bc_attn as b  # noqa: E402
    s = b._lc_div16({"a": "*fp16", "b": "i32", "c": "*i32:16"}, ("a", "b", "c"))
    check(s == {"a": "*fp16:16", "b": "i32:16", "c": "*i32:16"}, f"C: _lc_div16 {s}")
    print("[C] bc_attn flag parsing done")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", type=Path, default=None, help="patched exllamav3 package dir (default: installed)")
    ap.add_argument("--served", type=Path, default=None, help="unpatched stack-r2 exllamav3 dir, for the call-order check")
    a = ap.parse_args()
    pkg = a.src
    if pkg is None:
        spec = importlib.util.find_spec("exllamav3")
        assert spec and spec.origin, "exllamav3 not installed; pass --src"
        pkg = Path(spec.origin).parent
    part_a()
    part_b(pkg, a.served)
    if a.src is None:
        part_c()
    else:
        print("[C] skipped (--src given; run without --src inside the image)")
    if FAILS:
        print(f"CPU FAIL ({len(FAILS)} checks)")
        sys.exit(1)
    print("CPU PASS")


if __name__ == "__main__":
    main()
