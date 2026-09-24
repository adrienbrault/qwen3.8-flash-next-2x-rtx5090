#!/usr/bin/env python3
"""Serial global-load rounds per kernel, from a cuobjdump -sass listing that carries decoded control
info (`&wr=0xN` on a producer, `&req={..}` on a consumer; CUDA 12.8's cuobjdump prints them).

Every LDG marks its write scoreboard as pending. An instruction whose &req names a pending scoreboard
waits for those loads (a "wait"). A ROUND is a batch of loads issued before the first wait that
retires any of them; consecutive waits with no LDG issued in between belong to the same round (their
loads were all in flight together). Loads issued after a wait open the next round.

Output per kernel: the compressed event stream in address order, with `ldN` = N loads issued back to
back, `w(k)` = a wait retiring k pending loads, `[Rn]` = where round n's first wait falls, `fwd` / `<loop`
= a forward / backward branch (a backward branch closes a rolled loop body: its rounds repeat per
iteration), EXIT / BAR. The dots sum-of-squares CTA and the main CTAs are two branches of one kernel:
the stream shows the sum-of-squares path first (up to its EXITs), then the main path.

usage: sass_rounds.py listing.sass 'dots_i8_v4_kernel<2, 4, 0>' [...]   (substring match, spaces ignored)
"""
import re
import sys

HEAD = re.compile(r"/\*([0-9a-f]{4,5})\*/\s+(@!?U?P[0-9T]\s+)?([A-Z][A-Z0-9_.]*)\s*([^&?;]*)")
WR = re.compile(r"&wr=0x(\d)")
REQ = re.compile(r"&req=\{([0-9,]+)\}")
BRA_T = re.compile(r"0x([0-9a-f]+)")


def funcs(path):
    cur, body = None, []
    for ln in open(path):
        if "Function :" in ln:
            if cur:
                yield cur, body
            cur, body = ln.split("Function :", 1)[1].strip(), []
        elif cur:
            body.append(ln)
    if cur:
        yield cur, body


def analyse(body):
    pending = {}
    ev = []
    for ln in body:
        m = HEAD.search(ln)
        if not m:
            continue
        addr, _, op, rest = m.groups()
        a = int(addr, 16)
        w = WR.search(ln)
        q = REQ.search(ln)
        if q:
            hit = [int(x) for x in q.group(1).split(",") if int(x) in pending]
            if hit:
                ev.append(("wait", sum(pending.pop(s) for s in hit)))
        if op.startswith("LDG"):
            if w:
                pending[int(w.group(1))] = pending.get(int(w.group(1)), 0) + 1
            ev.append(("ld", 1))
        elif op.startswith("BRA"):
            t = BRA_T.search(rest)
            if t and int(t.group(1), 16) < a:
                ev.append(("<loop", 0))
            elif t and int(t.group(1), 16) > a + 16:
                ev.append(("fwd", 0))
        elif op == "EXIT":
            ev.append(("EXIT", 0))
        elif op.startswith("BAR"):
            ev.append(("BAR", 0))
    return ev


def render(ev):
    out, rounds, open_batch = [], 0, False
    i = 0
    while i < len(ev):
        kind, n = ev[i]
        if kind == "ld":
            k = 0
            while i < len(ev) and ev[i][0] == "ld":
                k += 1
                i += 1
            out.append(f"ld{k}")
            open_batch = True
            continue
        if kind == "wait":
            if open_batch:
                rounds += 1
                out.append(f"[R{rounds}]")
                open_batch = False
            out.append(f"w({n})")
        elif kind == "fwd" and out and out[-1] == "fwd":
            pass
        else:
            out.append(kind)
        i += 1
    return " ".join(out), rounds


def main():
    path, pats = sys.argv[1], [p.replace(" ", "") for p in sys.argv[2:]]
    for name, body in funcs(path):
        if not any(p in name.replace(" ", "") for p in pats):
            continue
        s, rounds = render(analyse(body))
        print("==", name.split("(")[0])
        print("  ", s)
        print(f"   rounds, static count over all paths in address order: {rounds}")


if __name__ == "__main__":
    main()
