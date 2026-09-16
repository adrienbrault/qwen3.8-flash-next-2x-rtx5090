#!/usr/bin/env python3
"""Matched SWE-bench comparison: this seat's subsets against the daily's full scored run.

Why it exists: the daily has a 500-instance scored run on this box, so any subset of instances this seat runs can be
compared against the daily's outcome *on those same instances* — no sampling error on the comparison, only on how
well the subset represents the 500. The alternative (comparing subset scores to 387/500) mixes sampling error into
a number that the repository already says carries coin-flip variance at k=1.

The trap this tool avoids: `preds.json` is ordered by completion, not by dataset order. Reading "the first ten" out
of it picks the first ten *finished* instances, which are not the first ten the harness ran. The instance list comes
from the trajectory files, which are named by instance id.

usage: match-daily-swebench.py <this-run-dir> [<daily-run-dir>]

Prints the substituted instances, the daily's outcome and its own overall figure, then the subset tally.
"""
import glob, json, os, sys


def traj_instances(run_dir):
    """The instances this run actually executed, from the trajectory files."""
    out = {}
    for f in glob.glob(os.path.join(run_dir, "out", "*", "*.traj.json")):
        iid = os.path.basename(f).replace(".traj.json", "")
        try:
            t = json.load(open(f))
        except Exception:
            continue
        info = t.get("info", {})
        out[iid] = {"exit": info.get("exit_status"), "steps": len(t.get("messages", [])),
                    "patch_chars": len(info.get("submission") or "")}
    return out


def daily_report(daily_dir):
    """The daily's report: resolved ids, plus what it submitted and how it exited."""
    cands = sorted(glob.glob(os.path.join(daily_dir, "qwen3.8-27b-miniswe*.json")))
    if not cands:
        return None
    r = json.load(open(cands[0]))
    return {"resolved": set(r.get("resolved_ids") or []),
            "submitted": r.get("submitted_instances"), "completed": r.get("completed_instances"),
            "errors": r.get("error_instances"), "empty": r.get("empty_patch_instances"),
            "source": os.path.relpath(cands[0], daily_dir)}


def score_of(run_dir):
    """This run's official score, when the scorer has written it."""
    for f in glob.glob(os.path.join(run_dir, "qwen*.json")):
        try:
            r = json.load(open(f))
        except Exception:
            continue
        if "resolved_ids" in r:
            return len(set(r.get("resolved_ids") or [])), r.get("submitted_instances")
    return None


def main():
    mine_dir = sys.argv[1]
    daily_dir = sys.argv[2] if len(sys.argv) > 2 else "/srv/qwen5090/results/2026-09-02-miniswe-rh-nvidia"
    mine = traj_instances(mine_dir)
    daily = daily_report(daily_dir)
    if daily is None:
        print(f"no daily report under {daily_dir}")
        return
    print(f"daily: {daily['resolved'].__len__() if isinstance(daily['resolved'], set) else daily['resolved']} "
          f"resolved of {daily['submitted']} submitted, {daily['completed']} completed, "
          f"{daily['errors']} errors, {daily['empty']} empty patches  [{daily['source']}]")
    print(f"this run: {len(mine)} instances finished so far")
    print()
    print(f"{'instance':<48}{'daily':<10}{'exit':<20}{'steps':>6}{'patch':>8}")
    tally = 0
    for iid in sorted(mine):
        d = "RESOLVED" if iid in daily["resolved"] else "-"
        m = mine[iid]
        print(f"{iid[:46]:<48}{d:<10}{str(m['exit'])[:18]:<20}{m['steps']:>6}{m['patch_chars']:>8}")
    resolved_of_these = sum(1 for iid in mine if iid in daily["resolved"])
    print()
    print(f"the daily resolved {resolved_of_these} of these {len(mine)} instances")
    s = score_of(mine_dir)
    print(f"this seat's official score on this subset: {s[0]}/{s[1]}" if s else
          "this seat's official score: not written yet (the scorer has not run)")


if __name__ == "__main__":
    main()
