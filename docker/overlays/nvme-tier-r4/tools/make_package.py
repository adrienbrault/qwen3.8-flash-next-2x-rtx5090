#!/usr/bin/env python3
"""
Build the round-3 deliverable from the work trees:

  work/exllamav3/...      src/ + this round's edits (disk_cache.py, pagetable.py, recurrent.py, async_generator.py);
                          generator.py is regenerated here from src/ by port_generator.py
  work-tip/exllamav3/...  the same + recurrent-tip-r1's recurrent_tip.py; generator.py regenerated from tip's
  generator.py for the mtp-pruned-r1 base is regenerated from ref/mtp-pruned-r1's overlay generator.py (round 4)

Outputs (paths relative to the deliverable directory):
  overlay/exllamav3/<file>                          files for the stack-r4-e3r2 base (site-packages relative)
  overlay-tip/exllamav3/generator/generator.py      generator.py for the stack-r4-e3r2 + recurrent-tip-r1 base
  overlay-pruned/exllamav3/generator/generator.py   generator.py for the stack-r4-e3r2 + mtp-pruned-r1 base (the served
                                                    chain: + tool-choice-r1 + ple-ckpt-clone-r1 leave generator.py alone)
  manifest.json                                     all baseline sets, overlay hashes, soft dependencies
  served-source.patch                               diff -ruN against src/ (base variant)
  served-source-tip.patch                           generator.py diff against recurrent-tip-r1's generator.py
  served-source-pruned.patch                        generator.py diff against mtp-pruned-r1's generator.py
"""
import hashlib
import json
import pathlib
import shutil
import subprocess
import sys

TOOLS = pathlib.Path(__file__).resolve().parent
DELIV = TOOLS.parent
ROOT = DELIV.parent.parent
SRC = ROOT / "src"
TIP = ROOT / "ref" / "recurrent-tip-r1" / "overlay"
PRUNED = ROOT / "ref" / "mtp-pruned-r1" / "overlay"          # its paths are site-packages relative too (exllamav3/...)
WORK = DELIV / "work"
WORK_TIP = DELIV / "work-tip"

FILES = [
    "exllamav3/cache/recurrent.py",
    "exllamav3/generator/async_generator.py",
    "exllamav3/generator/disk_cache.py",
    "exllamav3/generator/generator.py",
    "exllamav3/generator/pagetable.py",
]
DEPENDS = [
    "exllamav3/constants.py",
    "exllamav3/tokenizer/mm_embedding.py",
    "exllamav3/generator/job.py",
    "exllamav3/generator/cpu_cache.py",
    "exllamav3/cache/quant.py",
    "exllamav3/cache/qsa.py",
    "exllamav3/modules/gated_delta_net.py",
]
TIP_FILE = "exllamav3/cache/recurrent_tip.py"
# mtp-pruned-r1's other files: this round does not replace them; report (do not refuse) a different version
PRUNED_DEPENDS = [
    "exllamav3/architecture/qwen4_exp_mtp.py",
    "exllamav3/modules/embedding.py",
    "exllamav3/modules/embedding_pruned.py",
]
GEN = "exllamav3/generator/generator.py"


# ple.py of stack-r4-e3r2 after ple-ckpt-clone r1 (ref/ple-ckpt-clone-r1/fix_ple.py POST): what R526 try 5 ran
PLE_FIXED = "cbd597f6507f0d82f0cf59608340985cab339f6029a88ee75d7c04128164db44"


def sha(p: pathlib.Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def main():
    sys.path.insert(0, str(TOOLS))
    import port_generator
    (WORK / GEN).write_text(port_generator.port((SRC / GEN).read_text()))
    # work-tip mirrors work for every shared file
    for rel in FILES:
        if rel != GEN:
            (WORK_TIP / rel).parent.mkdir(parents = True, exist_ok = True)
            shutil.copyfile(WORK / rel, WORK_TIP / rel)
    shutil.copyfile(TIP / TIP_FILE, WORK_TIP / TIP_FILE)
    (WORK_TIP / GEN).write_text(port_generator.port((TIP / GEN).read_text()))
    pruned_gen = port_generator.port((PRUNED / GEN).read_text())

    for d in ("overlay", "overlay-tip", "overlay-pruned"):
        shutil.rmtree(DELIV / d, ignore_errors = True)
    for rel in FILES:
        dst = DELIV / "overlay" / rel
        dst.parent.mkdir(parents = True, exist_ok = True)
        shutil.copyfile(WORK / rel, dst)
    dst = DELIV / "overlay-tip" / GEN
    dst.parent.mkdir(parents = True, exist_ok = True)
    shutil.copyfile(WORK_TIP / GEN, dst)
    dst = DELIV / "overlay-pruned" / GEN
    dst.parent.mkdir(parents = True, exist_ok = True)
    dst.write_text(pruned_gen)

    def src_sha(rel):
        p = SRC / rel
        return sha(p) if p.exists() else None

    base_pre = {rel: src_sha(rel) for rel in FILES}
    tip_pre = dict(base_pre)
    tip_pre[GEN] = sha(TIP / GEN)
    # recurrent_tip.py is not replaced by this round: require it to be present, report (do not refuse) a different
    # version, so a tip bug fix that leaves generator.py alone does not block the stacked build
    post = {rel: {"from": f"overlay/{rel}", "sha256": sha(DELIV / "overlay" / rel)} for rel in FILES}
    tip_post = dict(post)
    tip_post[GEN] = {"from": f"overlay-tip/{GEN}", "sha256": sha(DELIV / "overlay-tip" / GEN)}
    # mtp-pruned-r1 replaces generator.py only among this round's files (its manifest: qwen4_exp_mtp.py, embedding.py,
    # generator.py, + embedding_pruned.py); the other four baselines are the stack's
    pruned_pre = dict(base_pre)
    pruned_pre[GEN] = sha(PRUNED / GEN)
    pruned_post = dict(post)
    pruned_post[GEN] = {"from": f"overlay-pruned/{GEN}", "sha256": sha(DELIV / "overlay-pruned" / GEN)}
    manifest = {
        "round": "nvme-tier-r4",
        "flag": "EXL3_NVME_TIER",
        # informational: install.py detects the variant from file hashes; every base must carry ple-ckpt-clone r1
        "base_images": {
            "stack-r4-e3r2": "tabbyapi:stack-r4-e3r2 + ple-ckpt-clone r1",
            "stack-r4-e3r2+recurrent-tip-r1": "tabbyapi:recurrent-tip-r1 + ple-ckpt-clone r1",
            "stack-r4-e3r2+mtp-pruned-r1": "tabbyapi:mtp-pruned-r1-tc1-plefix (served), or any mtp-pruned-r1 descendant "
                                           "with ple-ckpt-clone r1",
        },
        "paths": "relative to site-packages",
        "variants": {
            "stack-r4-e3r2": {"pre": base_pre, "post": post},
            "stack-r4-e3r2+recurrent-tip-r1": {"pre": tip_pre, "post": tip_post, "present": [TIP_FILE],
                                               "depends": {TIP_FILE: sha(TIP / TIP_FILE)}},
            "stack-r4-e3r2+mtp-pruned-r1": {"pre": pruned_pre, "post": pruned_post, "present": PRUNED_DEPENDS,
                                            "depends": {rel: sha(PRUNED / rel) for rel in PRUNED_DEPENDS}},
        },
        # checked on every variant before anything else: the tier needs the PLE checkpoint fix (R526 try 4 vs try 5)
        "requires": {"exllamav3/modules/ple.py": {
            "contains": "        return (self.conv_state[slot, :, :self.win].cpu(), self.id_state[slot, :self.ctx].clone())\n",
            "why": "ple-ckpt-clone r1 (PLELayerState.stash() must .clone() the host id context)",
            "sha256_tested": PLE_FIXED}},
        "depends": {rel: src_sha(rel) for rel in DEPENDS},
    }
    (DELIV / "manifest.json").write_text(json.dumps(manifest, indent = 1) + "\n")

    # served-source.patch: src/ vs src/ + base overlay, touched files only
    tmp = DELIV / ".patch-tmp"
    shutil.rmtree(tmp, ignore_errors = True)
    for side, tree in (("a", SRC), ("b", DELIV / "overlay")):
        for rel in FILES:
            p = tree / rel
            if p.exists():
                (tmp / side / rel).parent.mkdir(parents = True, exist_ok = True)
                shutil.copyfile(p, tmp / side / rel)
    out = subprocess.run(["diff", "-ruN", "a", "b"], cwd = tmp, capture_output = True, text = True).stdout
    (DELIV / "served-source.patch").write_text(out)
    shutil.rmtree(tmp)
    tmp.mkdir()
    (tmp / "a" / GEN).parent.mkdir(parents = True)
    (tmp / "b" / GEN).parent.mkdir(parents = True)
    shutil.copyfile(TIP / GEN, tmp / "a" / GEN)
    shutil.copyfile(DELIV / "overlay-tip" / GEN, tmp / "b" / GEN)
    out = subprocess.run(["diff", "-ruN", "a", "b"], cwd = tmp, capture_output = True, text = True).stdout
    (DELIV / "served-source-tip.patch").write_text(out)
    shutil.copyfile(PRUNED / GEN, tmp / "a" / GEN)
    shutil.copyfile(DELIV / "overlay-pruned" / GEN, tmp / "b" / GEN)
    out = subprocess.run(["diff", "-ruN", "a", "b"], cwd = tmp, capture_output = True, text = True).stdout
    (DELIV / "served-source-pruned.patch").write_text(out)
    shutil.rmtree(tmp)
    print(json.dumps({k: {"pre": len(v["pre"]), "post": len(v["post"])} for k, v in manifest["variants"].items()}))


if __name__ == "__main__":
    main()
