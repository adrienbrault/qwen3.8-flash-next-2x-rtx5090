# stack-r2

`tabbyapi:stack-r2` was the image served from 2026-09-24 10:46 to 22:10 CEST ([R701](../../../bench/results/r701-stack-r2.md)) and is the base of `tabbyapi:stack-r3` ([`../stack-r3/README.md`](../stack-r3/README.md)). It is `tabbyapi:slotfix-r1` with two overlays applied in sequence. Each overlay applies its patch to the installed `exllamav3` package with `patch --fuzz=0` after a dry run, rebuilds `exllamav3_ext` for `sm_120` and asserts its landing marker. Both patches touch `exllamav3_ext/bindings.cpp`, and they apply in this order at fuzz 0 with no offsets on the `slotfix-r1` tree.

Build order, from the repository root:

```sh
# 1. slotfix-r1 on stack-r1 (R676)
docker build -f docker/overlays/slotfix-r1/Dockerfile.box -t tabbyapi:slotfix-r1 docker/overlays/slotfix-r1

# 2. hcfast-r1 on slotfix-r1 (R698, R699)
docker build -f docker/overlays/hcfast-r1/Dockerfile.box --build-arg BASE=tabbyapi:slotfix-r1 \
  --build-arg MAX_JOBS=4 -t tabbyapi:hcfast-r1 docker/overlays/hcfast-r1

# 3. moefast-r1 on hcfast-r1 (R700b); the result is stack-r2
docker build -f docker/overlays/moefast-r1/Dockerfile.box --build-arg BASE=tabbyapi:hcfast-r1 \
  --build-arg MAX_JOBS=4 -t tabbyapi:stack-r2 docker/overlays/moefast-r1

# landing check: both markers
docker run --rm --entrypoint python3 tabbyapi:stack-r2 -c 'import torch, exllamav3_ext as e, exllamav3.modules.hyperconnections as h; print("hcfast", h._HC_MIX_V3_BUILD, "moefast", e.moe_coop_v3_revision)'
# expected: hcfast r1 moefast 1
```

The `slotfix-r1` Dockerfile starts `FROM tabbyapi:stack-r1`; the chain below that is in [`docker/README.md`](../../README.md). The moefast overlay's Dockerfile defaults to `BASE=tabbyapi:slotfix-r1`, which builds `tabbyapi:moefast-r1` alone (the R700b image); `BASE=tabbyapi:hcfast-r1` builds the stack.

Every change in the two overlays is off unless its environment key is set. The served launcher sets all four:

| key | overlay | value served |
| --- | --- | --- |
| `EXL3_HC_MIX_V3` | hcfast-r1 | `1` |
| `EXL3_HC_MIX_V3_DOTS_B` | hcfast-r1 | `2` (the default 8 regressed at 8 streams, [R698](../../../bench/results/r698-hcfast.md)) |
| `EXL3_HC_MIX_V3_UP_B` | hcfast-r1 | `8` |
| `EXL3_MOE_COOP_V3` | moefast-r1 | `2` |

Rollback is `IMG=tabbyapi:slotfix-r1` with the four keys removed from `EXTRA_ENV`; with the keys unset, `tabbyapi:stack-r2` runs the `slotfix-r1` kernels.
