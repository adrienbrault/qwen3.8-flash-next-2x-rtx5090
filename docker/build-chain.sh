#!/usr/bin/env bash
# Builds the image scripts/launch-flashnext.sh serves (DAILY_IMG), from a clean clone, in one command:
#   bash docker/build-chain.sh
# 35 layers in the order of docker/README.md, each tagged the way the launcher and the next layer's BASE expect:
#   qsa-cid                                   Dockerfile.tabbyapi-qsa-cid     TabbyAPI 53da7919 + ExLlamaV3 v1.5.0 on the CUDA devel base,
#                                                                              QSA multi-job + draft depth, native rebuild
#   qsa-cid-pr337                             Dockerfile.tabbyapi-pr337       exllamav3#337
#   …-bszn16, …-coopwide                      Dockerfile.tabbyapi-bszn        bszn16.patch, coopwide.patch (native rebuild each)
#   …-coopwide-hcmix1, …-coopwide-hcmix2      Dockerfile.tabbyapi-bszn        mixer V2 round 1 (hc-mix-v2.patch), then round 2 on it
#   …-hostgap, …-ppipe                        Dockerfile.tabbyapi-pyfile / -pypatch
#   …-nosync, …-mtpfix2, …-moecoopv2          overlays/prefill-nosync-overlay, prefill-pipeline-mtp-overlay, moe-coop-v2-overlay
#   decode-kernels-r2 … mtpwin-r2-metrics1    one overlay each (overlays/<name>/)
#   bverify-r1, mtpnorm-r1, mixstate-r1, stack-r1   Dockerfile.tabbyapi-bverify / -mtpnorm / -mixstate / -prefbatch
#   slotfix-r1, hcfast-r1, stack-r2, stack-r3, stack-r3-rows32   overlays (stack-r2 = moefast-r1)   <- the served tag
# The first image of docker/README.md's table (tabbyapi:53da7919-rqcount, Dockerfile.tabbyapi) is not built: no layer uses it as
# its base, Dockerfile.tabbyapi-qsa-cid starts again from nvidia/cuda:12.8.1-devel-ubuntu24.04.
#
# Requirements: x86_64 Linux, Docker with BuildKit, and a builder that sees locally built tags (`docker buildx use default`,
# the `docker` driver): every layer after the first is `FROM <previous tag>`. No GPU is used; CUDA code compiles for sm_120
# with nvcc from the CUDA devel base. The native rebuilds (qsa-cid, the four -bszn layers, mixstate-r1, hcfast-r1, stack-r2,
# stack-r3, stack-r3-rows32 and the JIT builds of the overlays) dominate the time; MAX_JOBS sets their parallelism.
# Network: GitHub (TabbyAPI, the ExLlamaV3 wheel), PyPI and the PyTorch index (TabbyAPI's cu12 extra), Docker Hub (the CUDA base).
# Cost (R739, 2026-09-26, bench/results/r739-build-verify.md): 57 min on a Ryzen 7 9800X3D with MAX_JOBS=12 and the CUDA base
# already pulled, niced beside a serving engine; 59 GB of disk beyond the base; the final image is 48 GB. TabbyAPI's Python
# dependencies are pinned by docker/constraints-stack-r3-rows32.txt (R745).
#
# Knobs:
#   DRY_RUN=1   print every docker command and check the COPY sources of every Dockerfile; runs no docker command
#   NO_CACHE=1  pass --no-cache to every layer, so nothing is reused from an earlier build on this host
#   FORCE=1     rebuild tags that already exist (default: an existing tag is skipped, so a failed run resumes where it stopped).
#               With the default IMAGE_REPO this replaces the tags the launcher serves.
#   IMAGE_REPO  repository name for all tags (default tabbyapi, the one the launcher serves). Another name builds the chain
#               beside the served one without touching it; the final check then compares the tag after the colon.
#   MAX_JOBS    parallel compile jobs, passed to every Dockerfile that takes it (default: each Dockerfile's own, 4 to 6)
#   LOG_DIR     per-layer build logs (default ./build-logs; *.log is gitignored)
#   DOCKER="sudo docker"   the docker command (default docker)
set -euo pipefail
cd "$(dirname "$0")/.."

DRY_RUN=${DRY_RUN:-0}; NO_CACHE=${NO_CACHE:-0}; FORCE=${FORCE:-0}
read -r -a DOCKER_CMD <<< "${DOCKER:-docker}"
IMAGE_REPO=${IMAGE_REPO:-tabbyapi}
LOG_DIR=${LOG_DIR:-build-logs}
LAUNCHER=scripts/launch-flashnext.sh
D=docker; O=docker/overlays
JOBS=(); [ -n "${MAX_JOBS:-}" ] && JOBS=(--build-arg MAX_JOBS="$MAX_JOBS")
export DOCKER_BUILDKIT=1

log(){ echo "$(date -u +%Y-%m-%dT%H:%M:%SZ) $*"; }
die(){ log "FAILED: $*"; exit 1; }
sha(){ if command -v sha256sum >/dev/null 2>&1; then sha256sum "$1" | cut -c1-64; else shasum -a 256 "$1" | cut -c1-64; fi; }

EXPECT=$(sed -nE 's/^DAILY_IMG=([^ ]+).*/\1/p' "$LAUNCHER" | head -1)
[ -n "$EXPECT" ] || die "no DAILY_IMG= line in $LAUNCHER"

# Every source a Dockerfile COPYs must exist in its build context and must not be excluded by the context's .dockerignore
# (checked in DRY_RUN too). ${PATCH} and ${SRC} resolve from the --build-arg values of the call.
check_copy_sources(){ # $1 Dockerfile, $2 context, rest = build args
  local df=$1 ctx=$2 missing=0 line src pat rel p a patch_arg="" src_arg=""; shift 2
  while [ $# -gt 0 ]; do
    if [ "$1" = --build-arg ]; then case $2 in PATCH=*) patch_arg=${2#PATCH=};; SRC=*) src_arg=${2#SRC=};; esac; shift; fi
    shift; done
  [ -f "$df" ] || die "$df missing"
  local ign=()
  if [ -f "$ctx/.dockerignore" ]; then while IFS= read -r p; do [ -n "$p" ] && ign+=("${p%/}"); done < "$ctx/.dockerignore"; fi
  while IFS= read -r line; do
    set -- $line; shift
    while [ $# -gt 1 ]; do src=$1; shift; case $src in --*) continue;; esac
      src=${src//\$\{PATCH\}/$patch_arg}; src=${src//\$\{SRC\}/$src_arg}
      case $src in ''|*'$'*) log "unresolved COPY source in $df: '$src'"; missing=1; continue;; esac
      [ -e "$ctx/$src" ] || { log "missing COPY source in $df: $ctx/$src"; missing=1; continue; }
      rel=${src%/}; rel=${rel#./}
      for pat in ${ign[@]+"${ign[@]}"}; do p=$rel
        while :; do
          # shellcheck disable=SC2254
          case $p in $pat|${pat#\*\*/}) log "COPY source $src of $df is excluded by $ctx/.dockerignore ($pat)"; missing=1; break 2;; esac
          case $p in */*) p=${p%/*};; *) break;; esac
        done
      done
    done
  done < <(sed -e ':a' -e '/\\$/N; s/\\\n//; ta' "$df" | grep -E '^COPY ')
  [ $missing = 0 ] || die "$df has missing COPY sources"; }

N=0
build(){ # $1 tag, $2 Dockerfile, $3 context, rest = docker build args
  local tag=$1 df=$2 ctx=$3 rc start lf attempt; shift 3
  N=$((N + 1))
  check_copy_sources "$df" "$ctx" "$@"
  local cmd=("${DOCKER_CMD[@]}" build --progress=plain -f "$df" "$@" -t "$tag" "$ctx")
  [ "$NO_CACHE" = 1 ] && cmd=("${cmd[@]:0:${#DOCKER_CMD[@]}+1}" --no-cache "${cmd[@]:${#DOCKER_CMD[@]}+1}")
  if [ "$DRY_RUN" = 1 ]; then printf '[%02d] ' "$N"; printf '%q ' "${cmd[@]}"; echo; return 0; fi
  if [ "$FORCE" != 1 ] && "${DOCKER_CMD[@]}" image inspect "$tag" >/dev/null 2>&1; then log "--- [$N] $tag exists, skipped (FORCE=1 rebuilds)"; return 0; fi
  lf="$LOG_DIR/$(printf %02d "$N")-${tag##*:}.log"
  for attempt in 1 2 3; do
    log "--- [$N] build $tag ($df) attempt $attempt, log $lf"
    start=$(date +%s); rc=0; "${cmd[@]}" > "$lf" 2>&1 || rc=$?
    [ $rc = 0 ] && { log "[$N] $tag OK in $(( $(date +%s) - start )) s"; return 0; }
    tail -15 "$lf" | cut -c1-240 | sed "s/^/    [${tag##*:}] /"
    # containerd content-store race while another build or pull runs on the same host: retry
    if grep -aq "failed to export layer\|failed to commit: rename" "$lf"; then log "containerd layer-export race, retrying in 60 s"; sleep 60; continue; fi
    break
  done
  die "[$N] build $tag rc=$rc (log $lf)"; }

log "=== build of ${IMAGE_REPO}:${EXPECT#*:} (DRY_RUN=$DRY_RUN NO_CACHE=$NO_CACHE FORCE=$FORCE MAX_JOBS=${MAX_JOBS:-default}) ==="
if [ "$DRY_RUN" != 1 ]; then
  mkdir -p "$LOG_DIR"
  "${DOCKER_CMD[@]}" buildx version >/dev/null 2>&1 || die "'${DOCKER_CMD[*]} buildx' is not available (BuildKit)"
  drv=$("${DOCKER_CMD[@]}" buildx inspect 2>/dev/null | sed -nE 's/^Driver:[[:space:]]+//p' | head -1 || true)
  [ "$drv" = docker ] || die "the active buildx builder uses driver '$drv'; the chain needs the 'docker' driver to see its own FROM ${IMAGE_REPO}:* tags (docker buildx use default)"
fi

R=$IMAGE_REPO
build "$R:qsa-cid"                     $D/Dockerfile.tabbyapi-qsa-cid $D
build "$R:qsa-cid-pr337"               $D/Dockerfile.tabbyapi-pr337   $D --build-arg BASE="$R:qsa-cid"
B=$R:qsa-cid-pr337
build "$B-bszn16"                      $D/Dockerfile.tabbyapi-bszn    $D --build-arg BASE="$B" --build-arg PATCH=bszn16.patch ${JOBS[@]+"${JOBS[@]}"}
build "$B-bszn16-coopwide"             $D/Dockerfile.tabbyapi-bszn    $D --build-arg BASE="$B-bszn16" --build-arg PATCH=coopwide.patch ${JOBS[@]+"${JOBS[@]}"}
C=$B-bszn16-coopwide
build "$C-hcmix1"                      $D/Dockerfile.tabbyapi-bszn    $D --build-arg BASE="$C" --build-arg PATCH=hc-mix-v2.patch ${JOBS[@]+"${JOBS[@]}"}
build "$C-hcmix2"                      $D/Dockerfile.tabbyapi-bszn    $D --build-arg BASE="$C-hcmix1" --build-arg PATCH=hc-mix-v2-r2.patch ${JOBS[@]+"${JOBS[@]}"}
build "$C-hcmix2-hostgap"              $D/Dockerfile.tabbyapi-pyfile  $D --build-arg BASE="$C-hcmix2" \
  --build-arg SRC=hostgap-gated_delta_net.py --build-arg DST=exllamav3/modules/gated_delta_net.py
P=$C-hcmix2-hostgap-ppipe
build "$P"                             $D/Dockerfile.tabbyapi-pypatch $D --build-arg BASE="$C-hcmix2-hostgap" --build-arg PATCH=prefill-pipeline.patch
build "$P-nosync"                      $O/prefill-nosync-overlay/Dockerfile        $O/prefill-nosync-overlay        --build-arg BASE="$P"
build "$P-nosync-mtpfix2"              $O/prefill-pipeline-mtp-overlay/Dockerfile  $O/prefill-pipeline-mtp-overlay  --build-arg BASE="$P-nosync"
build "$P-nosync-mtpfix2-moecoopv2"    $O/moe-coop-v2-overlay/Dockerfile           $O                               --build-arg BASE="$P-nosync-mtpfix2"
build "$R:decode-kernels-r2"           $O/decode-kernels-r2/Dockerfile.box   $O/decode-kernels-r2   --build-arg BASE="$P-nosync-mtpfix2-moecoopv2"
build "$R:decode-kernels-r2-refbase"   $O/refbase/Dockerfile.box             $O/refbase             --build-arg BASE="$R:decode-kernels-r2"
build "$R:decode-kernels-r4"           $O/decode-kernels-r4/Dockerfile.box   $O/decode-kernels-r4   --build-arg BASE="$R:decode-kernels-r2-refbase"
build "$R:stack-r4-e3r2"               $O/stack-r4-e3r2/Dockerfile.box       $O/stack-r4-e3r2       --build-arg BASE="$R:decode-kernels-r4"
build "$R:mtp-pruned-r1"               $O/mtp-pruned-r1/Dockerfile.box       $O/mtp-pruned-r1       --build-arg BASE="$R:stack-r4-e3r2"
build "$R:mtp-pruned-r1-tc1"           $O/tool-choice-r1/Dockerfile.box      $O/tool-choice-r1      --build-arg BASE="$R:mtp-pruned-r1"
build "$R:mtp-pruned-r1-tc1-plefix"    $O/ple-ckpt-clone-r1/Dockerfile.box   $O/ple-ckpt-clone-r1   --build-arg BASE="$R:mtp-pruned-r1-tc1"
build "$R:nvme-tier-r4"                $O/nvme-tier-r4/Dockerfile.box        $O/nvme-tier-r4        --build-arg BASE="$R:mtp-pruned-r1-tc1-plefix"
N4=$R:nvme-tier-r4
build "$N4-e3det"                      $O/e3-det-r1/Dockerfile.box                   $O/e3-det-r1          --build-arg BASE="$N4"
build "$N4-e3det-r6"                   $O/decode-kernels-r6/overlay/Dockerfile.box   $O/decode-kernels-r6  --build-arg BASE="$N4-e3det"
build "$N4-e3det-r6-rawk"              $O/qsa-rawk-ring-r1/Dockerfile.box            $O/qsa-rawk-ring-r1   --build-arg BASE="$N4-e3det-r6"
build "$N4-e3det-r6-rawk-gdnbf16"      $O/gdn-state-bf16-r1/Dockerfile.box           $O/gdn-state-bf16-r1  --build-arg BASE="$N4-e3det-r6-rawk" \
  --build-arg MANIFEST=manifest-on-r6.json
build "$R:ngram-prefetch-r1-gdnbf16"   $O/ngram-prefetch-r1/Dockerfile.box   $O/ngram-prefetch-r1   --build-arg BASE="$N4-e3det-r6-rawk-gdnbf16"
build "$R:mtpwin-r2"                   $O/mtp-kv-window-r2/Dockerfile.box    $O/mtp-kv-window-r2    --build-arg BASE="$R:ngram-prefetch-r1-gdnbf16"
build "$R:mtpwin-r2-metrics1"          $O/metrics-r1/Dockerfile.box          $O/metrics-r1          --build-arg BASE="$R:mtpwin-r2"
build "$R:bverify-r1"                  $D/Dockerfile.tabbyapi-bverify   $D --build-arg BASE="$R:mtpwin-r2-metrics1"
build "$R:mtpnorm-r1"                  $D/Dockerfile.tabbyapi-mtpnorm   $D --build-arg BASE="$R:bverify-r1"
build "$R:mixstate-r1"                 $D/Dockerfile.tabbyapi-mixstate  $D --build-arg BASE="$R:mtpnorm-r1" ${JOBS[@]+"${JOBS[@]}"}
build "$R:stack-r1"                    $D/Dockerfile.tabbyapi-prefbatch $D --build-arg BASE="$R:mixstate-r1"
build "$R:slotfix-r1"                  $O/slotfix-r1/Dockerfile.box   $O/slotfix-r1   --build-arg BASE="$R:stack-r1"
build "$R:hcfast-r1"                   $O/hcfast-r1/Dockerfile.box    $O/hcfast-r1    --build-arg BASE="$R:slotfix-r1" ${JOBS[@]+"${JOBS[@]}"}
build "$R:stack-r2"                    $O/moefast-r1/Dockerfile.box   $O/moefast-r1   --build-arg BASE="$R:hcfast-r1" ${JOBS[@]+"${JOBS[@]}"}
build "$R:stack-r3"                    $O/stack-r3/Dockerfile.box     $O/stack-r3     --build-arg BASE="$R:stack-r2" \
  --build-arg INCLUDE="mf3 dg2" --build-arg DGV2_NVCC_DEFS= --build-arg SERIES_SHA="$(sha $O/stack-r3/series/SERIES)" ${JOBS[@]+"${JOBS[@]}"}
# rows32-r4 records its base's image ID and the two stack-r3 labels it must rebuild with (overlays/rows32-r4/README.md)
S3=$R:stack-r3
if [ "$DRY_RUN" = 1 ]; then
  S3ID="<docker image inspect $S3 --format {{.Id}}>"
  S3INC="<label local.stack.include of $S3>"; S3DEF="<label local.stack.dgv2_defs of $S3>"
else
  S3ID=$("${DOCKER_CMD[@]}" image inspect "$S3" --format '{{.Id}}')
  S3INC=$("${DOCKER_CMD[@]}" image inspect "$S3" --format '{{index .Config.Labels "local.stack.include"}}')
  S3DEF=$("${DOCKER_CMD[@]}" image inspect "$S3" --format '{{index .Config.Labels "local.stack.dgv2_defs"}}')
fi
FINAL=$R:stack-r3-rows32
build "$FINAL"                         $O/rows32-r4/Dockerfile.box    $O/rows32-r4    --build-arg BASE="$S3" --build-arg BASE_ID="$S3ID" \
  --build-arg STACK_INCLUDE="$S3INC" --build-arg DGV2_NVCC_DEFS="$S3DEF" --build-arg PATCH_SHA="$(sha $O/rows32-r4/rows32-r4.patch)" ${JOBS[@]+"${JOBS[@]}"}

[ "${FINAL#*:}" = "${EXPECT#*:}" ] || die "built $FINAL but $LAUNCHER serves $EXPECT"
if [ "$DRY_RUN" = 1 ]; then log "=== DRY_RUN: $N layers; final tag $FINAL matches DAILY_IMG=$EXPECT in $LAUNCHER ==="; exit 0; fi
"${DOCKER_CMD[@]}" image inspect "$FINAL" >/dev/null 2>&1 || die "$FINAL not present after the build"
"${DOCKER_CMD[@]}" image ls --format '{{.Repository}}:{{.Tag}} {{.ID}} {{.Size}}' | grep -F "$R:" | tee "$LOG_DIR/images.txt" | sed 's/^/[image] /' || true
log "=== DONE: $N layers, $FINAL = DAILY_IMG of $LAUNCHER (tag ${EXPECT#*:}) ==="
