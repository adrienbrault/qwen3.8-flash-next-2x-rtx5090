# tune268 r2 (codex 2a + 2b): default-off EXL3_MOE_PREFILL_E3_HALF (K2.5 experts in the E3 prefill MoE path, NOT bitwise
# vs the K=0 fused fallback), EXL3_MOE_PREFILL_HALF_HIST, EXL3_MOE_PREFILL_HALF_STATIC, EXL3_MOE_COOP_HALF_A_UNCAP (bitwise
# levers), on the SERVED image tabbyapi:tune268-r1 (sha256:48f4ad3a...). Same rebuild recipe as r1; every r1 function's
# SASS must be unchanged and exactly 14 kernels added (sass_check_r2.py). Built by build.sh, CPU only.
ARG BASE=tabbyapi:tune268-r1
FROM ${BASE}
ARG BASE
ARG MAX_JOBS=6
ARG PATCH_SHA=unset
COPY tune268-r2.patch sass_hashes.py sass_check_r2.py /opt/tune268r2/
RUN set -e; site=$(python3 -c 'import sysconfig; print(sysconfig.get_paths()["purelib"])'); \
    so=$(ls $site/exllamav3_ext*.so); \
    test "$(sha256sum /opt/tune268r2/tune268-r2.patch | cut -c1-64)" = "${PATCH_SHA}"; \
    cuobjdump -sass "$so" | python3 /opt/tune268r2/sass_hashes.py > /opt/tune268r2/sass-base.txt; \
    patch --batch --forward --fuzz=0 -p1 -d "$site" < /opt/tune268r2/tune268-r2.patch; \
    test -z "$(find "$site/exllamav3" -name '*.rej' -o -name '*.orig')"; \
    b=$(mktemp -d); cd "$b"; \
    DGV2_NVCC_DEFS= MAX_JOBS=${MAX_JOBS} python3 /opt/rebase-dev-r3/rebuild-native.py build_ext --force --build-lib "$site" > /opt/tune268r2/build-native.log 2>&1 || { tail -80 /opt/tune268r2/build-native.log; exit 1; }; \
    cd /; rm -rf "$b"; \
    grep -E "registers|spill" /opt/tune268r2/build-native.log > /opt/tune268r2/ptxas.txt || true; \
    cuobjdump -sass "$so" | python3 /opt/tune268r2/sass_hashes.py > /opt/tune268r2/sass-new.txt; \
    python3 /opt/tune268r2/sass_check_r2.py /opt/tune268r2/sass-base.txt /opt/tune268r2/sass-new.txt /opt/tune268r2/sass-report.txt; \
    find "$site/exllamav3" -name __pycache__ -prune -exec rm -rf {} +; \
    CUDA_VISIBLE_DEVICES= python3 -c 'import torch, exllamav3_ext, exllamav3; print("tune268-r2 import PASS")'
ENV EXL3_MOE_PREFILL_E3_HALF=0 EXL3_MOE_PREFILL_HALF_HIST=0 EXL3_MOE_PREFILL_HALF_STATIC=0 EXL3_MOE_COOP_HALF_A_UNCAP=0
LABEL local.tune268.round="r2" local.tune268.base="${BASE}" local.tune268.patch_sha256="${PATCH_SHA}"
