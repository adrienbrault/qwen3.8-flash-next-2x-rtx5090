# tune268 r1: K6 dense V2 twins (EXL3_DENSE_V2_K6) + K2.5 MoE on the V2/V3/r2 kernels (EXL3_MOE_COOP_HALF), both
# default off, on the SERVED image tabbyapi:r828-prompt-lookup-r3 (sha256:dfaed2cb...). The native extension is
# rebuilt from the image's own site-packages tree with the image's own /opt/rebase-dev-r3/rebuild-native.py (the
# 2026-09-27 rebase-dev-r3 recipe, DGV2_NVCC_DEFS=''), then every served function's SASS must be unchanged
# (sass_check.py). Built by build.sh, CPU only, --network=none. Never retags the served image.
ARG BASE=tabbyapi:r828-prompt-lookup-r3
FROM ${BASE}
ARG BASE
ARG MAX_JOBS=6
ARG PATCH_SHA=unset
COPY tune268.patch sass_hashes.py sass_check.py /opt/tune268/
RUN set -e; site=$(python3 -c 'import sysconfig; print(sysconfig.get_paths()["purelib"])'); \
    so=$(ls $site/exllamav3_ext*.so); \
    test "$(sha256sum /opt/tune268/tune268.patch | cut -c1-64)" = "${PATCH_SHA}"; \
    cuobjdump -sass "$so" | python3 /opt/tune268/sass_hashes.py > /opt/tune268/sass-base.txt; \
    sha256sum "$so" > /opt/tune268/so-base.sha256; \
    patch --batch --forward --fuzz=0 -p1 -d "$site/exllamav3" < /opt/tune268/tune268.patch; \
    test -z "$(find "$site/exllamav3" -name '*.rej' -o -name '*.orig')"; \
    b=$(mktemp -d); cd "$b"; \
    DGV2_NVCC_DEFS= MAX_JOBS=${MAX_JOBS} python3 /opt/rebase-dev-r3/rebuild-native.py build_ext --force --build-lib "$site" > /opt/tune268/build-native.log 2>&1 || { tail -60 /opt/tune268/build-native.log; exit 1; }; \
    cd /; rm -rf "$b"; \
    cuobjdump -sass "$so" | python3 /opt/tune268/sass_hashes.py > /opt/tune268/sass-new.txt; \
    python3 /opt/tune268/sass_check.py /opt/tune268/sass-base.txt /opt/tune268/sass-new.txt /opt/tune268/sass-report.txt; \
    find "$site/exllamav3" -name __pycache__ -prune -exec rm -rf {} +; \
    CUDA_VISIBLE_DEVICES= python3 -c 'import torch, exllamav3_ext as e; assert e.dense_v2_k6() is False; assert e.dense_v2_modes() == (0, 0); print("tune268 bindings PASS: dense_v2_k6 default off")'
ENV EXL3_DENSE_V2_K6=0 EXL3_MOE_COOP_HALF=0
LABEL local.tune268.round="r1" local.tune268.base="${BASE}" local.tune268.patch_sha256="${PATCH_SHA}"
