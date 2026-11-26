# rebase167 candidate image: the served daily image (tabbyapi:flashnext-names-fast-r1) with its ExLlamaV3 package replaced by the
# integrated v1.6.7 merge (ctx/exllamav3-src) and its TabbyAPI tree replaced by the pinned TabbyAPI branch (ctx/tabbyapi); the
# native extension is rebuilt in-image against the image's own Torch/CUDA (torch 2.9.0+cu128, nvcc 12.8) with the
# served nvcc flags. CPU only: built with --network=none --pull=false, no GPU. Never retags the served image.
#
# Build context (assembled by build-image.sh):
#   ctx/exllamav3-src/   integrated exllamav3 repo checkout (exllamav3/ package at its root)
#   ctx/tabbyapi/        integrated TabbyAPI tree
#   native/              this kit (rebuild-native-167.py, sass_inventory.py, smoke.py, tests/, run-cpu-tests.sh)
#
# Outputs inside the image, /opt/rebase167/:
#   inv/sass-served.txt, inv/sass-new.txt, inv/sass-report.txt   per-function SASS hashes + comparison
#   inv/elf-served.txt, inv/elf-new.txt                           cuobjdump --list-elf (arch inventory)
#   inv/pybind-served.txt, inv/pybind-new.txt, inv/pybind-check.txt  exported names; served names must survive
#   inv/so-served.sha256, inv/so-new.sha256, inv/sources.txt, build-native.log, smoke.txt, app-only-in-served.txt
ARG BASE=sha256:679274ac51b5ad603347b2364816a118780fc81cda020e20e5291c8a22a7b125
FROM ${BASE}
ARG BASE
ARG MAX_JOBS=4
ARG ARCH_LIST=
ARG EXL3_REV=unset
ARG TABBY_REV=unset
# Screen-harness identity: the fix round this image belongs to.
ARG ROUND=r2
SHELL ["/bin/bash", "-o", "pipefail", "-c"]

COPY native/rebuild-native-167.py native/sass_inventory.py native/smoke.py native/run-cpu-tests.sh /opt/rebase167/tools/
COPY native/tests /opt/rebase167/tests
COPY cpu /opt/rebase167/cpu
COPY ctx/exllamav3-src /opt/rebase167/exllamav3-src
COPY ctx/tabbyapi /opt/rebase167/tabbyapi-src

# 1. Inventory of the served extension, before anything is replaced
RUN set -eu; \
    site=$(python3 -c 'import sysconfig; print(sysconfig.get_paths()["purelib"])'); \
    set -- $site/exllamav3_ext*.so; test $# -eq 1 -a -f "$1"; so=$1; \
    I=/opt/rebase167/inv; mkdir -p $I; \
    sha256sum "$so" > $I/so-served.sha256; \
    cuobjdump --list-elf "$so" > $I/elf-served.txt; \
    cuobjdump --gpu-architecture sm_120 --dump-sass "$so" | python3 /opt/rebase167/tools/sass_inventory.py hashes > $I/sass-served.txt; \
    python3 /opt/rebase167/tools/sass_inventory.py require-tune268 $I/sass-served.txt > $I/tune268-served.txt; \
    cd /tmp && CUDA_VISIBLE_DEVICES= python3 -c 'import torch, exllamav3_ext as e; print("\n".join(n for n in sorted(dir(e)) if not n.startswith("__")))' > $I/pybind-served.txt; \
    wc -l $I/sass-served.txt $I/pybind-served.txt

# 2. Replace the ExLlamaV3 package (python + extension sources) with the integrated tree
RUN set -eu; \
    site=$(python3 -c 'import sysconfig; print(sysconfig.get_paths()["purelib"])'); \
    src=/opt/rebase167/exllamav3-src; \
    test -f $src/exllamav3/version.py -a -f $src/exllamav3/util/cuda_flags.py -a -d $src/exllamav3/exllamav3_ext; \
    test -z "$(grep -rlE '^(<<<<<<<|>>>>>>>) ' $src/exllamav3 || true)"; \
    rm -rf $site/exllamav3; cp -a $src/exllamav3 $site/exllamav3; \
    find $site/exllamav3 -name __pycache__ -prune -exec rm -rf {} +; \
    v=$(python3 -c "import runpy; print(runpy.run_path('$site/exllamav3/version.py')['__version__'])"); \
    echo "exllamav3 package version $v"; \
    for m in $site/exllamav3-*.dist-info/METADATA; do \
      if [ -f "$m" ]; then sed -i "s/^Version: .*/Version: $v/" "$m"; grep '^Version:' "$m"; fi; \
    done; \
    test "$v" = "1.6.7"

# 3. Native rebuild with the served flags (asserted inside rebuild-native-167.py)
RUN set -eu; \
    site=$(python3 -c 'import sysconfig; print(sysconfig.get_paths()["purelib"])'); \
    export TORCH_CUDA_ARCH_LIST="${ARCH_LIST:-${TORCH_CUDA_ARCH_LIST:-12.0}}"; \
    echo "TORCH_CUDA_ARCH_LIST=$TORCH_CUDA_ARCH_LIST" | tee /opt/rebase167/arch.txt; \
    rm -f $site/exllamav3_ext*.so; \
    b=$(mktemp -d); cd "$b"; \
    REBASE167_SITE=$site REBASE167_SOURCES_OUT=/opt/rebase167/inv/sources.txt MAX_JOBS=${MAX_JOBS} \
      python3 /opt/rebase167/tools/rebuild-native-167.py build_ext --force --build-lib "$site" \
      > /opt/rebase167/build-native.log 2>&1 || { tail -120 /opt/rebase167/build-native.log; exit 1; }; \
    cd /; rm -rf "$b"; \
    set -- $site/exllamav3_ext*.so; test $# -eq 1 -a -f "$1"; \
    grep -c 'warning' /opt/rebase167/build-native.log || true

# 4. Inventory of the new extension; every served pybind name must still be exported
RUN set -eu; \
    site=$(python3 -c 'import sysconfig; print(sysconfig.get_paths()["purelib"])'); \
    set -- $site/exllamav3_ext*.so; so=$1; I=/opt/rebase167/inv; \
    sha256sum "$so" > $I/so-new.sha256; \
    cuobjdump --list-elf "$so" > $I/elf-new.txt; \
    grep -q 'sm_120' $I/elf-new.txt; \
    cuobjdump --gpu-architecture sm_120 --dump-sass "$so" | python3 /opt/rebase167/tools/sass_inventory.py hashes > $I/sass-new.txt; \
    python3 /opt/rebase167/tools/sass_inventory.py require-tune268 $I/sass-new.txt > $I/tune268-new.txt; \
    python3 /opt/rebase167/tools/sass_inventory.py compare $I/sass-served.txt $I/sass-new.txt $I/sass-report.txt; \
    cd /tmp && CUDA_VISIBLE_DEVICES= python3 -c 'import torch, exllamav3_ext as e; print("\n".join(n for n in sorted(dir(e)) if not n.startswith("__")))' > $I/pybind-new.txt; \
    python3 /opt/rebase167/tools/sass_inventory.py pybind $I/pybind-served.txt $I/pybind-new.txt /opt/rebase167/tests/pybind_dropped_ok.txt > $I/pybind-check.txt || { cat $I/pybind-check.txt; exit 1; }; \
    cat $I/pybind-check.txt

# 5. Replace the TabbyAPI tree in /app. Runtime files the served /app holds and the new tree lacks are listed; only
#    api_tokens.yml / config.yml are carried over (the launcher bind-mounts config.yml and sampler_overrides/*.yml)
RUN set -eu; \
    test -f /opt/rebase167/tabbyapi-src/main.py; \
    test -z "$(grep -rlE '^(<<<<<<<|>>>>>>>) ' --include='*.py' /opt/rebase167/tabbyapi-src || true)"; \
    mv /app /opt/rebase167/app-served; cp -a /opt/rebase167/tabbyapi-src /app; rm -rf /app/.git; \
    (cd /opt/rebase167/app-served && find . -type f ! -path '*/__pycache__/*' | sort) > /tmp/old.txt; \
    (cd /app && find . -type f ! -path '*/__pycache__/*' | sort) > /tmp/new.txt; \
    comm -23 /tmp/old.txt /tmp/new.txt > /opt/rebase167/app-only-in-served.txt; \
    for f in api_tokens.yml config.yml; do \
      if [ -f /opt/rebase167/app-served/$f ] && [ ! -e /app/$f ]; then cp -a /opt/rebase167/app-served/$f /app/$f; echo "carried /app/$f"; fi; \
    done; \
    mkdir -p /app/sampler_overrides; \
    echo "files only in the served /app: $(wc -l < /opt/rebase167/app-only-in-served.txt)"; rm -f /tmp/old.txt /tmp/new.txt

# 6. Byte-compile, import smoke (no GPU), CPU tests: all fatal. REBASE167_NO_SKIP=1: torch and the integrated package
#    are build inputs, so a skipped test (an import that failed) fails the build like a failed one
RUN set -eu; \
    site=$(python3 -c 'import sysconfig; print(sysconfig.get_paths()["purelib"])'); \
    python3 -m compileall -q $site/exllamav3 /app/backends /app/common /app/endpoints; \
    cd /app; export PYTHONPATH=/app; \
    rc=0; CUDA_VISIBLE_DEVICES= python3 /opt/rebase167/tools/smoke.py > /opt/rebase167/smoke.txt 2>&1 || rc=$?; \
    cat /opt/rebase167/smoke.txt; test $rc -eq 0; \
    rc=0; CUDA_VISIBLE_DEVICES= REBASE167_NO_SKIP=1 bash /opt/rebase167/tools/run-cpu-tests.sh > /opt/rebase167/cpu-tests.txt 2>&1 || rc=$?; \
    cat /opt/rebase167/cpu-tests.txt; echo "cpu tests rc=$rc"; test "$rc" -eq 0; \
    CUDA_VISIBLE_DEVICES= EXL3_STUB_EXT=0 TABBY_DIR=/app EXL3_TREE="$(dirname "$site/exllamav3")" \
      /opt/rebase167/cpu/run.sh >> /opt/rebase167/cpu-tests.txt 2>&1 || { cat /opt/rebase167/cpu-tests.txt; exit 1; }; \
    python3 /opt/rebase167/exllamav3-src/tests/test_r2_defaults.py >> /opt/rebase167/cpu-tests.txt 2>&1 \
      || { cat /opt/rebase167/cpu-tests.txt; exit 1; }; \
    find $site/exllamav3 /app -name __pycache__ -prune -exec rm -rf {} + ; python3 -m compileall -q $site/exllamav3 /app/backends /app/common /app/endpoints

WORKDIR /app
RUN test -n "${ROUND}" && test "${EXL3_REV}" != unset && test "${TABBY_REV}" != unset
LABEL local.rebase167.round="${ROUND}" local.rebase167.base="${BASE}" local.rebase167.exllamav3="${EXL3_REV}" local.rebase167.tabbyapi="${TABBY_REV}"
