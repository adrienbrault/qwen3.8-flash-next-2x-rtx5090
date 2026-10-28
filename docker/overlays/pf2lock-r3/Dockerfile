# pf2lock r3: tabbyapi:tune268-r2 + one Python patch, pf2lock.patch, to exllamav3/modules/ngram_embedding.py. A
# threading.Lock makes the EXL3_NGRAM_PREFETCH2 staging-set selection and its held mark one step and guards the
# _pending list; every failure path releases the set it holds (transactional grow, reservation rollback, finally-based
# release on failed staging, retirement, forward and submit; drain always shuts the executor down); a per-entry
# commit/abort gate keeps a submitted task off every set until its publication is decided. No native rebuild: the
# extension .so is the base's byte for byte (asserted), no ENV change, no other exllamav3 file touched.
# Build gates inside the RUN (no GPU): four CPU stress scenarios (5,000 iterations each) and 24 deterministic
# ownership tests (pf2lock_tests.py) on the installed, patched module. Built by build.sh, CPU only, no network.
ARG BASE=tabbyapi:tune268-r2
FROM ${BASE}
ARG BASE
ARG PATCH_SHA=unset
ARG MODULE_BASE_SHA=b9b227f87b2ec1fb3b635c90268bcb789ebb3ae9fb9ce8dccdcbb556416838b8
ARG MODULE_NEW_SHA=unset
COPY pf2lock.patch cpu_stress.py pf2lock_harness.py pf2lock_tests.py /opt/pf2lock/
RUN set -e; site=$(python3 -c 'import sysconfig; print(sysconfig.get_paths()["purelib"])'); \
    mod="$site/exllamav3/modules/ngram_embedding.py"; so=$(ls "$site"/exllamav3_ext*.so); test -f "$so"; \
    test "$(sha256sum /opt/pf2lock/pf2lock.patch | cut -c1-64)" = "${PATCH_SHA}"; \
    test "$(sha256sum "$mod" | cut -c1-64)" = "${MODULE_BASE_SHA}"; \
    sha256sum "$so" > /opt/pf2lock/so-base.sha256; \
    patch --batch --forward --fuzz=0 -p1 -d "$site" < /opt/pf2lock/pf2lock.patch; \
    test -z "$(find "$site/exllamav3" -name '*.rej' -o -name '*.orig')"; \
    test "$(sha256sum "$mod" | cut -c1-64)" = "${MODULE_NEW_SHA}"; \
    sha256sum -c /opt/pf2lock/so-base.sha256; \
    rm -f "$site"/exllamav3/modules/__pycache__/ngram_embedding.*.pyc; \
    export CUDA_VISIBLE_DEVICES= PYTHONDONTWRITEBYTECODE=1; \
    for sc in share steady long; do \
      EXL3_NGRAM_PREFETCH2=1 python3 /opt/pf2lock/cpu_stress.py --iterations 5000 --scenario $sc --json /opt/pf2lock/stress-$sc.json > /opt/pf2lock/stress-$sc.log 2>&1 || { cat /opt/pf2lock/stress-$sc.log; exit 1; }; \
    done; \
    EXL3_NGRAM_PREFETCH2=0 python3 /opt/pf2lock/cpu_stress.py --iterations 5000 --scenario offpath --json /opt/pf2lock/stress-offpath.json > /opt/pf2lock/stress-offpath.log 2>&1 || { cat /opt/pf2lock/stress-offpath.log; exit 1; }; \
    EXL3_NGRAM_PREFETCH2=1 python3 /opt/pf2lock/pf2lock_tests.py -v > /opt/pf2lock/tests.log 2>&1 || { cat /opt/pf2lock/tests.log; exit 1; }; \
    python3 -c 'import torch, exllamav3_ext, exllamav3; from exllamav3.modules import ngram_embedding as ne; m = ne.NGramEmbedding(None, "k", 2, 1, ne.ROW_DIM, 0); assert hasattr(m, "_lock") and hasattr(m, "_abandon") and hasattr(m, "_publish") and hasattr(ne, "_pinned_empty"); print("pf2lock import PASS")'
LABEL local.pf2lock.round="r3" local.pf2lock.base="${BASE}" local.pf2lock.patch_sha256="${PATCH_SHA}" \
      local.pf2lock.module_sha256="${MODULE_NEW_SHA}"
