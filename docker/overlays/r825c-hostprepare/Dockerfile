# Operator build only; verify the parent tag resolves to the pinned ID first.
ARG BASE=tabbyapi:r825b-resumable
FROM ${BASE}
LABEL local.r825c.base="sha256:01c967e088a35f3ffb02f98a458fefa5fb3da1604a18591c05f71399c6d548b9"
ENV EXL3_PREFILL_WHOLE_PROMPT=0 EXL3_PREFILL_RESUMABLE=1
COPY fix.patch SHA256SUMS.exl3.base SHA256SUMS.exl3.src landing_r825c.py test_frontend.py BASE-PROVENANCE.json /opt/r825c/
COPY base /opt/r825c/base
COPY src /opt/r825c/src
COPY fixtures /opt/r825c/fixtures
RUN set -e; site=$(python3 -c "import exllamav3,os; print(os.path.dirname(exllamav3.__file__))"); \
    (cd "$site" && sha256sum -c /opt/r825c/SHA256SUMS.exl3.base); \
    patch --fuzz=0 -p1 --forward -d "$site" < /opt/r825c/fix.patch; \
    (cd "$site" && sha256sum -c /opt/r825c/SHA256SUMS.exl3.src); \
    test -z "$(find "$site" -name '*.rej' -o -name '*.orig')"; \
    find "$site" -name __pycache__ -type d -prune -exec rm -rf {} +
RUN CUDA_VISIBLE_DEVICES= python3 -B /opt/r825c/test_frontend.py > /opt/r825c/frontend-selftest.txt 2>&1 \
    || { cat /opt/r825c/frontend-selftest.txt; exit 1; }
RUN CUDA_VISIBLE_DEVICES= python3 -B /opt/r825c/landing_r825c.py > /opt/r825c/landing.txt 2>&1 \
    || { cat /opt/r825c/landing.txt; exit 1; }
