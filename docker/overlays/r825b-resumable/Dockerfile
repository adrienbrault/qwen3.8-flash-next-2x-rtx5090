# Operator build only: docker build --network=none -t tabbyapi:r825b-resumable docker/overlays/r825b-resumable
ARG BASE=tabbyapi:r825-whole-prompt
FROM ${BASE}
LABEL local.r825b.base="sha256:d8a81e276dce0e86b511a243971696a8819012e7c1c2c92b45e055aa305fa0ff"
ENV EXL3_PREFILL_WHOLE_PROMPT=0 EXL3_PREFILL_RESUMABLE=1
COPY fix.patch SHA256SUMS.exl3.base SHA256SUMS.exl3.src landing_r825b.py /opt/r825b/
RUN set -e; site=$(python3 -c "import exllamav3,os; print(os.path.dirname(exllamav3.__file__))"); \
    (cd "$site" && sha256sum -c /opt/r825b/SHA256SUMS.exl3.base); \
    patch --fuzz=0 -p1 --forward -d "$site" < /opt/r825b/fix.patch; \
    (cd "$site" && sha256sum -c /opt/r825b/SHA256SUMS.exl3.src); \
    test -z "$(find "$site" -name '*.rej' -o -name '*.orig')"; \
    find "$site" -name __pycache__ -type d -prune -exec rm -rf {} +
RUN python3 /opt/r825b/landing_r825b.py > /opt/r825b/landing.txt 2>&1 \
    || { cat /opt/r825b/landing.txt; exit 1; }
