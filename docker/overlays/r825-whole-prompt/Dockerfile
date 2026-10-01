# Operator builds separately from flan:
# docker build --network=none -t tabbyapi:r825-whole-prompt docker/overlays/r825-whole-prompt
ARG BASE=tabbyapi:r823c-cachetail-inforward
FROM ${BASE}
LABEL local.r825.base="tabbyapi:r823c-cachetail-inforward"
ENV EXL3_PREFILL_WHOLE_PROMPT=0
COPY fix.patch SHA256SUMS.exl3.base SHA256SUMS.exl3.src landing_r825.py /opt/r825/
RUN set -e; site=$(python3 -c "import exllamav3,os; print(os.path.dirname(exllamav3.__file__))"); \
    (cd "$site" && sha256sum -c /opt/r825/SHA256SUMS.exl3.base); \
    patch --fuzz=0 -p1 --forward -d "$site" < /opt/r825/fix.patch; \
    (cd "$site" && sha256sum -c /opt/r825/SHA256SUMS.exl3.src); \
    test -z "$(find "$site" -name '*.rej' -o -name '*.orig')"; \
    find "$site" -name __pycache__ -type d -prune -exec rm -rf {} +
RUN python3 /opt/r825/landing_r825.py > /opt/r825/landing.txt 2>&1 \
    || { cat /opt/r825/landing.txt; exit 1; }
