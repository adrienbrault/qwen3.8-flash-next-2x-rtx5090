# From flan: docker build --network=none -t tabbyapi:r823-cachetrace docker/overlays/r823-cachetrace
ARG BASE=tabbyapi:merge-tok-r1
FROM ${BASE}
ENV EXL3_CACHE_TRACE=1
LABEL local.r823.base_id="sha256:ac16920f72cf41864ed7f4151bd18bc4ce593015dc0bc263284a8fde8c111454"
COPY fix.patch exl3.patch app.patch SHA256SUMS.exl3.base SHA256SUMS.exl3.src SHA256SUMS.app.base SHA256SUMS.app.src landing_r823.py /opt/r823-cachetrace/
RUN set -e; site=$(python3 -c "import importlib.util,os;print(os.path.dirname(importlib.util.find_spec('exllamav3').origin))"); \
    (cd "$site" && sha256sum -c /opt/r823-cachetrace/SHA256SUMS.exl3.base); \
    (cd /app && sha256sum -c /opt/r823-cachetrace/SHA256SUMS.app.base); \
    patch --fuzz=0 -p1 --forward -d "$site" < /opt/r823-cachetrace/fix.patch; \
    patch --fuzz=0 -p1 --forward -d "$site" < /opt/r823-cachetrace/exl3.patch; \
    patch --fuzz=0 -p1 --forward -d /app < /opt/r823-cachetrace/app.patch; \
    (cd "$site" && sha256sum -c /opt/r823-cachetrace/SHA256SUMS.exl3.src); \
    (cd /app && sha256sum -c /opt/r823-cachetrace/SHA256SUMS.app.src); \
    test -z "$(find "$site" /app -name '*.rej' -o -name '*.orig')"; \
    find "$site" /app -name __pycache__ -type d -prune -exec rm -rf {} +
RUN python3 /opt/r823-cachetrace/landing_r823.py > /opt/r823-cachetrace/landing.txt 2>&1 \
    || { cat /opt/r823-cachetrace/landing.txt; exit 1; }
RUN cat /opt/r823-cachetrace/landing.txt
