# From flan: docker build --network=none -t tabbyapi:r823c-cachetail-inforward docker/overlays/r823c-cachetail-inforward
ARG BASE=tabbyapi:merge-tok-r1
FROM ${BASE}
ENV EXL3_CACHE_TRACE=1
LABEL local.r823c.base="tabbyapi:merge-tok-r1"
COPY trace/ /opt/r823-cachetrace/
COPY tail/ /opt/r823b-cachetail/
COPY fix.patch SHA256SUMS.exl3.base SHA256SUMS.exl3.src landing_r823c.py /opt/r823c-cachetail-inforward/
RUN set -e; site=$(python3 -c "import importlib.util,os;print(os.path.dirname(importlib.util.find_spec('exllamav3').origin))"); \
    (cd "$site" && sha256sum -c /opt/r823-cachetrace/SHA256SUMS.exl3.base); \
    (cd /app && sha256sum -c /opt/r823-cachetrace/SHA256SUMS.app.base); \
    patch --fuzz=0 -p1 --forward -d "$site" < /opt/r823-cachetrace/fix.patch; \
    patch --fuzz=0 -p1 --forward -d "$site" < /opt/r823-cachetrace/exl3.patch; \
    patch --fuzz=0 -p1 --forward -d /app < /opt/r823-cachetrace/app.patch; \
    (cd "$site" && sha256sum -c /opt/r823-cachetrace/SHA256SUMS.exl3.src); \
    (cd /app && sha256sum -c /opt/r823-cachetrace/SHA256SUMS.app.src); \
    (cd "$site" && sha256sum -c /opt/r823b-cachetail/SHA256SUMS.exl3.base); \
    patch --fuzz=0 -p1 --forward -d "$site" < /opt/r823b-cachetail/fix.patch; \
    (cd "$site" && sha256sum -c /opt/r823b-cachetail/SHA256SUMS.exl3.src); \
    (cd "$site" && sha256sum -c /opt/r823c-cachetail-inforward/SHA256SUMS.exl3.base); \
    patch --fuzz=0 -p1 --forward -d "$site" < /opt/r823c-cachetail-inforward/fix.patch; \
    (cd "$site" && sha256sum -c /opt/r823c-cachetail-inforward/SHA256SUMS.exl3.src); \
    test -z "$(find "$site" /app -name '*.rej' -o -name '*.orig')"; \
    find "$site" /app -name __pycache__ -type d -prune -exec rm -rf {} +
RUN python3 /opt/r823c-cachetail-inforward/landing_r823c.py > /opt/r823c-cachetail-inforward/landing.txt 2>&1 \
    || { cat /opt/r823c-cachetail-inforward/landing.txt; exit 1; }
