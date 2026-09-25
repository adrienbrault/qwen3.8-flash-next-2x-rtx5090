ARG BASE
FROM ${BASE}
COPY . /tmp/exl3-prefill-pipeline-mtp-overlay/
RUN python3 /tmp/exl3-prefill-pipeline-mtp-overlay/install.py
# R441 free-VRAM guard fix (the "2" in -mtpfix2): the pipeline's 320 MB check also counts the caching allocator's
# reusable reserve. Without it every pipelined window is rejected once the draft cache is allocated (R439/R440).
# The served -mtpfix2 image carries prefill_pipeline.py 7f6e4b34..., which later layers pin (ngram-prefetch-r1).
RUN set -eu; \
    f="$(python3 -c 'import importlib.util as u, os; print(os.path.dirname(u.find_spec("exllamav3").origin))')/generator/prefill_pipeline.py"; \
    echo "1a5af53b8b17390a3835c8b28171e2e2a98b3df850a7f6a023102be9ef1ea8ba  $f" | sha256sum -c -; \
    python3 /tmp/exl3-prefill-pipeline-mtp-overlay/memfix.py "$f"; \
    echo "7f6e4b34bda7d0437b3352f98404fb3d4da98b9c65790cf48f8c16e91c54b177  $f" | sha256sum -c -
