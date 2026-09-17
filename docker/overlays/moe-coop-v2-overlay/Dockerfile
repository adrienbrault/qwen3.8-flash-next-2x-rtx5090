# Build context is out/, never the clean v1.5.0 source or an upstream daily tag.
# BASE must name/digest the operator's served, patched cu128 image.
ARG BASE
FROM ${BASE}
COPY moe-coop-v2-overlay /opt/moe-coop-v2
COPY test_moe_coop_v2.py /opt/moe-coop-v2/test_moe_coop_v2.py
RUN python3 /opt/moe-coop-v2/install.py --disable-precompiled
ENV TORCH_CUDA_ARCH_LIST="12.0" \
    TORCH_EXTENSIONS_DIR="/opt/moe-coop-v2/torch-extensions" \
    EXL3_MOE_COOP_V2="0"
# A fresh JIT build includes every existing K/codebook compilation unit. Never
# permit an old precompiled .so to make the patched source a no-op.
RUN python3 -c 'from exllamav3.ext import exllamav3_ext; print(exllamav3_ext.__file__)'
# Keep the BASE entrypoint/cmd; GPU tests override it with --entrypoint python3.
