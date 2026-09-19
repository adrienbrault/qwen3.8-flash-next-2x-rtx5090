ARG BASE=tabbyapi:qsa-cid-pr337-bszn16-coopwide-hcmix2-hostgap-ppipe-nosync-mtpfix2-moecoopv2
FROM ${BASE}

COPY out/decode-kernels-r2/overlay /opt/decode-kernels-r2
COPY out/decode-kernels-r2/tests /opt/decode-kernels-r2/tests

RUN python3 /opt/decode-kernels-r2/install.py
ENV TORCH_CUDA_ARCH_LIST=12.0
RUN python3 /opt/decode-kernels-r2/build_extension.py
