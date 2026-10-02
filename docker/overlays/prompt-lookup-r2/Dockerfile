# Operator: docker build -t tabbyapi:r827-prompt-lookup prompt-lookup-r2
# Pin/inspect parent first: sha256:aa04a1cbe94b60bbd91e3d69d15f5e1b6eb679f14885e599404de218d8a897c9
ARG BASE=tabbyapi:r825c-hostprepare
FROM ${BASE}
LABEL local.r827.base="sha256:aa04a1cbe94b60bbd91e3d69d15f5e1b6eb679f14885e599404de218d8a897c9"
ENV EXL3_PROMPT_LOOKUP=0
COPY . /opt/r827
RUN CUDA_VISIBLE_DEVICES= python3 -B /opt/r827/install.py
RUN CUDA_VISIBLE_DEVICES= python3 -B /opt/r827/install.py --check
RUN CUDA_VISIBLE_DEVICES= python3 -B /opt/r827/selftest.py
