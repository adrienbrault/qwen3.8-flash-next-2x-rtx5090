ARG BASE=tabbyapi:r828-prompt-lookup-r3
FROM ${BASE}
COPY out/exllamav3-1.6.4+cu128.torch2.9.0-cp312-cp312-linux_x86_64.whl /opt/flashnext-v164/
RUN python3 -m pip uninstall -y exllamav3 && rm -rf /opt/venv/lib/python3.12/site-packages/exllamav3 && python3 -m pip install --no-deps /opt/flashnext-v164/*.whl
COPY out/engine/exllamav3 /opt/venv/lib/python3.12/site-packages/exllamav3
COPY out/tabby /app
COPY manifest.json /opt/flashnext-v164/manifest.json
COPY cpu-check.py /opt/flashnext-v164/cpu-check.py
ENV EXL3_CACHE_TRACE=0
RUN python3 -m compileall -q /app/backends /app/common /app/endpoints /opt/venv/lib/python3.12/site-packages/exllamav3 && \
    python3 -c 'import exllamav3; from exllamav3 import Generator; from inspect import signature; assert "num_draft_tokens_by_batch" in signature(Generator.__init__).parameters; import endpoints.OAI.utils.chat_completion; import backends.exllamav3.model; print("actual engine and Tabby imports PASS")'
RUN python3 /opt/flashnext-v164/cpu-check.py
LABEL local.flashnext.engine="da68d57de61781e5b74ad68698ac4caf36408d7a" local.flashnext.tabby="a270c555db91a5178d72f7dee7bdd34951ac1dad"
