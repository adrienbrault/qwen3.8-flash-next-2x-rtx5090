# Only build-image.sh supplies BASE after verifying its local served digest.
ARG BASE
FROM ${BASE}
USER root
COPY overlay /opt/flashnext-names-r1/overlay
RUN python3 -B /opt/flashnext-names-r1/overlay/install.py
# Feature availability is opt-in at the operator screen, not baked into the image.
LABEL flashnext.names.parent="sha256:da386167cc762f08763a3ab41d98e50403504790b4495053d568f70e61432021"
ARG OVERLAY_SHA
LABEL flashnext.names.overlay="${OVERLAY_SHA}"
