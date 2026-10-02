# R828c: adaptive prompt lookup beside MTP

Served since 2026-10-02 19:10 UTC after R828c confirmation, as `tabbyapi:r828-prompt-lookup-r3`. Completed request/counter records, conditions and decisions are in [R827–R828c](../../../bench/results/r827-r828-prompt-lookup.md). [PORT.md](PORT.md) retains the pre-run design and may contain superseded pending-run statements.

The installer checks exact base, installed and unchanged source SHA-256 entries. The five overlay files retain the source packet's bytes; Python/client fixtures and launcher paths are adapted for public distribution. ExLlamaV3 derivatives are MIT and TabbyAPI derivatives AGPL-3.0; [THIRD_PARTY.md](../../../THIRD_PARTY.md) credits peonist-ai's `halogen-flash-server` `36988cb` for the underlying idea, its Codex r1 implementation and this repository's r3 adaptive trigger.

```sh
python3 -B docker/overlays/prompt-lookup-r3/selftest.py
python3 -B bench/r827_r828_summary.py
```

The Dockerfile supports `--build-arg BASE=tabbyapi:r825c-hostprepare`, checks source hashes and runs CPU selftests without CUDA. It builds directly on R825c; r3 does not build on r2. Image-ID labels and driver MD5 guards retain operator-recorded provenance and require repinning rebuilt/adapted artifacts before reuse. Copy this packet under `/srv/qwen5090/prompt-lookup-r3` with the corresponding scripts and existing queue, serving and gateway-drain helpers. The public launcher preserves public deployment paths and the private launcher's runtime delta; the source MD5 is not the public launcher's byte hash.
