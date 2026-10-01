# r825c-hostprepare

Runs the CPU-only prepare_for_queue at enqueue, before TabbyAPI attaches its trace hook. GPU admission remains deferred until window ownership clears. The trace accepts unknown page digests and publishes them after preparation; failed enqueue removes the async mapping.

The 2026-10-01 measurements, raw records and serving gates are in [R824–R825p](../../../bench/results/r825-whole-prompt-window.md). `fix.patch` and the input/output SHA-256 manifests are copied unchanged from the measured packet. `Dockerfile` accepts `BASE` for the public build chain and verifies both manifests with fuzz zero before running the import landing. ExLlamaV3-derived code is MIT; copied TabbyAPI frontend fixtures in R825c are AGPL-3.0, as recorded in [THIRD_PARTY.md](../../../THIRD_PARTY.md).

`DESIGN.md` preserves the pre-run packet design with private paths scrubbed; statements about pending experiments describe that packet stage. The installed R825p launcher carries its image ID literally and does not read the experiment pin file. The operator drivers are archived under `scripts/`, instruments under `bench/`; they require the deployment helper libraries and fixtures named in their headers under `/srv/qwen5090`. Historical MD5 pins describe the original deployment bytes; adapted public scripts require repinning before an operator run.

The public frontend self-test reads `fixtures/late-arrival-failure.json`, an extraction of the observed exception markers, prompt size and allocation ordering used by the original test. The container-log excerpt is omitted. Copied frontend source hashes remain unchanged.
