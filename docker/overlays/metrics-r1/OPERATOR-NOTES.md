# metrics-r1 — hashes to fill into manifest.json

For each manifest entry:

- `overlay_sha256` = sha256 of `out/metrics-r1/overlay/<path>`
- `baseline_sha256` = sha256 of the served file the entry modifies, taken from
  the reference tree `src/<same path>` — or `null` for new files.

| manifest path | overlay file to hash | baseline file to hash |
|---|---|---|
| `common/gen_logging.py` | `out/metrics-r1/overlay/common/gen_logging.py` | `src/common/gen_logging.py` |
| `common/metrics.py` | `out/metrics-r1/overlay/common/metrics.py` | none — new file, keep `null` |
| `endpoints/core/router.py` | `out/metrics-r1/overlay/endpoints/core/router.py` | `src/endpoints/core/router.py` |
| `tests/test_metrics_exposition.py` | `out/metrics-r1/overlay/tests/test_metrics_exposition.py` | none — new file, keep `null` |
| `tests/test_metrics_record.py` | `out/metrics-r1/overlay/tests/test_metrics_record.py` | none — new file, keep `null` |

Compute with:

```
shasum -a 256 <file>
```

Check before hashing:

- `diff src/common/gen_logging.py out/metrics-r1/overlay/common/gen_logging.py`
  should show exactly two additions: the `common.metrics` import and the
  `record_completion` call at the end of `log_metrics`.
- `diff src/endpoints/core/router.py out/metrics-r1/overlay/endpoints/core/router.py`
  should show exactly two additions: the `common.metrics` import and the
  `/metrics` route.

Optional parity with the example package: `ref/example-tabbyapi-overlay/local/make_package.py --ref src`
regenerates `manifest.json` (with real hashes) and a `served-source.patch` from
the overlay tree. If used, note it writes the manifest `name` field as
`tool-choice-r1` — change it to `metrics-r1`.

Run the tests from the overlay root (or `/app` after install) with:

```
python -m unittest tests.test_metrics_exposition tests.test_metrics_record -v
```

`tests.test_metrics_record.LogMetricsHookTests` skips itself where the app
dependencies (pydantic, loguru) are absent; everything else is stdlib-only.
