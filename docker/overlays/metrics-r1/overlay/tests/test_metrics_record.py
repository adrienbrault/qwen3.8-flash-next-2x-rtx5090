"""
Tests for the record path: synthetic log_metrics payloads in, counters and
histogram buckets out. Mirrors how common.gen_logging.log_metrics calls
record_completion once per completed generation job.

Payloads copy the shape of the finish chunk built by
backends/exllamav3/model.py handle_finish_chunk. common.metrics is stdlib-only
at import time; the model container and the app logger are faked through
sys.modules, the same lazy lookups the module uses in production.
"""

import pathlib
import sys
import unittest
from types import SimpleNamespace
from unittest.mock import patch

from common import metrics

try:
    from common import gen_logging
except Exception:  # app deps (pydantic, loguru, ...) absent outside the image
    gen_logging = None


def parse_exposition(text):
    """Parse sample lines -> {(name, sorted-labels-tuple): float}."""

    series = {}
    for line in text.splitlines():
        if not line or line.startswith("#"):
            continue
        left, _, value = line.partition(" ")
        if "{" in left:
            name, _, rest = left.partition("{")
            labels = tuple(
                sorted(
                    (kv.split("=", 1)[0], kv.split("=", 1)[1].strip('"'))
                    for kv in rest.rstrip("}").split(",")
                )
            )
        else:
            name, labels = left, ()
        series[(name, labels)] = float(value)
    return series


def sample(series, name, model="test-model", **extra):
    labels = tuple(sorted(({"model": model} | extra).items()))
    return series[(name, labels)]


class FakeContainer:
    def __init__(self, model_name="test-model"):
        self.model_dir = pathlib.Path("/models") / model_name
        self.active_job_ids = {}
        self.cache_size = 262144
        self.max_batch_size = 4
        self.max_seq_len = 131072


# A realistic sequence: one quick reply, then the slow long-generation tail.
PAYLOADS = [
    {
        "request_id": "r1",
        "prompt_tokens": 1200,
        "prompt_time": 0.05,
        "prompt_tokens_per_sec": 8000,
        "gen_tokens": 50,
        "gen_time": 0.62,
        "gen_tokens_per_sec": 80.65,
        "total_time": 0.69,
        "queue_time": 0.02,
        "cached_tokens": 800,
        "finish_reason": "stop",
        "eos_reason": "stop_token",
        "stop_str": None,
        "full_text": "ok",
    },
    {
        "request_id": "r2",
        "prompt_tokens": 30000,
        "prompt_time": 0.4,
        "prompt_tokens_per_sec": 5000,
        "gen_tokens": 2700,
        "gen_time": 150.0,
        "gen_tokens_per_sec": 18.0,
        "total_time": 150.4,
        "queue_time": 0.0,
        "cached_tokens": 28000,
        "finish_reason": "stop",
        "eos_reason": "stop_token",
        "stop_str": None,
        "full_text": "...",
    },
    {
        "request_id": "r3",
        "prompt_tokens": 31000,
        "prompt_time": 0.5,
        "prompt_tokens_per_sec": 4000,
        "gen_tokens": 3900,
        "gen_time": 221.0,
        "gen_tokens_per_sec": 17.65,
        "total_time": 222.5,
        "queue_time": 1.0,
        "cached_tokens": 29000,
        "finish_reason": "length",
        "eos_reason": "max_new_tokens",
        "stop_str": None,
        "full_text": "...",
    },
]


class RecordTestBase(unittest.TestCase):
    def setUp(self):
        metrics.reset()
        self._saved = {k: sys.modules.get(k) for k in ("common.model", "common.logger")}
        sys.modules["common.model"] = SimpleNamespace(container=FakeContainer())
        self.warnings = []
        sys.modules["common.logger"] = SimpleNamespace(
            xlogger=SimpleNamespace(
                warning=lambda *a, **k: self.warnings.append((a, k))
            )
        )

    def tearDown(self):
        for key, prev in self._saved.items():
            if prev is None:
                sys.modules.pop(key, None)
            else:
                sys.modules[key] = prev
        metrics.reset()

    def series(self):
        return parse_exposition(metrics.render_metrics())


class PayloadSequenceTests(RecordTestBase):
    """Known inputs in; exact counter totals and bucket counts out."""

    def setUp(self):
        super().setUp()
        for payload in PAYLOADS:
            metrics.record_completion("t", payload)

    def test_counter_totals(self):
        s = self.series()
        self.assertEqual(sample(s, "tabby_requests_total"), 3)
        self.assertEqual(sample(s, "tabby_prompt_tokens_total"), 62200)
        self.assertEqual(sample(s, "tabby_generated_tokens_total"), 6650)
        self.assertEqual(sample(s, "tabby_cached_tokens_total"), 57800)
        self.assertAlmostEqual(sample(s, "tabby_prompt_seconds_total"), 0.95)
        self.assertAlmostEqual(sample(s, "tabby_generate_seconds_total"), 371.62)
        self.assertAlmostEqual(sample(s, "tabby_queue_seconds_total"), 1.02)

    def test_generated_tokens_buckets(self):
        s = self.series()
        self.assertEqual(sample(s, "tabby_generated_tokens_count"), 3)
        # 50 -> le=64; 2700 -> le=3072; 3900 -> le=4096 (cumulative)
        self.assertEqual(sample(s, "tabby_generated_tokens_bucket", le="64"), 1)
        self.assertEqual(sample(s, "tabby_generated_tokens_bucket", le="2048"), 1)
        self.assertEqual(sample(s, "tabby_generated_tokens_bucket", le="3072"), 2)
        self.assertEqual(sample(s, "tabby_generated_tokens_bucket", le="4096"), 3)

    def test_decode_rate_buckets(self):
        s = self.series()
        # 80.6 -> le=120; 18.0 and 17.65 -> le=20
        self.assertEqual(sample(s, "tabby_decode_tokens_per_second_count"), 3)
        self.assertEqual(sample(s, "tabby_decode_tokens_per_second_bucket", le="20"), 2)
        self.assertEqual(sample(s, "tabby_decode_tokens_per_second_bucket", le="120"), 3)

    def test_ttft_and_queue_buckets(self):
        s = self.series()
        # ttft = queue + prompt: 0.07 -> le=0.1; 0.4 -> le=0.5; 1.5 -> le=2.5
        self.assertEqual(
            sample(s, "tabby_time_to_first_token_seconds_bucket", le="0.1"), 1
        )
        self.assertEqual(
            sample(s, "tabby_time_to_first_token_seconds_bucket", le="0.5"), 2
        )
        self.assertEqual(
            sample(s, "tabby_time_to_first_token_seconds_bucket", le="2.5"), 3
        )
        # queue: 0.02 -> le=0.05; 0.0 -> le=0.01; 1.0 -> le=1
        self.assertEqual(sample(s, "tabby_queue_seconds_bucket", le="0.05"), 2)
        self.assertEqual(sample(s, "tabby_queue_seconds_bucket", le="1"), 3)


class MissingAndBadKeysTests(RecordTestBase):
    """Payloads with absent or unusable keys record what they can; nothing raises."""

    def test_empty_payload_records_a_request_with_zeros(self):
        metrics.record_completion("t", {})
        s = self.series()
        self.assertEqual(sample(s, "tabby_requests_total"), 1)
        self.assertEqual(sample(s, "tabby_generated_tokens_total"), 0)

    def test_none_and_indeterminate_values(self):
        metrics.record_completion(
            "t",
            {
                "prompt_tokens": None,
                "prompt_time": "Indeterminate",
                "prompt_tokens_per_sec": "Indeterminate",
                "gen_tokens": 10,
                "gen_time": None,
                "gen_tokens_per_sec": "Indeterminate",
                "queue_time": None,
                "cached_tokens": None,
            },
            context_len=2048,
        )
        s = self.series()
        self.assertEqual(sample(s, "tabby_requests_total"), 1)
        self.assertEqual(sample(s, "tabby_generated_tokens_total"), 10)
        # prompt_tokens falls back to the context_len argument
        self.assertEqual(sample(s, "tabby_prompt_tokens_total"), 2048)
        self.assertEqual(sample(s, "tabby_prompt_seconds_total"), 0)
        # no gen_time -> no decode-rate observation at all
        self.assertFalse(
            any(k[0] == "tabby_decode_tokens_per_second_count" for k in s)
        )

    def test_zero_decode_time_skips_rate_histogram(self):
        metrics.record_completion("t", {"gen_tokens": 5, "gen_time": 0})
        s = self.series()
        self.assertFalse(
            any(k[0] == "tabby_decode_tokens_per_second_bucket" for k in s)
        )
        # the zero-time request still lands in the tokens and seconds counters
        self.assertEqual(sample(s, "tabby_generated_tokens_total"), 5)

    def test_negative_and_bool_values_clamped(self):
        metrics.record_completion(
            "t", {"gen_tokens": -5, "queue_time": -0.5, "cached_tokens": True}
        )
        s = self.series()
        self.assertEqual(sample(s, "tabby_generated_tokens_total"), 0)
        self.assertEqual(sample(s, "tabby_queue_seconds_total"), 0)
        self.assertEqual(sample(s, "tabby_cached_tokens_total"), 0)

    def test_non_dict_payload(self):
        metrics.record_completion("t", None)
        s = self.series()
        self.assertEqual(sample(s, "tabby_requests_total"), 1)


class FailureTests(RecordTestBase):
    """A failure inside the metrics code is swallowed and logged once."""

    def test_record_failure_swallowed_warned_once_counted(self):
        with patch.object(metrics, "_model_label", side_effect=RuntimeError("boom")):
            metrics.record_completion("t", {"gen_tokens": 5})
            metrics.record_completion("t", {"gen_tokens": 5})
        s = self.series()
        self.assertEqual(s[("tabby_metrics_record_errors_total", ())], 2)
        self.assertEqual(len(self.warnings), 1)
        # the failed records added nothing
        self.assertFalse(any(k[0] == "tabby_requests_total" for k in s))


class LogMetricsHookTests(RecordTestBase):
    """The real log_metrics function feeds the registry (needs app deps)."""

    @unittest.skipIf(gen_logging is None, "common.gen_logging needs the app deps")
    def test_log_metrics_records(self):
        gen_logging.log_metrics("#1 chat/completions", dict(PAYLOADS[0]), 1200, 131072)
        s = self.series()
        self.assertEqual(sample(s, "tabby_requests_total"), 1)
        self.assertEqual(sample(s, "tabby_generated_tokens_total"), 50)
        self.assertAlmostEqual(sample(s, "tabby_generate_seconds_total"), 0.62)


if __name__ == "__main__":
    unittest.main()
