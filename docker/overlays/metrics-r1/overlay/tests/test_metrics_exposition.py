"""
Unit tests for the Prometheus exposition writer in common.metrics.

common.metrics is stdlib-only at import time, so these tests run anywhere with
the tree on sys.path; no app dependencies needed. The model container is
faked through sys.modules, the same lazy lookup the module uses in production.
"""

import pathlib
import sys
import unittest
from types import SimpleNamespace

from common import metrics


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


def parse_header(text, field):
    """Parse '# <field> <name> <rest>' lines -> {name: rest}."""

    prefix = f"# {field} "
    out = {}
    for line in text.splitlines():
        if line.startswith(prefix):
            name, _, value = line[len(prefix) :].partition(" ")
            out[name] = value
    return out


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


class MetricsTestBase(unittest.TestCase):
    def setUp(self):
        metrics.reset()
        self._prev_model = sys.modules.get("common.model")
        sys.modules["common.model"] = SimpleNamespace(container=FakeContainer())

    def tearDown(self):
        if self._prev_model is None:
            sys.modules.pop("common.model", None)
        else:
            sys.modules["common.model"] = self._prev_model
        metrics.reset()


class FamilyDeclarationTests(MetricsTestBase):
    def test_every_family_has_help_and_type(self):
        text = metrics.render_metrics()
        helps = parse_header(text, "HELP")
        types = parse_header(text, "TYPE")
        names = (
            [c.name for c in metrics._COUNTERS]
            + [h.name for h in metrics._HISTOGRAMS]
            + [metrics._REQUESTS_ACTIVE[0]]
            + [name for name, _doc, _attr in metrics._CONTAINER_GAUGES]
        )
        for name in names:
            self.assertIn(name, helps)
            self.assertIn(name, types)
        for fam in metrics._COUNTERS:
            self.assertEqual(types[fam.name], "counter")
        for fam in metrics._HISTOGRAMS:
            self.assertEqual(types[fam.name], "histogram")
        self.assertEqual(types["tabby_requests_active"], "gauge")
        self.assertEqual(types["tabby_max_seq_len"], "gauge")

    def test_empty_registry_emits_headers_only(self):
        metrics.reset()
        sys.modules["common.model"] = SimpleNamespace(container=None)
        text = metrics.render_metrics()
        series = parse_exposition(text)
        self.assertEqual(series, {})


class CounterAndGaugeTests(MetricsTestBase):
    def test_counter_series_carries_model_label(self):
        metrics.record_completion("t", {"gen_tokens": 5, "gen_time": 1.0})
        series = parse_exposition(metrics.render_metrics())
        self.assertEqual(sample(series, "tabby_requests_total"), 1)
        self.assertEqual(sample(series, "tabby_generated_tokens_total"), 5)

    def test_live_gauges_from_container(self):
        sys.modules["common.model"].container.active_job_ids = {"a": None, "b": None}
        series = parse_exposition(metrics.render_metrics())
        self.assertEqual(sample(series, "tabby_requests_active"), 2)
        self.assertEqual(sample(series, "tabby_cache_size_tokens"), 262144)
        self.assertEqual(sample(series, "tabby_max_batch_size"), 4)
        self.assertEqual(sample(series, "tabby_max_seq_len"), 131072)

    def test_gauges_absent_without_container(self):
        sys.modules["common.model"] = SimpleNamespace(container=None)
        metrics.record_completion("t", {"gen_tokens": 5})
        series = parse_exposition(metrics.render_metrics())
        # Recorded data falls back to the last-known/unknown label
        self.assertEqual(
            series[("tabby_requests_total", (("model", "unknown"),))], 1
        )
        for name in ("tabby_requests_active", "tabby_max_seq_len"):
            self.assertFalse(any(k[0] == name for k in series))

    def test_label_value_is_escaped(self):
        sys.modules["common.model"].container.model_dir = pathlib.Path('/m/evil"\\\nname')
        metrics.record_completion("t", {"gen_tokens": 1})
        text = metrics.render_metrics()
        self.assertIn('model="evil\\"\\\\\\nname"', text)

    def test_numbers_render_plainly(self):
        metrics.record_completion("t", {"gen_tokens": 3, "queue_time": 0.5})
        text = metrics.render_metrics()
        self.assertIn('tabby_requests_total{model="test-model"} 1\n', text)
        self.assertIn('tabby_queue_seconds_total{model="test-model"} 0.5\n', text)


class HistogramShapeTests(MetricsTestBase):
    def test_bucket_sum_count_consistent(self):
        for tokens in (50, 2700, 3900):
            metrics.record_completion("t", {"gen_tokens": tokens})
        series = parse_exposition(metrics.render_metrics())

        bounds = metrics.GENERATED_TOKEN_BUCKETS
        le_counts = [
            sample(series, "tabby_generated_tokens_bucket", le=str(int(b)))
            for b in bounds
        ]
        # Cumulative and non-decreasing, +Inf catches everything
        self.assertEqual(le_counts, sorted(le_counts))
        self.assertEqual(
            sample(series, "tabby_generated_tokens_bucket", le="+Inf"), 3
        )
        self.assertEqual(sample(series, "tabby_generated_tokens_count"), 3)
        self.assertEqual(sample(series, "tabby_generated_tokens_sum"), 6650)
        # 50 -> le=64, 2700 -> le=3072, 3900 -> le=4096
        self.assertEqual(le_counts[1], 1)  # le=64
        self.assertEqual(le_counts[6], 2)  # le=3072
        self.assertEqual(le_counts[7], 3)  # le=4096

    def test_bucket_boundary_is_inclusive(self):
        metrics.record_completion("t", {"gen_tokens": 3072})
        series = parse_exposition(metrics.render_metrics())
        self.assertEqual(
            sample(series, "tabby_generated_tokens_bucket", le="3072"), 1
        )
        self.assertEqual(
            sample(series, "tabby_generated_tokens_bucket", le="2048"), 0
        )


if __name__ == "__main__":
    unittest.main()
