"""Prospective rule and actual R828 raw replay; entirely CPU/read-only source."""
import contextlib
import io
import json
import math
import os
from pathlib import Path
import re
import shutil
import statistics
import tempfile
import unittest
import confirm_c3 as c3
import gate

RAW = Path(os.environ.get('R828_RAW_RESULTS', str(gate.P.parents[2] / 'bench/results/2026-10-02-r828-prompt-lookup-r3-em6iO4')))


class TestRule(unittest.TestCase):
    def test_pass_fail_strict_edge(self):
        for ratio, decision in [(1.0, 'PASS'), (0.98, 'FAIL'), (0.99, 'FAIL'),
                                (math.nextafter(0.99, 1), 'PASS')]:
            with self.subTest(ratio=ratio):
                result = c3.rule([ratio] * 8, [1.0] * 8)
                self.assertEqual(result['DECISION'], decision)
                self.assertEqual(result['sd'], 0)
                self.assertEqual(result['lower_bound'], math.log(ratio))

    def test_sample_sd_arithmetic_and_variance_failure(self):
        logs = [-0.05, 0.06] * 4
        result = c3.rule([math.exp(g) for g in logs], [1.0] * 8)
        expected = sum(logs) / 8 - 1.8946 * math.sqrt(sum((g - sum(logs) / 8) ** 2 for g in logs) / 7) / math.sqrt(8)
        self.assertAlmostEqual(result['lower_bound'], expected, places=14)
        self.assertGreater(result['mean'], 0)
        self.assertEqual(result['DECISION'], 'FAIL')
        # The original failing individual pair does not veto a passing new bound.
        result = c3.rule([0.985] + [1.02] * 7, [1.0] * 8)
        self.assertEqual(result['DECISION'], 'PASS')

    def test_fixed_n_and_invalid_rates(self):
        for n in (0, 7, 9):
            with self.assertRaises(AssertionError): c3.rule([1.] * n, [1.] * n)
        for value in (0., -1., math.inf, math.nan):
            with self.assertRaises(AssertionError): c3.rule([value] + [1.] * 7, [1.] * 8)

    def test_order_and_verbatim_machinery(self):
        path = gate.P.parents[2] / 'scripts/r828c-c3-confirm.sh'
        if not path.exists(): path = gate.P.parent / 'unit.sh'
        if not path.exists(): self.skipTest('unit not included in image selftest')
        unit = path.read_text()
        order = re.search(r'^for tag in (P01_OFF .*); do$', unit, re.M)[1].split()
        self.assertEqual(order, [tag for pair in c3.ORDER for tag in pair])
        header = unit[:unit.index('set -euo pipefail')]
        for pair in c3.ORDER: self.assertIn(' -> '.join(pair), header)
        self.assertEqual(sum(p[0].endswith('_OFF') for p in c3.ORDER), 4)
        self.assertIn('"$Q/confirm_c3.py" declare', unit)
        self.assertLess(unit.index('"$Q/confirm_c3.py" declare'), unit.index('"$Q/selftest.py" >'))
        loop = unit[unit.index('for tag in P01_OFF'):unit.index('"$Q/confirm_c3.py" report')]
        self.assertIn('fn_shape "$tag" 3 code "$flag"', loop)
        self.assertNotIn('identity_suite', loop)
        self.assertNotIn('DECISION', loop)
        original = gate.P.parents[2] / 'scripts/r828-prompt-lookup-r3.sh'
        if not original.exists(): original = gate.P / 'fixtures/r828-unit.sh'
        base = original.read_text()
        for name in ('run', 'snapshot', 'clocks', 'cell_capture', 'identity_suite', 'fn_shape'):
            pattern = r'^' + name + r'\(\)\{.*?^\}'
            self.assertEqual(re.search(pattern, unit, re.M | re.S)[0], re.search(pattern, base, re.M | re.S)[0])
        self.assertEqual(unit[unit.index('# Pre-approved promotion;'):], base[base.index('# Pre-approved promotion;'):])
        self.assertIn('cp -p "$LIVE.pre-r828" "$LIVE.new" && mv "$LIVE.new" "$LIVE"', unit)
        self.assertIn('env -i HOME="$HOME" PATH="$CLEAN_PATH" bash "$LIVE"', unit)


@unittest.skipUnless((RAW / 'packet').is_dir(), 'full private packet archive required for byte-frozen deployment replay; public scalar replay is bench/r827_r828_summary.py')
class TestRealReplay(unittest.TestCase):
    def test_actual_review_numbers_and_retained_reject(self):
        before = (RAW / 'decision.json').read_bytes()
        with tempfile.TemporaryDirectory() as directory, contextlib.redirect_stdout(io.StringIO()):
            root = Path(directory)
            data = c3.replay(RAW, root)
            expected = {'OFF1': (176.24, 0, 0, 3455, 15.383720535962867, 2.667438494934877),
                        'ON1': (178.99, 69, 6, 3411, 15.097269381599865, 2.6482758620689655),
                        'ON2': (177.36, 52, 8, 3403, 15.314502274420775, 2.667438494934877),
                        'OFF2': (179.55, 0, 0, 3405, 15.367569622314274, 2.7066079295154184)}
            for tag, (rate, checks, reads, ineligible, ms, tokens) in expected.items():
                row = data[tag]
                self.assertEqual((row['median'], row['checks'], row['opener_reads'], row['ineligible_steps']),
                                 (rate, checks, [reads, reads], ineligible))
                self.assertAlmostEqual(row['ms_step'], ms)
                self.assertAlmostEqual(row['tokens_step'], tokens)
            self.assertEqual(json.loads((root / 'original-decision.json').read_text())['DECISION'], 'REJECT')
        self.assertEqual((RAW / 'decision.json').read_bytes(), before)

    def fixture(self, root):
        c3.declare(root)
        # Synthetic new pair rates ONLY for a report-path test; raw counters
        # and geometry are replayed from real data, never used as GPU evidence.
        for pair in c3.ORDER:
            for tag in pair:
                source = 'ON1' if tag.endswith('_ON') else 'OFF1'
                for prefix, suffix in [('fn', 'jsonl'), ('counters', 'json'), ('container', 'log')]:
                    dest = root / f'{prefix}-{tag}-c3-code.{suffix}'
                    shutil.copyfile(RAW / f'{prefix}-{source}-c3-code.{suffix}', dest)
                    if prefix == 'fn':
                        rows = [json.loads(line) for line in dest.read_text().splitlines()]
                        for row in rows:
                            row['tag'] = f'{tag}-c3-code'
                            row['decode_tps'] = 101 if tag.endswith('_ON') else 100
                        dest.write_text(''.join(json.dumps(row) + '\n' for row in rows))
                for suffix in ('', '-end'):
                    gate.dump(root / f'inspect-{tag}{suffix}.json', [dict(Id=tag, RestartCount=0)])

    def test_full_eight_pair_report_and_invalid_runs(self):
        for mutation in (None, 'missing', 'extra', 'order', 'continuity', 'reuse', 'counter'):
            with self.subTest(mutation=mutation), tempfile.TemporaryDirectory() as directory, contextlib.redirect_stdout(io.StringIO()):
                root = Path(directory); self.fixture(root)
                if mutation == 'missing': (root / 'fn-P08_ON-c3-code.jsonl').unlink()
                if mutation == 'extra': (root / 'fn-P09_ON-c3-code.jsonl').write_text('{}\n')
                if mutation == 'order': gate.dump(root / 'order.json', dict(order=list(reversed(c3.ORDER))))
                if mutation == 'continuity': gate.dump(root / 'inspect-P01_ON-end.json', [dict(Id='changed', RestartCount=0)])
                if mutation == 'reuse':
                    for suffix in ('', '-end'): gate.dump(root / f'inspect-P01_ON{suffix}.json', [dict(Id='P01_OFF', RestartCount=0)])
                if mutation == 'counter':
                    path = root / 'counters-P08_ON-c3-code.json'
                    data = json.loads(path.read_text()); data['counters'][0]['lookup_checks'] += 1; gate.dump(path, data)
                if mutation is None:
                    result = c3.report(root)
                    self.assertEqual(result['DECISION'], 'PASS')
                    self.assertEqual(len(result['pairs']), 8)
                    self.assertAlmostEqual(result['lower_bound'], math.log(1.01))
                    self.assertEqual(result['pairs'][0]['ON']['opener_reads'], [6, 6])
                else:
                    with self.assertRaises(AssertionError): c3.report(root)


if __name__ == '__main__': unittest.main()
