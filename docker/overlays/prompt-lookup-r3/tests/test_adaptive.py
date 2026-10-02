import unittest
from selftest import helper
m = helper()


def step(a, hit=True, accepted=2, eligible=True):
    state, check = a.begin(eligible)
    if check:
        match = m.PromptLookupMatch(0, 16, (1,2,3)) if hit else None
        a.match(match)
        a.opened(1, match, 3)
        if state != 'OFF':
            for pos in range(accepted+1): a.observe(pos, pos+1)
    a.finish()
    return state


class TestAdaptive(unittest.TestCase):
    def test_no_lookahead_and_probation(self):
        a=m.AdaptiveLookup()
        self.assertEqual([step(a) for _ in range(24)], ['PROBE']*24)
        self.assertEqual(a.state,'PROBE')
        self.assertEqual(step(a),'ON')
        self.assertEqual(a.metrics['lookup_enable_transitions'],1)

    def test_low_acceptance_never_on(self):
        a=m.AdaptiveLookup()
        states=[step(a, i%5==0, 0) for i in range(512)]
        self.assertNotIn('ON',states)
        self.assertIn('OFF',states)

    def test_cycle_hysteresis_and_reprobe(self):
        a=m.AdaptiveLookup()
        for _ in range(25):step(a)
        for i in range(30): step(a, True, 1 if i%4==0 else 2)
        self.assertEqual(a.state,'ON') # .875 > exit .85, but below entry .9
        for _ in range(80):step(a,True,0)
        self.assertEqual(a.state,'OFF')
        before=a.metrics['lookup_reprobe_checks']
        states=[step(a) for _ in range(300)]
        self.assertIn('PROBE',states);self.assertIn('ON',states)
        self.assertGreater(a.metrics['lookup_reprobe_checks'],before)
        self.assertGreaterEqual(a.metrics['lookup_reprobe_transitions'],1)

    def test_requeue_object_and_bounded_history(self):
        from pathlib import Path
        import ast
        job=Path(__file__).resolve().parents[1]/'overlay/exllamav3/generator/job.py'
        tree=ast.parse(job.read_text())
        prepare=next(n for n in ast.walk(tree) if isinstance(n,ast.FunctionDef) and n.name=='prepare_for_requeue')
        self.assertIn('"prompt_lookup_adaptive": self.prompt_lookup_adaptive',ast.unparse(prepare).replace("'",'"'))
        a=m.AdaptiveLookup();[step(a) for _ in range(200)]
        rq={'prompt_lookup_adaptive':a};b=rq['prompt_lookup_adaptive']
        self.assertIs(a,b);self.assertLessEqual(len(b.history),64)
        self.assertEqual(b.state,'ON')

    def test_batch_ineligible_does_no_checks(self):
        a=m.AdaptiveLookup()
        for _ in range(100):self.assertEqual(step(a,eligible=False),'PROBE')
        self.assertEqual(a.probe_age,0)
        self.assertEqual(sum(a.metrics[k] for k in a.metrics if k.startswith('lookup_match_')),0)
        self.assertEqual(a.metrics['lookup_ineligible_steps'],100)

    def test_opener_histogram_shadow_prefix(self):
        a=m.AdaptiveLookup();a.begin(True)
        match=m.PromptLookupMatch(0,40,(1,2,3));a.match(match)
        self.assertIsNone(a.opened(9,match,3));a.finish()
        self.assertEqual(a.metrics['lookup_match_32_plus'],1)
        self.assertEqual(a.metrics['lookup_opener_mismatches'],1)
        a.begin(True);a.opened(1,match,3)
        a.observe(0,1);a.observe(1,9);a.observe(2,3);a.finish()
        self.assertEqual(a.metrics['lookup_shadow_accepted'],0)
        self.assertEqual(a.metrics['lookup_shadow_proposed'],2)

    def test_env_validation(self):
        c=m.AdaptiveConfig.from_env({'EXL3_PROMPT_LOOKUP_WINDOW':'48','EXL3_PROMPT_LOOKUP_MIN_HIT':'.6'})
        self.assertEqual(c.window,48);self.assertEqual(c.min_hit,.6)
        for env in ({'EXL3_PROMPT_LOOKUP_REPROBE':'0'}, {'EXL3_PROMPT_LOOKUP_MIN_ACC':'nan'}, {'EXL3_PROMPT_LOOKUP_MIN_PROPOSALS':'65'}):
            with self.assertRaises(ValueError):m.AdaptiveConfig.from_env(env)
