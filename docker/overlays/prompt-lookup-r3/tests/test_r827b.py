import hashlib
import json
import unittest
import gate
import replay_r827b as replay


class TestRealCounterReplay(unittest.TestCase):
    def test_original_counter_provenance_and_join(self):
        root=gate.P/'fixtures/r827b'
        provenance=json.loads((root/'provenance.json').read_text())
        self.assertEqual(len(provenance),14)
        for name,source in provenance.items():
            data=json.loads((root/name).read_text())
            self.assertEqual(hashlib.sha256((root/name).read_bytes()).hexdigest(),source['sha256'])
            joined=json.loads((root/name.replace('counters-','joined-')).read_text())
            self.assertEqual([r['counter'] for r in joined],data['counters'])
            for r in joined:
                c,completion=r['counter'],r['completion']
                self.assertEqual(c['gen_tokens'],c['decode_steps']+completion['draft_accept'])
                self.assertEqual(c['lookup_proposed'],2*c['lookup_hits'])
                for mode in ('balanced','front','back','shuffle'):
                    events=replay.reconstruct(c,mode)
                    self.assertEqual(sum(h for h,a in events),c['lookup_hits'])
                    self.assertEqual(sum(a for h,a in events),c['lookup_accepted'])

    def test_balanced_proxy_separates_real_request_totals(self):
        result=replay.report()
        self.assertIn('NOT observed',result['kind'])
        self.assertGreater(result['shapes']['agent-c1']['balanced_on_fraction'],.9)
        self.assertEqual(result['shapes']['c1-prose']['balanced_on_fraction'],0)
