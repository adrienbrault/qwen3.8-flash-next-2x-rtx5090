import json
from pathlib import Path
import tempfile
import unittest
import gate


class TestCounters(unittest.TestCase):
    def record(self,on=True):
        d=dict.fromkeys(gate.COUNTERS,0)
        d.update(label='#1 completions (stream)',gen_tokens=64,gen_time=.3,decode_steps=25)
        d['lookup_'+('on' if on else 'off')+'_steps']=25
        d['lookup_'+('on' if on else 'off')+'_seconds']=.3
        d['lookup_accept_mtp_0']=25
        d['lookup_accept_mtp_1']=2 if on else 14
        if on:
            d.update(lookup_checks=20,lookup_hits=8,lookup_proposed=16,lookup_accepted=12,
                     lookup_all_hit_rounds=8,lookup_skipped_forwards=16,
                     lookup_accept_copy_1=8,lookup_accept_copy_2=4)
        return d

    def cell(self,d,flag):
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/'log';out=Path(directory)/'out.json'
            path.write_text('#1 completions (stream): 64 tokens generated at 100 t/s · draft 39/75 accepted\nR828_COUNTER '+json.dumps(d)+'\n')
            import contextlib,io
            with contextlib.redirect_stdout(io.StringIO()): gate.cell(str(path),0,1,flag,str(out))
            return json.loads(out.read_text())

    def test_on_off_source_and_position_join(self):
        for on in (False,True):
            d=self.record(on);result=self.cell(d,'1' if on else '0')
            self.assertEqual(result['counters'],[d])

    def test_invalid_residency_skip_source_and_float(self):
        for key,value in [('lookup_on_steps',24),('lookup_skipped_forwards',15),
                          ('lookup_accept_mtp_1',3),('lookup_off_seconds',float('nan')),
                          ('lookup_mixed_rounds',1)]:
            with self.subTest(key=key):
                d=self.record();d[key]=value
                with self.assertRaises(AssertionError):self.cell(d,'1')
