import json
from pathlib import Path
import sys
import unittest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from fn_driver import AuditedResponse


class Response(list):
    def __enter__(self):return self
    def __exit__(self,*args):return False

class TestWire(unittest.TestCase):
    def setUp(self):
        p=Path(__file__).resolve().parents[1]
        events=json.loads((p/'fixtures/real-sse.json').read_text())
        self.lines=[('data: '+json.dumps(e)+'\n').encode() for e in events]+[b'data: [DONE]\n']
    def test_actual_bytes_preserved(self):
        errors=[]
        with AuditedResponse(Response(self.lines),errors) as r:
            self.assertEqual(list(r),self.lines)
        self.assertEqual(errors,[])
    def test_truncated_stream_rejected(self):
        errors=[]
        with self.assertRaises(ValueError):
            with AuditedResponse(Response(self.lines[:-1]),errors) as r:list(r)
        self.assertTrue(errors)
    def test_malformed_json_rejected(self):
        errors=[]
        with self.assertRaises((ValueError,AssertionError)):
            with AuditedResponse(Response([b'data: {invalid}\n']),errors) as r:list(r)
        self.assertTrue(errors)

class TestSingleFrameIdentity(unittest.TestCase):
    def test_single_frame_is_valid_identity_but_not_speed(self):
        import tempfile
        import gate
        events=[dict(choices=[dict(index=0,text='Paris',finish_reason='stop')]),
                dict(choices=[],usage=dict(prompt_tokens=8,completion_tokens=1))]
        value,usage,first,last,frames=gate.consume(['data: '+json.dumps(e) for e in events]+['data: [DONE]'])
        self.assertEqual(first,last);self.assertEqual(frames,1)
        with tempfile.TemporaryDirectory() as t:
            p=Path(t)/'row.jsonl'
            p.write_text(json.dumps(dict(ok=True,value=value,completion_tokens=1,client_frames=1,decode_tps=None))+'\n')
            self.assertEqual(len(gate.read_rows(p,require_decode=False)),1)
            with self.assertRaises(AssertionError):gate.read_rows(p)
