#!/usr/bin/env python3
"""Run the unchanged R813 fn_bench with strict wire auditing.

Its historical reader skips malformed JSON and accepts truncated streams. Audit
before yielding each ORIGINAL byte line, preserving requests and timing formulas.
"""
import json
import sys
import urllib.request
import fn_bench


class AuditedResponse:
    def __init__(self, response, errors):
        self.response=response; self.errors=errors; self.done=False; self.finish=False; self.usage=False
    def __enter__(self):
        self.response.__enter__(); return self
    def __iter__(self):
        for raw in self.response:
            try:
                line=raw.decode('utf-8').strip()
                if line and not line.startswith(':'):
                    assert line.startswith('data:'), 'unknown SSE line'
                    data=line[5:].strip()
                    if data=='[DONE]': self.done=True
                    else:
                        d=json.loads(data)
                        assert 'choices' in d and not d.get('error'), 'SSE error/missing choices'
                        assert isinstance(d['choices'],list)
                        if d.get('usage') is not None:
                            assert type(d['usage']['completion_tokens']) is int
                            assert type(d['usage']['prompt_tokens']) is int
                            self.usage=True
                        for c in d['choices']:
                            assert c['index']==0
                            assert any(k in c for k in ('text','delta','message'))
                            self.finish |= bool(c.get('finish_reason'))
            except Exception as error:
                self.errors.append(str(error)); raise
            yield raw
    def __exit__(self,*args):
        result=self.response.__exit__(*args)
        if not (self.done and self.finish and self.usage):
            self.errors.append('missing DONE/finish/usage')
            raise ValueError('fn_bench truncated or incomplete SSE')
        return result


def main():
    errors=[]; original=urllib.request.urlopen
    def open_audited(*args,**kwargs):
        return AuditedResponse(original(*args,**kwargs),errors)
    urllib.request.urlopen=open_audited
    try: fn_bench.main()
    finally: urllib.request.urlopen=original
    assert not errors, 'fn_bench wire schema failures: '+repr(errors)

if __name__=='__main__':main()
