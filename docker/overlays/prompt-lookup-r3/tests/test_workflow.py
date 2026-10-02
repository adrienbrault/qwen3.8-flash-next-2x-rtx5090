"""Guard generation and post-promotion failure gates without a daemon or endpoint."""
import json
from pathlib import Path
import subprocess
import tempfile
import unittest
import gate
import promote
import replay_gate


class TestWorkflow(unittest.TestCase):
    def fixture(self,root):
        replay_gate.make(root)
        for suite in ('fn','chat','fnstyle'):
            (root/f'identity-PROMOTE-{suite}.jsonl').write_bytes((root/f'identity-OFF2-{suite}.jsonl').read_bytes())
        (root/'fn-PROMOTE-c1-code.jsonl').write_bytes((root/'fn-OFF2-c1-code.jsonl').read_bytes())
        (root/'boot-promote.log').write_text('FASTWARM ok\n')
        gate.dump(root/'vram-reference.json',[1100,1600]);gate.dump(root/'vram-promote.json',[1068,1568])

    def test_promotion_pass_and_failures(self):
        for mutation in (None,'warm','vram','fn','identity','missing'):
            with self.subTest(mutation=mutation),tempfile.TemporaryDirectory() as d:
                root=Path(d);self.fixture(root)
                if mutation=='warm':(root/'boot-promote.log').write_text('FASTWARM FAILED\n')
                if mutation=='vram':gate.dump(root/'vram-promote.json',[1067,1568])
                if mutation=='fn':replay_gate.change_rates(root,'PROMOTE',98,'c1-code')
                if mutation=='identity':
                    path=root/'identity-PROMOTE-chat.jsonl'
                    rows=[json.loads(l) for l in path.read_text().splitlines()];rows[0]['value']['content']='flip'
                    path.write_text(''.join(json.dumps(r)+'\n' for r in rows))
                if mutation=='missing':(root/'vram-promote.json').unlink()
                if mutation is None:self.assertEqual(promote.verify(root)['status'],'PASS')
                else:
                    with self.assertRaises((AssertionError,FileNotFoundError)):promote.verify(root)

    def test_no_boot_guard_launcher_runs_original_pin_checks(self):
        # Fake commands, no Docker invocation. Execute the actual candidate
        # launcher until PREBOOT exit, then assert no destructive command occurred.
        live=(gate.P/'fixtures/live-launcher.sh').read_text()
        candidate=gate.candidate(live,'1','sha256:'+'1'*64)
        self.assertIn('R825C_IMAGE_ID=sha256:'+'1'*64,candidate)
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);(root/'bin').mkdir();(root/'ckpt').mkdir();(root/'logs').mkdir()
            (root/'models/qwen3.8-flash-next-exl3-2.50bpw-r0b0tlab').mkdir(parents=True)
            fake=root/'bin/sudo'
            fake.write_text('''#!/bin/bash
printf '%s\\n' "$*" >> "$TEST_RECORD"
if [ "$1 $2 $3" = 'docker image inspect' ]; then
  if [ "$4" = -f ]; then echo "$TEST_IMAGE_ID"; fi
  exit 0
fi
if [ "$1 $2" = 'docker exec' ]; then cat >/dev/null; exit "${TEST_VALIDATE_RC:-0}"; fi
exit 97
''');fake.chmod(0o755)
            # Redirect host paths for a CPU fixture; generated guard remains verbatim.
            candidate=candidate.replace('/srv/qwen5090',str(root))
            # checkpoint path includes /models/{name}; make its default directory.
            import re
            # Avoid depending on the checkpoint name: override it using CKPT env.
            script=root/'candidate.sh';script.write_text(candidate)
            for image_id,validation_rc,success in [('sha256:'+'1'*64,'0',True),('sha256:'+'2'*64,'0',False),('sha256:'+'1'*64,'1',False)]:
                import os
                env=dict(os.environ,HOME=str(root/'home'),PATH=str(root/'bin')+':'+os.environ['PATH'],
                    R828_PREBOOT_ONLY='1',R828_PREFLIGHT_DIR=str(root/'preflight'),
                    CKPT=str(root/'ckpt'),TEST_RECORD=str(root/'calls'),
                    TEST_IMAGE_ID=image_id,TEST_VALIDATE_RC=validation_rc)
                (root/'preflight').mkdir(exist_ok=True);(root/'calls').write_text('')
                run=subprocess.run(['bash',str(script)],env=env,capture_output=True,text=True)
                self.assertEqual(run.returncode==0,success,run.stdout+run.stderr)
                calls=(root/'calls').read_text()
                self.assertNotIn('docker run',calls);self.assertNotIn('docker rm',calls)
                if success:self.assertIn('R828 PREBOOT PASS',run.stdout)

    def test_unit_rollback_and_commit_order(self):
        path=gate.P.parent/'r828-prompt-lookup-r3.sh'
        unit=(path if path.exists() else gate.P/'fixtures/r828-unit.sh').read_text()
        self.assertIn('BOOTED=1\n  served_stop',unit)
        self.assertIn('cp -p "$LIVE.pre-r828" "$LIVE.new" && mv "$LIVE.new" "$LIVE"',unit)
        self.assertIn('[ "$BOOTED" = 1 ] && [ "$PROMOTED" = 0 ]',unit)
        self.assertLess(unit.index('MUTATED=1\ncp'),unit.index('mv "$LIVE.new" "$LIVE" || fail'))
        self.assertLess(unit.index('"$Q/promote.py" --root'),unit.index('PROMOTED=1\nlog'))
