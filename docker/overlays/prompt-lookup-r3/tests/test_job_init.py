"""R828 try 1: Job.__init__ read self.generator.prompt_lookup_config, but TabbyAPI constructs AsyncJob/Job before
prepare_for_queue attaches the generator, so every request failed with AttributeError on a NoneType."""
import ast
import unittest
from pathlib import Path

JOB = Path(__file__).resolve().parents[1] / 'overlay/exllamav3/generator/job.py'


class JobInitDoesNotUseGenerator(unittest.TestCase):
    def test_init_never_dereferences_self_generator(self):
        tree = ast.parse(JOB.read_text())
        job = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == 'Job')
        init = next(n for n in job.body if isinstance(n, ast.FunctionDef) and n.name == '__init__')
        uses = [n.lineno for n in ast.walk(init)
                if isinstance(n, ast.Attribute) and isinstance(n.value, ast.Attribute)
                and n.value.attr == 'generator' and isinstance(n.value.value, ast.Name) and n.value.value.id == 'self']
        self.assertEqual(uses, [], f'Job.__init__ dereferences self.generator at lines {uses}')

    def test_prepare_for_queue_builds_adaptive_state(self):
        src = JOB.read_text()
        body = src[src.index('def prepare_for_queue('):]
        body = body[:body.index('\n    def ', 1)]
        self.assertIn('AdaptiveLookup(generator.prompt_lookup_config)', body)


if __name__ == '__main__':
    unittest.main()
