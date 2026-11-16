#!/usr/bin/env python3
"""Build the decodekinds-r1 prompt set: Spec-Bench's 480 questions plus 80 MBPP coding prompts, in Spec-Bench's
own jsonl format (question_id, category, turns), so one vllm bench run covers every category and the shim's
pandas-free Spec-Bench loader reads it. Also writes sha16(prompt) -> category, the key the bench shim records.

MBPP stands in for HumanEval/InstructCoder: it is the only code-generation set cached on flan (offline runs).
Rows 0-79 of the test split (task_id 11-90), one assert shown, as in the usual MBPP prompt.
Run with pyarrow available:  uv run --no-project --with pyarrow python prep-dataset.py SPEC_BENCH MBPP_ARROW OUTDIR
"""
import hashlib
import json
import sys
from pathlib import Path

import pyarrow as pa

spec, arrow, out = Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3])
out.mkdir(parents=True, exist_ok=True)
rows = [json.loads(l) for l in spec.read_text().splitlines() if l.strip()]
assert len(rows) == 480, len(rows)
with arrow.open('rb') as f:
    try:
        mbpp = pa.ipc.open_stream(f).read_all().to_pylist()
    except pa.ArrowInvalid:
        f.seek(0)
        mbpp = pa.ipc.open_file(f).read_all().to_pylist()
assert len(mbpp) == 500 and mbpp[0]['task_id'] == 11, (len(mbpp), mbpp[0]['task_id'])
for r in mbpp[:80]:
    rows.append(dict(question_id=f"mbpp-{r['task_id']}", category='mbpp', turns=[
        f"You are an expert Python programmer. {r['text']} Your code should pass this test:\n"
        f"{r['test_list'][0]}\nReply with only the code in one python block."]))
cats = {}
for r in rows:
    sha = hashlib.sha256(r['turns'][0].encode()).hexdigest()[:16]
    assert sha not in cats or cats[sha] == r['category'], ('sha collision across categories', sha)
    cats[sha] = r['category']
(out / 'decodekinds.jsonl').write_text(''.join(json.dumps(r) + '\n' for r in rows))
(out / 'categories.json').write_text(json.dumps(cats, indent=0, sort_keys=True) + '\n')
counts = {}
for c in cats.values():
    counts[c] = counts.get(c, 0) + 1
print(json.dumps(dict(prompts=len(rows), unique=len(cats), per_category=counts)))
