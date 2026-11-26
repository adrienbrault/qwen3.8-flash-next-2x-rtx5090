"""CPU evidence for the served cutoff policy and its candidate wiring."""

import ast
from pathlib import Path
import os

import torch


def test_qsa_scores_keep_served_fp16_cutoff():
    root = Path(os.environ.get("REBASE167_PKG", "/opt/rebase167/exllamav3-src/exllamav3"))
    idx = (root / "modules/qsa_indexer.py").read_text()
    bc = (root / "modules/attention_fn/bc_attn.py").read_text()
    cls = next(node for node in ast.parse(idx).body if isinstance(node, ast.ClassDef) and node.name == "QSAIndexer")
    dtype = next(node.value for node in cls.body if isinstance(node, ast.Assign)
                 and isinstance(node.targets[0], ast.Name) and node.targets[0].id == "SCORE_DTYPE")
    resolve = compile(ast.Expression(dtype), str(root / "modules/qsa_indexer.py"), "eval")
    assert idx.count("self.SCORE_DTYPE") == 2  # backing and tile merge
    assert 'dtype = idx.SCORE_DTYPE' in bc
    assert '"*fp32:16" if idx.SCORE_DTYPE == torch.float else "*fp16:16"' in bc
    served = torch.tensor([[1.0001, 1.0002], [1.0001, -torch.inf]])
    # These are the prefill and BC ragged cutoff cases: for an FP16 store,
    # the first block wins the rounded tie in both the full and shorter row.
    for env, expected in (({}, [0, 0]), ({"EXL3_QSA_SCORE_FP32": "0"}, [0, 0]),
                          ({"EXL3_QSA_SCORE_FP32": "1"}, [1, 0])):
        selected = eval(resolve, {"torch": torch, "os": type("Env", (), {"environ": env})})
        assert served.to(selected).argmax(dim=1).tolist() == expected
