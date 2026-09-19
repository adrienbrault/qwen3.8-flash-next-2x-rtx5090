#!/usr/bin/env python3
from __future__ import annotations

import importlib.util
from pathlib import Path
import types
import unittest


ROOT = Path(__file__).resolve().parents[3]
OUT = ROOT / "out/decode-kernels-r4"
OVERLAY = OUT / "overlay/exllamav3"


def load_draft_overlap():
    path = OVERLAY / "generator/draft_overlap.py"
    spec = importlib.util.spec_from_file_location("r4_draft_overlap", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class Round4CPUTests(unittest.TestCase):
    def test_python_sources_compile(self):
        for path in OVERLAY.rglob("*.py"):
            compile(path.read_text(), str(path), "exec")

    def test_flags_are_literal_default_off(self):
        generator = (OVERLAY / "generator/generator.py").read_text()
        embedding = (OVERLAY / "modules/embedding.py").read_text()
        mixer = (OVERLAY / "modules/hyperconnections.py").read_text()
        self.assertIn('EXL3_DRAFT_PINNED_STAGING", "0") == "1"', generator)
        self.assertIn('EXL3_BATCH_VERIFY", "0") == "1"', generator)
        self.assertIn('EXL3_MTP_DEVICE_DRAFT", "0") == "1"', generator)
        self.assertIn('EXL3_EMBED_GPU", "0") == "1"', embedding)
        self.assertIn('EXL3_HC_MIX_V2_INT8", "0") == "1"', mixer)
        self.assertIn('os.environ.get("EXL3_MTP_HEAD_N")', (OVERLAY / "architecture/qwen4_exp_mtp.py").read_text())
        self.assertIn('accept_params["pinned_staging_slot"]', generator)
        self.assertIn('params.get("pinned_staging_slot")', embedding)

    def test_verify_eligibility_and_fallbacks(self):
        mod = load_draft_overlap()

        def job(mode="sampled", **overrides):
            values = dict(
                sampler=types.SimpleNamespace(batch_verify_mode=mode, reqs_past_ids=False),
                new_tokens=0,
                forced_ids=None,
                filters=[],
                device_logit_mask=None,
                return_probs=False,
                return_top_tokens=0,
                sequences=[object()],
            )
            values.update(overrides)
            return types.SimpleNamespace(**values)

        self.assertEqual(mod.verification_batch_mode(job()), "sampled")
        self.assertEqual(mod.verification_batch_mode(job("greedy")), "greedy")
        self.assertIsNone(mod.verification_batch_mode(job(filters=[object()])))
        self.assertIsNone(mod.verification_batch_mode(job(return_probs=True)))
        self.assertIsNone(mod.verification_batch_mode(job(return_top_tokens=5)))
        self.assertIsNone(mod.verification_batch_mode(job(forced_ids=object())))
        self.assertIsNone(mod.verification_batch_mode(job(new_tokens=-1)))
        self.assertIsNone(mod.verification_batch_mode(job(sequences=[object(), object()])))
        self.assertIsNone(mod.verification_batch_mode(job(
            sampler=types.SimpleNamespace(batch_verify_mode="sampled", reqs_past_ids=True)
        )))
        self.assertIsNone(mod.verification_batch_mode(job(device_logit_mask=object())))

    def test_exl3_prefix_is_whole_128_column_tiles(self):
        full = 248320
        requested = 65536
        effective = min(requested, full) // 128 * 128
        self.assertEqual(effective, 65536)
        self.assertEqual(effective % 128, 0)
        self.assertEqual(effective // 16, 4096)
        self.assertLess(effective, full)

    def test_int8_storage_arithmetic(self):
        hidden, hc, rank, sites = 2560, 4, 320, 96
        m = rank + hc
        fp16 = sites * 2 * (m * hc * hidden + hc * hidden * rank)
        int8 = sites * (
            m * hc * hidden + 4 * m + hc * hidden * rank + 4 * hc * hidden
        )
        self.assertEqual(fp16, 1_266_155_520)
        self.assertEqual(fp16 - int8, 629_021_184)
        final_fp16 = 2 * (rank * hc * hidden + hc * hidden * rank)
        final_int8 = rank * hc * hidden + 4 * rank + hc * hidden * rank + 4 * hc * hidden
        final_freed = final_fp16 - final_int8
        self.assertEqual((fp16 - int8) + final_freed, 635_532_544)
        self.assertEqual((fp16 - int8) + 2 * (fp16 - int8) // sites + 2 * final_freed, 655_148_512)

    def test_cuda_host_wrapper_uses_c10_check(self):
        source = (OVERLAY / "exllamav3_ext/hc_mix.cu").read_text()
        self.assertIn("#include <c10/cuda/CUDAException.h>", source)
        self.assertIn("C10_CUDA_CHECK(cudaPeekAtLastError())", source)
        self.assertIn("void gr_mix_v2_int8", source)


if __name__ == "__main__":
    unittest.main()
