import asyncio
import os
import unittest
from types import SimpleNamespace as NS
from unittest.mock import patch

from _bootstrap import install

install()
from fastapi import HTTPException
from common.sampling import BaseSamplerRequest
from endpoints.OAI.utils import common_
from exllamav3.generator import name_policy as policy


class NamesAndLoopDefaults(unittest.TestCase):
    def test_fast_alias_reuses_loaded_model_and_fails_when_disabled(self):
        loaded = NS(loaded=True, model_dir=NS(name=policy.RAW))
        with patch.object(common_.model, "container", loaded):
            with patch.dict(os.environ, EXL3_FLASHNEXT_FAST="1"):
                asyncio.run(common_.load_inline_model(policy.FAST, None))
                self.assertEqual(policy.response_name(policy.FAST, policy.RAW), policy.FAST)
            with patch.dict(os.environ, EXL3_FLASHNEXT_FAST="0"):
                with self.assertRaises(HTTPException) as err:
                    asyncio.run(common_.load_inline_model(policy.FAST, None))
                self.assertEqual(err.exception.status_code, 400)

    def test_served_and_upstream_loop_modes(self):
        self.assertEqual(BaseSamplerRequest().get_stop_on_loop(), (800, 2))
        self.assertEqual(BaseSamplerRequest(loop_detect_window=90).get_stop_on_loop(), (90, 2))
        self.assertIsNone(BaseSamplerRequest(loop_detect_window=0).get_stop_on_loop())
        self.assertIs(BaseSamplerRequest(loop_detect=True).get_stop_on_loop(), True)
        self.assertIs(BaseSamplerRequest(loop_detect=False).get_stop_on_loop(), False)


if __name__ == "__main__":
    unittest.main()
