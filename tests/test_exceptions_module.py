from __future__ import annotations

import unittest

from autosr.exceptions import LLMCallError, LLMParseError
from autosr.llm_client import LLMCallError as ClientLLMCallError
from autosr.llm_client import LLMParseError as ClientLLMParseError


class TestExceptionsModule(unittest.TestCase):
    def test_llm_exceptions_are_defined_in_top_level_module(self) -> None:
        self.assertTrue(issubclass(LLMCallError, RuntimeError))
        self.assertTrue(issubclass(LLMParseError, RuntimeError))

    def test_llm_client_reexports_top_level_exceptions(self) -> None:
        self.assertIs(ClientLLMCallError, LLMCallError)
        self.assertIs(ClientLLMParseError, LLMParseError)


if __name__ == "__main__":
    unittest.main()
