import pytest
from google.genai import errors

from researchpilot.generation.llm import GeminiLLM


class FakeModels:
    def __init__(self, failures, code=503):
        self.failures = failures
        self.code = code
        self.calls = 0

    def generate_content(self, **kwargs):
        self.calls += 1

        if self.calls <= self.failures:
            raise errors.ServerError(
                self.code, {"error": {"message": "overloaded"}}
            )

        return type("R", (), {"text": '{"answer": "ok"}'})()


def make_llm(models, max_retries=3):
    llm = GeminiLLM.__new__(GeminiLLM)
    llm.client = type("C", (), {"models": models})()
    llm.max_retries = max_retries
    llm.base_delay = 0
    return llm


def test_retries_then_succeeds():
    models = FakeModels(failures=2)

    assert make_llm(models).generate("p") == '{"answer": "ok"}'
    assert models.calls == 3


def test_gives_up_after_max_retries():
    models = FakeModels(failures=99)

    with pytest.raises(errors.ServerError):
        make_llm(models, max_retries=2).generate("p")

    assert models.calls == 3


def test_does_not_retry_client_errors():
    models = FakeModels(failures=99, code=400)

    with pytest.raises(errors.APIError):
        make_llm(models).generate("p")

    assert models.calls == 1
