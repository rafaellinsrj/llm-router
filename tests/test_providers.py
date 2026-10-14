"""Real adapters are never called here: only their pure helpers and the key lookup."""
import pytest

from llm_router.providers import AnthropicProvider, FakeProvider, GoogleProvider, MissingAPIKey
from llm_router.providers.base import add_tokens, api_key_from_env


def test_api_key_comes_only_from_environment(monkeypatch):
    monkeypatch.delenv("LLM_ROUTER_TEST_KEY", raising=False)
    with pytest.raises(MissingAPIKey):
        api_key_from_env("LLM_ROUTER_TEST_KEY")
    monkeypatch.setenv("LLM_ROUTER_TEST_KEY", "value")
    assert api_key_from_env("LLM_ROUTER_TEST_KEY") == "value"


def test_reasoning_headroom_is_visible_to_the_router():
    provider = AnthropicProvider(reasoning_models=("reasoner-",), reasoning_headroom=4096, effort="low")
    assert provider.billable_max_tokens("reasoner-1", 1000) == 5096
    assert provider.billable_max_tokens("plain-1", 1000) == 1000
    assert provider.request_kwargs("reasoner-1", 1000)["output_config"] == {"effort": "low"}
    assert GoogleProvider().billable_max_tokens("any", 10) == 10


def test_unknown_token_counters_stay_unknown():
    assert add_tokens(10, 5) == 15
    assert add_tokens(10, 0) == 10
    assert add_tokens(None, 5) is None
    assert add_tokens(10, "bad") is None


def test_fake_provider_scripts_in_order_and_repeats_last():
    fake = FakeProvider("x", script={"m": [{"a": 1}, {"a": 2}]})
    assert [fake.complete("m", "s", "u", max_tokens=10).text for _ in range(3)] == ['{"a": 1}', '{"a": 2}', '{"a": 2}']
