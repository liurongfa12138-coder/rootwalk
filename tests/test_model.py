import asyncio
import json
import sys
from types import ModuleType, SimpleNamespace
from typing import Any

import pytest

import rootwalk.model as model_module
from rootwalk.model import (
    ModelAdapter,
    ModelConfigurationError,
    ModelProviderError,
    ModelResponseError,
    ModelTimeoutError,
    RecordedModelAdapter,
    redact_sensitive,
)


def test_runtime_context_can_require_nonempty_disposition_array() -> None:
    constrained = model_module._decision_json_schema(  # pyright: ignore[reportPrivateUsage]
        [
            {"role": "system", "content": "investigate"},
            {
                "role": "user",
                "content": json.dumps(
                    {
                        "frontier_control": {
                            "response_schema_constraints": {
                                "anomaly_disposition_update_min_items": 1
                            }
                        }
                    }
                ),
            },
        ]
    )
    unconstrained = model_module._decision_json_schema(  # pyright: ignore[reportPrivateUsage]
        [{"role": "user", "content": "decide"}]
    )

    constrained_array = constrained["$defs"]["Reflection"]["properties"][
        "anomaly_disposition_update"
    ]
    unconstrained_array = unconstrained["$defs"]["Reflection"]["properties"][
        "anomaly_disposition_update"
    ]
    assert constrained_array["minItems"] == 1
    assert "minItems" not in unconstrained_array


def _decision(summary: str) -> dict[str, Any]:
    return {
        "decision_summary": summary,
        "intent": "Explore",
        "base_revision": 0,
        "operations": [
            {"type": "query", "tool": "metric", "arguments": {"name": "cpu"}}
        ],
    }


class _Response:
    def __init__(self, content: str, *, usage: dict[str, Any] | None = None) -> None:
        self.choices = [SimpleNamespace(message=SimpleNamespace(content=content))]
        self.model = "provider-model"
        self.usage = usage or {"prompt_tokens": 3, "completion_tokens": 5}

    def model_dump(self, *, mode: str) -> dict[str, Any]:
        assert mode == "json"
        return {
            "model": self.model,
            "usage": self.usage,
            "choices": [{"message": {"content": self.choices[0].message.content}}],
        }


class _Completions:
    def __init__(self, responses: list[_Response | BaseException]) -> None:
        self.responses = iter(responses)
        self.calls: list[dict[str, Any]] = []

    async def create(self, **kwargs: Any) -> _Response:
        self.calls.append(kwargs)
        response = next(self.responses)
        if isinstance(response, BaseException):
            raise response
        return response


def _install_fake_ark(
    monkeypatch: pytest.MonkeyPatch,
    responses: list[_Response | BaseException],
) -> tuple[_Completions, dict[str, Any]]:
    completions = _Completions(responses)
    client = SimpleNamespace(chat=SimpleNamespace(completions=completions))
    init_kwargs: dict[str, Any] = {}

    def async_ark(**kwargs: Any) -> Any:
        init_kwargs.update(kwargs)
        return client

    module = ModuleType("volcenginesdkarkruntime")
    module.AsyncArk = async_ark  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, module.__name__, module)
    return completions, init_kwargs


def _sdk_error(
    name: str,
    message: str,
    *,
    status: int | None = None,
    request_id: str = "req-safe-123",
) -> Exception:
    error = type(name, (Exception,), {})(message)
    error.message = message  # type: ignore[attr-defined]
    error.request_id = request_id  # type: ignore[attr-defined]
    error.body = {  # type: ignore[attr-defined]
        "api_key": "never-print-this",
        "request": {
            "messages": ["private request body says json_schema is unsupported"]
        },
    }
    if status is not None:
        error.status_code = status  # type: ignore[attr-defined]
    return error


def test_model_adapter_requires_environment_key(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("ARK_API_KEY", raising=False)

    with pytest.raises(ModelConfigurationError, match="ARK_API_KEY") as error:
        ModelAdapter(model="configured-model", base_url="https://configured.invalid")

    assert "api_key=" not in str(error.value)


def test_recursive_redaction_preserves_standard_message_content() -> None:
    value = {
        "messages": [
            {"role": "user", "content": "ordinary token budget and cookie policy"}
        ],
        "Authorization": "Bearer credential",
        "nested": {
            "api_key": "credential",
            "access_token": "credential",
            "prompt_tokens": 7,
        },
    }

    assert redact_sensitive(value, ["known-ark-key"]) == {
        "messages": [
            {"role": "user", "content": "ordinary token budget and cookie policy"}
        ],
        "Authorization": "[REDACTED]",
        "nested": {
            "api_key": "[REDACTED]",
            "access_token": "[REDACTED]",
            "prompt_tokens": 7,
        },
    }
    assert (
        redact_sensitive(
            '{"Authorization":"Bearer other-secret","message":"ordinary token text"}',
            ["known-ark-key"],
        )
        == '{"Authorization":"[REDACTED]","message":"ordinary token text"}'
    )
    assert (
        redact_sensitive(
            "https://example.invalid/v1?token=other-secret&mode=test",
            ["known-ark-key"],
        )
        == "https://example.invalid/v1?token=[REDACTED]&mode=test"
    )
    assert (
        redact_sensitive("Authorization: Bearer other-secret", ["known-ark-key"])
        == "Authorization: [REDACTED]"
    )
    assert redact_sensitive("Bearer other-secret", ["known-ark-key"]) == (
        "Bearer [REDACTED]"
    )
    assert redact_sensitive(
        "https://example.invalid/v1?client_secret=outside-secret&custom_token=value",
        ["known-ark-key"],
    ) == ("https://example.invalid/v1?client_secret=[REDACTED]&custom_token=[REDACTED]")


def test_model_adapter_repairs_once_and_aggregates_all_response_usage(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    expected = _decision("inspect cpu")
    first_usage = {
        "prompt_tokens": 3,
        "completion_tokens": 5,
        "prompt_tokens_details": {"cached_tokens": 1, "source": "first"},
        "completion_tokens_details": {"reasoning_tokens": 2},
        "service_tier": "standard",
    }
    final_usage = {
        "prompt_tokens": 7,
        "completion_tokens": 11,
        "prompt_tokens_details": {"cached_tokens": 4, "source": "final"},
        "completion_tokens_details": {"reasoning_tokens": 6},
        "service_tier": "priority",
    }
    completions, init_kwargs = _install_fake_ark(
        monkeypatch,
        [
            _Response("not-json", usage=first_usage),
            _Response(json.dumps(expected), usage=final_usage),
        ],
    )
    monkeypatch.setenv("ARK_API_KEY", "never-print-this")
    adapter = ModelAdapter(
        model="configured-model",
        base_url="https://configured.invalid",
        timeout_s=2,
    )

    result = asyncio.run(adapter.decide([{"role": "user", "content": "decide"}]))

    assert result.decision.decision_summary == "inspect cpu"
    assert result.provider_model == "provider-model"
    assert result.usage == {
        "prompt_tokens": 10,
        "completion_tokens": 16,
        "prompt_tokens_details": {"cached_tokens": 5, "source": "final"},
        "completion_tokens_details": {"reasoning_tokens": 8},
        "cache_read_input_tokens": 5,
        "reasoning_tokens": 8,
        "service_tier": "priority",
    }
    assert result.raw_response["usage"] == final_usage
    assert init_kwargs == {
        "api_key": "never-print-this",
        "base_url": "https://configured.invalid",
        "timeout": 2,
    }
    assert len(completions.calls) == 2
    assert completions.calls[0]["max_tokens"] == 2_000
    assert completions.calls[0]["response_format"]["json_schema"]["strict"] is True
    repair_messages = completions.calls[1]["messages"]
    assert repair_messages[-2] == {"role": "assistant", "content": "not-json"}
    assert "Return only corrected JSON" in repair_messages[-1]["content"]
    assert "Validation errors" in repair_messages[-1]["content"]
    assert (
        "Finalize has exactly three mutually exclusive modes"
        in repair_messages[-1]["content"]
    )
    assert "Schema:" not in repair_messages[-1]["content"]


def test_repair_feedback_preserves_pydantic_validation_path_and_message(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    invalid = {**_decision("invalid intent"), "intent": "Investigate"}
    completions, _init_kwargs = _install_fake_ark(
        monkeypatch,
        [
            _Response(json.dumps(invalid)),
            _Response(json.dumps(_decision("corrected intent"))),
        ],
    )
    monkeypatch.setenv("ARK_API_KEY", "never-print-this")

    result = asyncio.run(
        ModelAdapter(
            model="configured-model", base_url="https://configured.invalid"
        ).decide([{"role": "user", "content": "decide"}])
    )

    assert result.decision.decision_summary == "corrected intent"
    feedback = completions.calls[1]["messages"][-1]["content"]
    assert '"path":["intent"]' in feedback
    assert '"message":"Input should be' in feedback


def test_repair_prompt_preserves_occurrence_finalize_mode(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    invalid = {
        "decision_summary": "mixed occurrence and abstention",
        "operations": [
            {
                "type": "finalize",
                "occurrence_evidence_refs": ["evidence-1"],
                "abstain_reason": "should not accompany occurrence evidence",
            }
        ],
    }
    corrected = {
        "decision_summary": "recommend occurrence evidence",
        "operations": [
            {
                "type": "finalize",
                "occurrence_evidence_refs": ["evidence-1"],
            }
        ],
    }
    completions, _init_kwargs = _install_fake_ark(
        monkeypatch,
        [_Response(json.dumps(invalid)), _Response(json.dumps(corrected))],
    )
    monkeypatch.setenv("ARK_API_KEY", "never-print-this")

    result = asyncio.run(
        ModelAdapter(
            model="configured-model", base_url="https://configured.invalid"
        ).decide([{"role": "user", "content": "decide"}])
    )

    assert result.decision.model_dump()["operations"][0][
        "occurrence_evidence_refs"
    ] == ["evidence-1"]
    repair_instruction = completions.calls[1]["messages"][-1]["content"]
    assert "provide a causal finding, occurrence recommendation" in repair_instruction
    assert "exactly three mutually exclusive modes" in repair_instruction
    assert "an occurrence recommendation provides one or more" in repair_instruction
    assert "occurrence_evidence_refs" in repair_instruction
    assert "fields for the other two modes empty or omitted" in repair_instruction


def test_repair_feedback_preserves_plain_value_error_message() -> None:
    feedback = json.loads(
        model_module._validation_feedback(  # pyright: ignore[reportPrivateUsage]
            ValueError("dynamic contract requires one disposition")
        )
    )

    assert feedback == [
        {
            "path": [],
            "message": "dynamic contract requires one disposition",
        }
    ]


def test_repair_bounds_oversized_invalid_output(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    invalid = (
        '{"decision_summary":"'
        + ("x" * (model_module.MAX_DECISION_REPAIR_OUTPUT_CHARS * 4))
        + '","intent":"Investigate","operations":[]}'
    )
    completions, _init_kwargs = _install_fake_ark(
        monkeypatch,
        [
            _Response(invalid),
            _Response(json.dumps(_decision("bounded repair"))),
        ],
    )
    monkeypatch.setenv("ARK_API_KEY", "never-print-this")

    result = asyncio.run(
        ModelAdapter(
            model="configured-model", base_url="https://configured.invalid"
        ).decide([{"role": "user", "content": "decide"}])
    )

    assert result.decision.decision_summary == "bounded repair"
    excerpt = completions.calls[1]["messages"][-2]["content"]
    assert excerpt != invalid
    assert len(excerpt) <= model_module.MAX_DECISION_REPAIR_OUTPUT_CHARS
    assert "invalid output truncated" in excerpt
    assert excerpt.startswith('{"decision_summary":"')
    assert excerpt.endswith('","intent":"Investigate","operations":[]}')


def test_structured_repair_does_not_repeat_decision_schema(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    completions, _init_kwargs = _install_fake_ark(
        monkeypatch,
        [
            _Response("not-json"),
            _Response(json.dumps(_decision("schema stays structured"))),
        ],
    )
    monkeypatch.setenv("ARK_API_KEY", "never-print-this")

    asyncio.run(
        ModelAdapter(
            model="configured-model", base_url="https://configured.invalid"
        ).decide([{"role": "user", "content": "decide"}])
    )

    repair_call = completions.calls[1]
    assert repair_call["response_format"]["json_schema"]["strict"] is True
    repair_instruction = repair_call["messages"][-1]["content"]
    decision_schema = json.dumps(
        repair_call["response_format"]["json_schema"]["schema"]
    )
    assert "Schema:" not in repair_instruction
    assert decision_schema not in repair_instruction


def test_model_adapter_repairs_dynamic_nonempty_disposition_constraint(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    empty: dict[str, Any] = {
        "decision_summary": "filled the anomaly slot",
        "reflection": {
            "based_on_observation_ids": ["observation-1"],
            "anomaly_disposition_update": [],
            "next_check": None,
            "can_finalize": False,
        },
        "operations": [{"type": "advance_frontier"}],
    }
    corrected: dict[str, Any] = {
        **empty,
        "reflection": {
            **empty["reflection"],
            "anomaly_disposition_update": [
                {
                    "anomaly_id": "anomaly-1",
                    "disposition": "background",
                    "evidence_refs": ["evidence-1"],
                    "explanation": "the typed negative profile excludes this event",
                }
            ],
        },
    }
    completions, _init_kwargs = _install_fake_ark(
        monkeypatch,
        [_Response(json.dumps(empty)), _Response(json.dumps(corrected))],
    )
    monkeypatch.setenv("ARK_API_KEY", "never-print-this")
    adapter = ModelAdapter(
        model="configured-model",
        base_url="https://configured.invalid",
        timeout_s=2,
    )
    messages = [
        {
            "role": "user",
            "content": json.dumps(
                {
                    "frontier_control": {
                        "response_schema_constraints": {
                            "anomaly_disposition_update_min_items": 1
                        }
                    }
                }
            ),
        }
    ]

    result = asyncio.run(adapter.decide(messages))

    assert result.decision.reflection is not None
    assert len(result.decision.reflection.anomaly_disposition_update) == 1
    assert len(completions.calls) == 2
    assert adapter.last_requests[0]["status"] == "invalid_response"
    assert adapter.last_requests[0]["retry_reason"] == "invalid_decision"
    schema = completions.calls[0]["response_format"]["json_schema"]["schema"]
    assert (
        schema["$defs"]["Reflection"]["properties"]["anomaly_disposition_update"][
            "minItems"
        ]
        == 1
    )


def test_usage_normalization_preserves_provider_top_level_values(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    usage = {
        "prompt_tokens": 3,
        "cache_read_input_tokens": 99,
        "prompt_tokens_details": {"cached_tokens": 4},
        "completion_tokens_details": {"reasoning_tokens": -1},
    }
    _install_fake_ark(
        monkeypatch,
        [_Response(json.dumps(_decision("normalized")), usage=usage)],
    )
    monkeypatch.setenv("ARK_API_KEY", "never-print-this")

    result = asyncio.run(
        ModelAdapter(
            model="configured-model", base_url="https://configured.invalid"
        ).decide([{"role": "user", "content": "decide"}])
    )

    assert result.usage["cache_read_input_tokens"] == 99
    assert "reasoning_tokens" not in result.usage
    assert result.raw_response["usage"] == usage


def test_unsupported_json_schema_400_falls_back_once(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    unsupported = _sdk_error(
        "ArkBadRequestError",
        "This model does not support response_format json_schema",
        status=400,
    )
    completions, _ = _install_fake_ark(
        monkeypatch,
        [unsupported, _Response(json.dumps(_decision("fallback")))],
    )
    monkeypatch.setenv("ARK_API_KEY", "never-print-this")

    result = asyncio.run(
        ModelAdapter(
            model="configured-model", base_url="https://configured.invalid"
        ).decide([{"role": "user", "content": "decide"}])
    )

    assert result.decision.decision_summary == "fallback"
    assert result.usage == {"prompt_tokens": 3, "completion_tokens": 5}
    assert "response_format" in completions.calls[0]
    assert "response_format" not in completions.calls[1]
    assert "decision_summary" in completions.calls[1]["messages"][-1]["content"]


def test_invalid_responses_are_kept_for_run_audit(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _install_fake_ark(
        monkeypatch,
        [_Response("bad-one"), _Response("bad-two"), _Response("bad-three")],
    )
    monkeypatch.setenv("ARK_API_KEY", "never-print-this")
    adapter = ModelAdapter(
        model="configured-model", base_url="https://configured.invalid"
    )

    with pytest.raises(ModelResponseError) as error:
        asyncio.run(adapter.decide([{"role": "user", "content": "decide"}]))

    assert len(error.value.responses) == 3
    assert error.value.responses[0]["choices"][0]["message"]["content"] == ("bad-one")
    assert error.value.responses[0]["usage"] == {
        "prompt_tokens": 3,
        "completion_tokens": 5,
    }
    assert error.value.responses[1]["choices"][0]["message"]["content"] == "bad-two"
    assert error.value.responses[2]["choices"][0]["message"]["content"] == ("bad-three")
    assert "bad-one" not in str(error.value)
    assert "bad-two" not in str(error.value)


def test_other_400_does_not_fall_back_and_hides_sensitive_data(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    error = _sdk_error("ArkBadRequestError", "private request body", status=400)
    completions, _ = _install_fake_ark(monkeypatch, [error])
    monkeypatch.setenv("ARK_API_KEY", "never-print-this")
    adapter = ModelAdapter(
        model="configured-model", base_url="https://configured.invalid"
    )

    with pytest.raises(ModelProviderError) as caught:
        asyncio.run(adapter.decide([{"role": "user", "content": "decide"}]))

    message = str(caught.value)
    assert "type=ArkBadRequestError" in message
    assert "status=400" in message
    assert "request_id=req-safe-123" in message
    assert "never-print-this" not in message
    assert "private request body" not in message
    assert len(completions.calls) == 1


@pytest.mark.parametrize(
    ("name", "status"),
    [
        ("ArkAuthenticationError", 401),
        ("ArkRateLimitError", 429),
        ("ArkAPIConnectionError", None),
        ("ArkInternalServerError", 503),
    ],
)
def test_provider_errors_are_safely_normalized(
    monkeypatch: pytest.MonkeyPatch,
    name: str,
    status: int | None,
) -> None:
    error = _sdk_error(name, "never-print-this private request body", status=status)
    _install_fake_ark(monkeypatch, [error])
    monkeypatch.setenv("ARK_API_KEY", "never-print-this")
    adapter = ModelAdapter(
        model="configured-model", base_url="https://configured.invalid"
    )

    with pytest.raises(ModelProviderError) as caught:
        asyncio.run(adapter.decide([{"role": "user", "content": "decide"}]))

    message = str(caught.value)
    assert f"type={name}" in message
    assert "request_id=req-safe-123" in message
    assert (f"status={status}" in message) is (status is not None)
    assert "never-print-this" not in message
    assert "private request body" not in message


def test_sdk_timeout_is_recognized_through_inheritance(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    ark_timeout = type("ArkAPITimeoutError", (Exception,), {})
    wrapped_timeout = type("WrappedTimeout", (ark_timeout,), {})
    error = wrapped_timeout("never-print-this private request body")
    error.request_id = "req-timeout"  # type: ignore[attr-defined]
    _install_fake_ark(monkeypatch, [error])
    monkeypatch.setenv("ARK_API_KEY", "never-print-this")
    adapter = ModelAdapter(
        model="configured-model", base_url="https://configured.invalid"
    )

    with pytest.raises(ModelTimeoutError) as caught:
        asyncio.run(adapter.decide([{"role": "user", "content": "decide"}]))

    message = str(caught.value)
    assert "type=WrappedTimeout" in message
    assert "request_id=req-timeout" in message
    assert "never-print-this" not in message
    assert "private request body" not in message


def test_recorded_adapter_is_ordered_and_validated() -> None:
    adapter = RecordedModelAdapter(
        [json.dumps(_decision("first")), _decision("second")]
    )

    first = asyncio.run(adapter.decide([]))
    second = asyncio.run(adapter.decide([]))

    assert first.decision.decision_summary == "first"
    assert second.decision.decision_summary == "second"
    assert first.provider_model == "recorded"
    assert first.usage["completion_tokens"] > 0
    assert first.requests[0]["recorded_cursor"] == 1
    assert first.requests[0]["recorded_count"] == 2
    assert second.requests[0]["recorded_cursor"] == 2
    with pytest.raises(ModelResponseError, match="No recorded"):
        asyncio.run(adapter.decide([]))


def test_model_adapter_does_not_swallow_cancellation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _install_fake_ark(monkeypatch, [asyncio.CancelledError()])
    monkeypatch.setenv("ARK_API_KEY", "never-print-this")
    adapter = ModelAdapter(
        model="configured-model",
        base_url="https://configured.invalid",
    )

    with pytest.raises(asyncio.CancelledError):
        asyncio.run(adapter.decide([{"role": "user", "content": "decide"}]))


def test_recorded_adapter_rejects_invalid_decision() -> None:
    adapter = RecordedModelAdapter([{"unexpected": True}])

    with pytest.raises(ModelResponseError, match="not a valid Decision"):
        asyncio.run(adapter.decide([]))
