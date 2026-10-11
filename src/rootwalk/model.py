"""Small model-provider boundary with deterministic offline replacement."""

from __future__ import annotations

import asyncio
import json
import os
import re
import time
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass
from importlib import import_module
from typing import Any, Literal, cast

from pydantic import ValidationError

from .contracts import Decision

Message = Mapping[str, Any]
RecordedJSON = str | Mapping[str, Any]
MAX_DECISION_REPAIR_ATTEMPTS = 2
MAX_DECISION_REPAIR_OUTPUT_CHARS = 4_000
_REDACTED = "[REDACTED]"
_SENSITIVE_KEYS = frozenset(
    {
        "api_key",
        "apikey",
        "authorization",
        "access_token",
        "refresh_token",
        "id_token",
        "token",
        "secret",
        "client_secret",
        "cookie",
        "cookies",
        "set_cookie",
        "password",
    }
)


def _anomaly_disposition_min_items(messages: Sequence[Message]) -> int:
    """Read one allowlisted constraint from the Runtime-owned context message."""
    for message in reversed(messages):
        if message.get("role") != "user" or not isinstance(message.get("content"), str):
            continue
        try:
            payload: object = json.loads(cast(str, message["content"]))
        except json.JSONDecodeError:
            break
        if not isinstance(payload, Mapping):
            break
        frontier = cast(Mapping[str, object], payload).get("frontier_control")
        constraints = (
            cast(Mapping[str, object], frontier).get("response_schema_constraints")
            if isinstance(frontier, Mapping)
            else None
        )
        raw_min_items = (
            cast(Mapping[str, object], constraints).get(
                "anomaly_disposition_update_min_items"
            )
            if isinstance(constraints, Mapping)
            else None
        )
        if isinstance(raw_min_items, int) and not isinstance(raw_min_items, bool):
            return 1 if raw_min_items == 1 else 0
        break
    return 0


def _decision_json_schema(messages: Sequence[Message]) -> dict[str, Any]:
    """Apply allowlisted Runtime-owned response constraints to a fresh schema."""
    schema = Decision.model_json_schema()
    min_items = _anomaly_disposition_min_items(messages)
    if min_items:
        definitions = schema.get("$defs")
        reflection = (
            cast(Mapping[str, object], definitions).get("Reflection")
            if isinstance(definitions, Mapping)
            else None
        )
        properties = (
            cast(Mapping[str, object], reflection).get("properties")
            if isinstance(reflection, Mapping)
            else None
        )
        disposition = (
            cast(Mapping[str, object], properties).get("anomaly_disposition_update")
            if isinstance(properties, Mapping)
            else None
        )
        if not isinstance(disposition, dict):
            raise ModelConfigurationError(
                "Decision schema lacks anomaly_disposition_update"
            )
        disposition["minItems"] = min_items
    return schema


def redact_sensitive(
    value: object, known_secrets: Iterable[str] | None = None
) -> object:
    """Recursively redact credential fields while preserving ordinary content."""
    secrets = tuple(
        secret
        for secret in (
            known_secrets
            if known_secrets is not None
            else (os.environ.get("ARK_API_KEY", ""),)
        )
        if secret
    )
    if isinstance(value, Mapping):
        return {
            key: (
                _REDACTED
                if isinstance(key, str) and _is_sensitive_key(key)
                else redact_sensitive(nested, secrets)
            )
            for key, nested in cast(Mapping[object, object], value).items()
        }
    if isinstance(value, (list, tuple)):
        return [
            redact_sensitive(item, secrets) for item in cast(Sequence[object], value)
        ]
    if isinstance(value, str):
        return _redact_string(value, secrets)
    return value


def _is_sensitive_key(key: str) -> bool:
    normalized = re.sub(r"[^a-z0-9]+", "_", key.casefold()).strip("_")
    return normalized in _SENSITIVE_KEYS or normalized.endswith(
        ("_api_key", "_authorization", "_token", "_secret", "_cookie")
    )


def _redact_string(value: str, secrets: Sequence[str]) -> str:
    stripped = value.lstrip()
    if stripped.startswith(("{", "[")):
        try:
            parsed: object = json.loads(value)
        except json.JSONDecodeError:
            pass
        else:
            if isinstance(parsed, (dict, list)):
                return json.dumps(
                    redact_sensitive(cast(object, parsed), secrets),
                    allow_nan=False,
                    ensure_ascii=False,
                    separators=(",", ":"),
                    sort_keys=True,
                )
    result = value
    for secret in secrets:
        result = result.replace(secret, _REDACTED)
    result = re.sub(
        r"(?i)(\bauthorization\s*[:=]\s*)(?:bearer\s+)?[^\s,;]+",
        rf"\1{_REDACTED}",
        result,
    )
    result = re.sub(
        r"(?i)(\bbearer\s+)[^\s,;]+",
        rf"\1{_REDACTED}",
        result,
    )
    return re.sub(
        r"(?i)([?&](?:[a-z0-9_.-]+[_-])?(?:api[_-]?key|authorization|token|"
        r"secret|cookie|password)=)[^&#\s]+",
        rf"\1{_REDACTED}",
        result,
    )


class ModelAdapterError(RuntimeError):
    """Base error raised by model adapters."""


class ModelConfigurationError(ModelAdapterError):
    """The live model adapter is not configured."""


class ModelResponseError(ModelAdapterError):
    """The model did not return a valid Decision."""

    def __init__(
        self, message: str, *, responses: Iterable[Mapping[str, Any]] = ()
    ) -> None:
        super().__init__(message)
        self.responses = [dict(response) for response in responses]


class ModelTimeoutError(ModelAdapterError):
    """The bounded model call timed out."""


class ModelProviderError(ModelAdapterError):
    """A model provider failed without exposing request or response bodies."""


@dataclass(frozen=True)
class ModelCallResult:
    decision: Decision
    usage: dict[str, Any]
    provider_model: str
    raw_response: dict[str, Any]
    requests: tuple[dict[str, Any], ...] = ()


class ModelAdapter:
    """Async Ark adapter. Model and endpoint must come from the caller."""

    def __init__(
        self,
        *,
        model: str,
        base_url: str,
        timeout_s: float = 12.0,
        max_output_tokens: int = 2_000,
        reasoning_effort: Literal["minimal", "low", "medium", "high", "xhigh", "max"]
        | None = None,
        _api_key_env: str = "ARK_API_KEY",
        _provider_name: str = "Ark",
        _use_ark_sdk: bool = True,
    ) -> None:
        if not model.strip():
            raise ModelConfigurationError(
                f"{_provider_name} model must be provided by configuration"
            )
        if not base_url.strip():
            raise ModelConfigurationError(
                f"{_provider_name} base URL must be provided by configuration"
            )
        if timeout_s <= 0:
            raise ModelConfigurationError(
                f"{_provider_name} timeout must be greater than zero"
            )
        if max_output_tokens < 1:
            raise ModelConfigurationError(
                f"{_provider_name} max output tokens must be greater than zero"
            )

        api_key = os.environ.get(_api_key_env)
        if not api_key:
            raise ModelConfigurationError(
                f"{_api_key_env} environment variable is required"
            )
        self._model = model
        self._provider_name = _provider_name
        self._base_url = base_url.rstrip("/")
        self._api_key = api_key
        self._timeout_s = timeout_s
        self._max_output_tokens = max_output_tokens
        self._reasoning_effort = reasoning_effort
        self.last_requests: list[dict[str, Any]] = []
        self._attempt_hook: (
            Callable[[Literal["started", "finished"], Mapping[str, Any]], None] | None
        ) = None
        self._client: Any | None = None
        if _use_ark_sdk:
            try:
                async_ark = import_module("volcenginesdkarkruntime").AsyncArk
            except (ImportError, AttributeError) as exc:
                raise ModelConfigurationError(
                    "Ark SDK is unavailable; install volcengine-python-sdk[ark]"
                ) from exc
            self._client = async_ark(
                api_key=api_key,
                base_url=base_url,
                timeout=timeout_s,
            )

    async def _create_completion(self, request: Mapping[str, Any]) -> Any:
        if self._client is None:
            raise RuntimeError("model client is unavailable")
        return await self._client.chat.completions.create(**dict(request))

    def set_attempt_hook(
        self,
        hook: Callable[[Literal["started", "finished"], Mapping[str, Any]], None]
        | None,
    ) -> None:
        self._attempt_hook = hook

    def _notify_attempt(
        self, event: Literal["started", "finished"], attempt: Mapping[str, Any]
    ) -> None:
        if self._attempt_hook is None:
            return
        try:
            self._attempt_hook(event, attempt)
        except Exception as exc:
            if event == "started":
                raise
            if isinstance(attempt, dict):
                attempt["audit_error_type"] = type(exc).__name__

    async def decide(self, messages: Sequence[Message]) -> ModelCallResult:
        """Keep every provider attempt, including repairs and interrupted requests."""
        self.last_requests = []
        try:
            async with asyncio.timeout(self._timeout_s):
                return await self._decide(messages)
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            normalized = _normalize_provider_error(
                exc, self._timeout_s, self._provider_name
            )
            if normalized is not None:
                raise normalized from None
            raise

    async def _decide(self, messages: Sequence[Message]) -> ModelCallResult:
        request_messages = [dict(message) for message in messages]
        schema = _decision_json_schema(request_messages)
        required_disposition_items = _anomaly_disposition_min_items(request_messages)
        response_format = {
            "type": "json_schema",
            "json_schema": {
                "name": "rootwalk_decision",
                "strict": True,
                "schema": schema,
            },
        }

        structured_output = True
        repair_attempts = 0
        failed_responses: list[dict[str, Any]] = []
        total_usage: dict[str, Any] = {}
        while True:
            request = {
                "model": self._model,
                "messages": [dict(message) for message in request_messages],
                "max_tokens": self._max_output_tokens,
                "reasoning_effort": self._reasoning_effort,
                **({"response_format": response_format} if structured_output else {}),
            }
            safe_request = cast(dict[str, Any], redact_sensitive(request))
            attempt: dict[str, Any] = {
                "request_index": len(self.last_requests) + 1,
                "started_at": time.time(),
                "status": "started",
                "request": safe_request,
            }
            self._notify_attempt("started", attempt)
            self.last_requests.append(attempt)
            started = time.perf_counter()
            try:
                response = await self._create_completion(safe_request)
            except asyncio.CancelledError:
                attempt.update(status="cancelled", error_type="CancelledError")
                attempt["ended_at"] = time.time()
                attempt["duration_ms"] = (time.perf_counter() - started) * 1_000
                self._notify_attempt("finished", attempt)
                raise
            except Exception as exc:
                attempt.update(
                    status="error",
                    error_type=type(exc).__name__,
                    error_summary=_safe_error_details(exc),
                )
                attempt["ended_at"] = time.time()
                attempt["duration_ms"] = (time.perf_counter() - started) * 1_000
                if structured_output and _is_unsupported_schema_error(exc):
                    attempt["retry_reason"] = "unsupported_response_format"
                    self._notify_attempt("finished", attempt)
                    structured_output = False
                    request_messages.append(
                        {
                            "role": "user",
                            "content": (
                                "Return only a JSON object matching this Decision "
                                f"schema: {json.dumps(schema)}"
                            ),
                        }
                    )
                    continue
                self._notify_attempt("finished", attempt)
                raise
            raw_response = _model_dump(response)
            audit_response = cast(dict[str, Any], redact_sensitive(raw_response))
            attempt["ended_at"] = time.time()
            attempt["duration_ms"] = (time.perf_counter() - started) * 1_000
            attempt["response"] = audit_response
            attempt["usage"] = aggregate_model_usage([raw_response])
            attempt["provider_model"] = str(raw_response.get("model") or self._model)
            attempt["status"] = "success"
            _merge_usage(total_usage, raw_response.get("usage"))
            content = _response_content(response)
            try:
                decision = Decision.model_validate_json(content)
                if required_disposition_items and (
                    decision.reflection is None
                    or len(decision.reflection.anomaly_disposition_update)
                    < required_disposition_items
                ):
                    raise ValueError(
                        "Decision.reflection.anomaly_disposition_update must contain "
                        f"at least {required_disposition_items} item(s) for this "
                        "Runtime state; decision_summary prose does not satisfy the "
                        "dynamic response contract"
                    )
            except ValueError as exc:
                attempt["status"] = "invalid_response"
                attempt["error_type"] = type(exc).__name__
                attempt["retry_reason"] = "invalid_decision"
                failed_responses.append(audit_response)
                if repair_attempts >= MAX_DECISION_REPAIR_ATTEMPTS:
                    self._notify_attempt("finished", attempt)
                    raise ModelResponseError(
                        f"{self._provider_name} returned an invalid Decision after "
                        "two repair attempts",
                        responses=failed_responses,
                    ) from exc
                repair_attempts += 1
                request_messages.extend(
                    (
                        {
                            "role": "assistant",
                            "content": _repair_output_excerpt(content),
                        },
                        {
                            "role": "user",
                            "content": (
                                "Return only corrected JSON. Validation errors: "
                                f"{_validation_feedback(exc)}. Only independent query "
                                "operations may be batched; update_hypothesis, "
                                "submit_path, or finalize must be the only operation. "
                                "Finalize has exactly three mutually exclusive modes: "
                                "a causal finding selects one or more "
                                "hypothesis_ids/path_ids; an occurrence recommendation "
                                "provides one or more occurrence_evidence_refs; or an "
                                "abstention provides a non-empty abstain_reason. Keep "
                                "the fields for the other two modes empty or omitted. "
                                "Never combine a state update with finalize."
                            ),
                        },
                    )
                )
                self._notify_attempt("finished", attempt)
                continue

            _normalize_usage(total_usage)
            self._notify_attempt("finished", attempt)
            return ModelCallResult(
                decision=decision,
                usage=total_usage,
                provider_model=str(raw_response.get("model") or self._model),
                raw_response=audit_response,
                requests=tuple(self.last_requests),
            )


# Explicit provider name for callers that prefer it; this is the same implementation.
ArkModelAdapter = ModelAdapter


class RecordedModelAdapter:
    """Return pre-recorded JSON Decisions in order without network access."""

    def __init__(
        self,
        records: Iterable[RecordedJSON],
        *,
        provider_model: str = "recorded",
    ) -> None:
        self._records = tuple(records)
        self._cursor = 0
        self._provider_model = provider_model
        self.last_requests: list[dict[str, Any]] = []

    async def decide(self, messages: Sequence[Message]) -> ModelCallResult:
        self.last_requests = []
        if self._cursor >= len(self._records):
            raise ModelResponseError("No recorded model responses remain")
        record = self._records[self._cursor]
        self._cursor += 1

        try:
            if isinstance(record, str):
                decision = Decision.model_validate_json(record)
                raw_response = cast(dict[str, Any], json.loads(record))
            else:
                raw_response = dict(record)
                decision = Decision.model_validate(raw_response)
        except (TypeError, ValueError) as exc:
            raise ModelResponseError(
                "Recorded response is not a valid Decision"
            ) from exc

        usage = {
            "prompt_tokens": _estimated_json_tokens(list(messages)),
            "completion_tokens": _estimated_json_tokens(raw_response),
        }
        self.last_requests = [
            {
                "request_index": 1,
                "status": "success",
                "provider_model": self._provider_model,
                "recorded_cursor": self._cursor,
                "recorded_count": len(self._records),
                "usage": usage,
                "request": {"messages": [dict(message) for message in messages]},
                "response": raw_response,
            }
        ]
        return ModelCallResult(
            decision=decision,
            usage=usage,
            provider_model=self._provider_model,
            raw_response=raw_response,
            requests=tuple(self.last_requests),
        )


def _estimated_json_tokens(value: object) -> int:
    encoded = json.dumps(
        value,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode()
    return (len(encoded) + 2) // 3


def _response_content(response: Any) -> str:
    if isinstance(response, Mapping):
        choices = cast(Mapping[str, Any], response).get("choices")
        if not isinstance(choices, Sequence) or isinstance(choices, (str, bytes)):
            return ""
        choice_items = cast(Sequence[object], choices)
        first: object | None = choice_items[0] if choice_items else None
        if not isinstance(first, Mapping):
            return ""
        message = cast(Mapping[str, Any], first).get("message")
        if not isinstance(message, Mapping):
            return ""
        content = cast(Mapping[str, Any], message).get("content")
        return content if isinstance(content, str) else ""
    try:
        content = response.choices[0].message.content
    except (AttributeError, IndexError, TypeError):
        return ""
    return content if isinstance(content, str) else ""


def _validation_feedback(exc: ValueError) -> str:
    errors: list[dict[str, object]]
    if isinstance(exc, ValidationError):
        errors = [
            {"path": list(error["loc"]), "message": error["msg"]}
            for error in exc.errors(include_input=False, include_url=False)
        ]
    else:
        message = str(exc).strip() or type(exc).__name__
        errors = [{"path": [], "message": message}]
    return json.dumps(
        errors,
        ensure_ascii=False,
        separators=(",", ":"),
    )


def _repair_output_excerpt(content: str) -> str:
    """Keep enough invalid output for repair without replaying an oversized body."""
    if len(content) <= MAX_DECISION_REPAIR_OUTPUT_CHARS:
        return content
    marker = (
        f"\n... [invalid output truncated; {len(content):,} total characters] ...\n"
    )
    remaining = MAX_DECISION_REPAIR_OUTPUT_CHARS - len(marker)
    head_chars = (remaining + 1) // 2
    tail_chars = remaining - head_chars
    return f"{content[:head_chars]}{marker}{content[-tail_chars:]}"


def _model_dump(response: Any) -> dict[str, Any]:
    if isinstance(response, Mapping):
        return dict(cast(Mapping[str, Any], response))
    dump = getattr(response, "model_dump", None)
    if callable(dump):
        value = dump(mode="json")
        if isinstance(value, Mapping):
            return dict(cast(Mapping[str, Any], value))
    return {}


def aggregate_model_usage(responses: Iterable[Mapping[str, Any]]) -> dict[str, Any]:
    """Sum provider-reported usage without treating missing usage as zero."""
    total: dict[str, Any] = {}
    for response in responses:
        _merge_usage(total, response.get("usage"))
    _normalize_usage(total)
    return total


def _merge_usage(total: dict[str, Any], raw_usage: object) -> None:
    if not isinstance(raw_usage, Mapping):
        return
    for key, value in cast(Mapping[str, Any], raw_usage).items():
        current = total.get(key)
        if (
            isinstance(value, (int, float))
            and not isinstance(value, bool)
            and isinstance(current, (int, float))
            and not isinstance(current, bool)
        ):
            total[key] = current + value
        elif isinstance(value, Mapping):
            nested = (
                dict(cast(Mapping[str, Any], current))
                if isinstance(current, Mapping)
                else {}
            )
            _merge_usage(nested, cast(Mapping[str, Any], value))
            total[key] = nested
        else:
            total[key] = value


def _normalize_usage(usage: dict[str, Any]) -> None:
    for detail_key, nested_key, top_level_key in (
        ("prompt_tokens_details", "cached_tokens", "cache_read_input_tokens"),
        ("completion_tokens_details", "reasoning_tokens", "reasoning_tokens"),
    ):
        if top_level_key in usage:
            continue
        details = usage.get(detail_key)
        value: object = (
            cast(Mapping[str, Any], details).get(nested_key)
            if isinstance(details, Mapping)
            else None
        )
        if (
            isinstance(value, (int, float))
            and not isinstance(value, bool)
            and value >= 0
        ):
            usage[top_level_key] = value


def _is_unsupported_schema_error(exc: Exception) -> bool:
    detail = _provider_error_text(exc).lower()
    mentions_schema = any(
        name in detail for name in ("response_format", "response format", "json_schema")
    )
    says_unsupported = any(
        phrase in detail
        for phrase in ("unsupported", "not support", "unavailable", "not available")
    )
    return _status_code(exc) == 400 and mentions_schema and says_unsupported


def _provider_error_text(exc: Exception) -> str:
    values: list[Any] = [
        getattr(exc, "message", ""),
        getattr(exc, "code", ""),
        getattr(exc, "param", ""),
    ]
    body = getattr(exc, "body", None)
    if isinstance(body, str):
        values.append(body)
    elif isinstance(body, Mapping):
        body_map = cast(Mapping[str, Any], body)
        values.extend(body_map.get(key, "") for key in ("message", "code", "param"))
        error = body_map.get("error")
        if isinstance(error, Mapping):
            error_map = cast(Mapping[str, Any], error)
            values.extend(
                error_map.get(key, "") for key in ("message", "code", "param")
            )
    return " ".join(str(value) for value in values)


def _normalize_provider_error(
    exc: Exception, timeout_s: float, provider_name: str = "Ark"
) -> ModelAdapterError | None:
    names = {kind.__name__ for kind in type(exc).__mro__}
    timeout_names = {
        "ArkAPITimeoutError",
        "ConnectTimeout",
        "PoolTimeout",
        "ReadTimeout",
        "TimeoutException",
        "WriteTimeout",
    }
    details = _safe_error_details(exc)
    if isinstance(exc, TimeoutError) or names & timeout_names:
        return ModelTimeoutError(
            f"{provider_name} model timeout ({details}; timeout_s={timeout_s:g})"
        )

    provider_names = {
        "ArkError",
        "HTTPError",
        "RequestError",
        "ConnectionError",
    }
    is_ark_error = any(
        name.startswith("Ark") and name.endswith("Error") for name in names
    )
    if _status_code(exc) is not None or names & provider_names or is_ark_error:
        return ModelProviderError(f"{provider_name} provider error ({details})")
    return None


def _safe_error_details(exc: Exception) -> str:
    details = [f"type={type(exc).__name__}"]
    status = _status_code(exc)
    if status is not None:
        details.append(f"status={status}")
    request_id = getattr(exc, "request_id", None)
    if isinstance(request_id, str) and request_id:
        redacted = cast(str, redact_sensitive(request_id))
        safe_request_id = redacted.replace("\r", " ").replace("\n", " ")[:128]
        details.append(f"request_id={safe_request_id}")
    return "; ".join(details)


def safe_exception_summary(exc: BaseException) -> str:
    """Return stable exception metadata without provider bodies or credentials."""
    if isinstance(exc, Exception):
        return _safe_error_details(exc)
    return f"type={type(exc).__name__}"


def _status_code(exc: Exception) -> int | None:
    status = getattr(exc, "status_code", None)
    if not isinstance(status, int):
        status = getattr(getattr(exc, "response", None), "status_code", None)
    return status if isinstance(status, int) else None
