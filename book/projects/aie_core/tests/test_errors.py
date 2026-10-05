# path: book/projects/aie_core/tests/test_errors.py
from aie_core.llm.errors import (
    InvalidRequestError,
    LLMError,
    MalformedResponseError,
    ProviderUnavailableError,
    RateLimitError,
    TimeoutError,
    map_http_error,
)


def test_default_retryability():
    assert RateLimitError("x").retryable and TimeoutError("x").retryable and ProviderUnavailableError("x").retryable
    assert not InvalidRequestError("x").retryable and not MalformedResponseError("x").retryable
    assert RateLimitError("x", retryable=False).retryable is False


def test_all_errors_share_base_and_fields():
    err = RateLimitError("limited", retry_after_s=3, status_code=429, provider="p")
    assert isinstance(err, LLMError) and err.retry_after_s == 3
    assert "status=429" in str(err) and "retry_after=3s" in str(err)
    assert LLMError("plain").retry_after_s is None


def test_map_http_error_reads_retry_after_header_case_insensitively():
    err = map_http_error(429, {"error": {"message": "m"}}, {"Retry-After": "12"}, "p")
    assert isinstance(err, RateLimitError) and err.retry_after_s == 12.0
    err = map_http_error(429, {}, {"retry-after": "not-a-number"}, "p")
    assert err.retry_after_s is None


def test_map_http_error_extracts_messages_from_shapes():
    assert "unknown_model" in str(map_http_error(404, {"error": {"code": "unknown_model", "message": "no"}}, {}, "p"))
    assert "plain text" in str(map_http_error(502, "plain text", {}, "p"))
    assert "HTTP 500" in str(map_http_error(500, None, {}, "p"))
