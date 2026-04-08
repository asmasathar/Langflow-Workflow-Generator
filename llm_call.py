import requests
import os
import re
import sys
import time
import json
import hashlib
from typing import Any


_REQ_CACHE: dict[str, tuple[float, str]] = {}
_LLM_CALL_COUNT = 0


def _bool_env(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def _int_env(name: str, default: int) -> int:
    raw = os.getenv(name)
    if raw is None:
        return default
    try:
        return int(raw)
    except Exception:
        return default


def _optional_timeout(name: str, default: int) -> int | None:
    """
    Read timeout from env.
    - missing/invalid -> default
    - 0 or negative -> no timeout (None)
    """
    raw = os.getenv(name)
    if raw is None:
        return default
    try:
        val = int(raw)
    except Exception:
        return default
    if val <= 0:
        return None
    return val

def _normalize_chat_completions_url(raw_url: str) -> str:
    url = (raw_url or "").strip().rstrip("/")
    if not url:
        return "http://127.0.0.1:1234/v1/chat/completions"
    if url.endswith("/chat/completions"):
        return url
    if url.endswith("/v1"):
        return f"{url}/chat/completions"
    if "://" in url and url.count("/") <= 2:
        return f"{url}/v1/chat/completions"
    return url


def _get_local_llm_settings() -> tuple[str, str]:
    api_url = os.getenv(
        "LOCAL_LLM_API_URL",
        os.getenv("OPENAI_BASE_URL", "http://127.0.0.1:1234/v1/chat/completions"),
    )
    model = os.getenv("LOCAL_LLM_MODEL", "qwen/qwen3-vl-8b")
    return _normalize_chat_completions_url(api_url), model


def get_langflow_headers() -> dict:
    headers = {"Content-Type": "application/json"}
    api_key = os.getenv("LANGFLOW_API_KEY")
    if api_key:
        headers["x-api-key"] = api_key
    return headers


def call_langflow_api(method: str, url: str, *, json_body: dict | None = None, timeout: int = 30):
    response = requests.request(
        method=method.upper(),
        url=url,
        headers=get_langflow_headers(),
        json=json_body,
        timeout=timeout,
    )
    response.raise_for_status()
    return response


def call_local_llm(
    messages: list[dict] | str,
    temperature: float = 0.1,
    max_tokens: int | None = None,
    timeout: int | None = None,
) -> str:
    """Send chat messages to the local LLM server and return the response text.

    Args:
        messages: Chat history in OpenAI chat-completions format, or a plain prompt string
        temperature: Model temperature (0.0-1.0)
    """
    if isinstance(messages, str):
        messages = [{"role": "user", "content": messages}]

    try:
        global _LLM_CALL_COUNT
        llm_api_url, llm_model = _get_local_llm_settings()
        show_think = _bool_env("LOCAL_LLM_SHOW_THINK", False)
        resolved_max_tokens = max_tokens if isinstance(max_tokens, int) and max_tokens > 0 else _int_env("LOCAL_LLM_MAX_TOKENS", 2048)
        resolved_timeout = timeout if timeout is not None else _optional_timeout("LOCAL_LLM_TIMEOUT", 90)
        dedupe_window = _int_env("LOCAL_LLM_DEDUPE_WINDOW_SEC", 20)
        log_calls = _bool_env("LOCAL_LLM_LOG_CALLS", True)
        max_calls = _int_env("LOCAL_LLM_MAX_CALLS", 1000)

        if max_calls > 0 and _LLM_CALL_COUNT >= max_calls:
            if log_calls:
                print(f"[LLM SKIP] call budget exceeded ({max_calls})")
            return ""

        # Short-lived dedupe cache prevents repeated identical calls from
        # printing/processing multiple times across planning-refine loops.
        payload_key = {
            "model": llm_model,
            "temperature": temperature,
            "max_tokens": resolved_max_tokens,
            "messages": messages,
        }
        sig = hashlib.sha256(json.dumps(payload_key, sort_keys=True, ensure_ascii=False).encode("utf-8")).hexdigest()
        now = time.time()
        cached = _REQ_CACHE.get(sig)
        if cached and (now - cached[0]) <= max(dedupe_window, 0):
            if log_calls:
                print("[LLM CACHE HIT] Reusing recent identical response")
            return cached[1]

        payload = {
            "model": llm_model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": resolved_max_tokens,
        }

        started = time.time()
        _LLM_CALL_COUNT += 1
        try:
            response = requests.post(
                llm_api_url,
                json=payload,
                timeout=resolved_timeout,
            )
        except requests.exceptions.ReadTimeout:
            # Do not consume call budget on timeout failures.
            _LLM_CALL_COUNT = max(_LLM_CALL_COUNT - 1, 0)
            if log_calls:
                print("[LLM RETRY] Read timeout; retrying once without timeout")
            _LLM_CALL_COUNT += 1
            response = requests.post(
                llm_api_url,
                json=payload,
                timeout=None,
            )
        response.raise_for_status()
        payload: dict[str, Any] = response.json() if response.content else {}
        raw_text = (
            payload.get("choices", [{}])[0]
            .get("message", {})
            .get("content", "")
        )

        think_matches = re.findall(r"<think>(.*?)</think>", raw_text, re.DOTALL | re.IGNORECASE)
        if show_think and think_matches:
            reasoning = "\n\n".join((match or "").strip() for match in think_matches if (match or "").strip())
            if reasoning:
                print("\n[Thought Process]\n" + reasoning + "\n")

        cleaned = re.sub(r"<think>.*?</think>", "", raw_text, flags=re.DOTALL | re.IGNORECASE).strip()
        _REQ_CACHE[sig] = (now, cleaned)
        # Light cleanup to avoid unbounded growth.
        if len(_REQ_CACHE) > 256:
            cutoff = now - max(dedupe_window, 0)
            for key, (ts, _) in list(_REQ_CACHE.items()):
                if ts < cutoff:
                    _REQ_CACHE.pop(key, None)

        if log_calls:
            elapsed = time.time() - started
            print(f"[LLM] #{_LLM_CALL_COUNT} model={llm_model} temp={temperature} max_tokens={resolved_max_tokens} took={elapsed:.2f}s")
        return cleaned
    except Exception as exc:
        # Do not consume call budget on failed attempts.
        _LLM_CALL_COUNT = max(_LLM_CALL_COUNT - 1, 0)
        # Keep failures visible so long-running generations don't look like silent loops.
        print(f"[LLM ERROR] {exc}", file=sys.stderr)
        return ""


if __name__ == "__main__":
    endpoint, model = _get_local_llm_settings()
    print(f"Using LOCAL_LLM_API_URL={endpoint}")
    print(f"Using LOCAL_LLM_MODEL={model}")
    print(call_local_llm([{"role": "user", "content": "Hello Qwen, what can you do?"}]))
