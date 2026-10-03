"""Jev transport foundation. No workflow routing or performance calculation."""

from dataclasses import dataclass, field
import hashlib
import json
import logging
import os
from pathlib import Path
from time import perf_counter

import yaml
from typesafe_sdk import RetryPolicy, TypeSafeClient

from lib.jev_translation import assert_english_payload


ROOT = Path(__file__).resolve().parents[1]
log = logging.getLogger("triton-ascend-workflow")


@dataclass(frozen=True)
class JevSettings:
    api_key: str = field(repr=False)
    base_url: str = "https://api.typesafe.ai"
    model: str = "jev-1.13.0"
    timeout_seconds: float = 30
    max_retries: int = 1
    max_state_question_bytes: int = 28000
    max_request_bytes: int = 56000


def load_settings(config_path=None):
    """Prefer environment, then inline config; explicit legacy files remain supported."""
    path = Path(config_path) if config_path else ROOT / "config.yaml"
    cfg = yaml.safe_load(path.read_text(encoding="utf-8-sig"))["jev"]
    key = os.environ.get(cfg.get("api_key_env", "TYPESAFE_API_KEY"), "").strip()
    if not key:
        key = cfg.get("api_key", "")
        if not isinstance(key, str):
            raise ValueError("jev.api_key must be a string.")
        key = key.strip()
    # A legacy file is opt-in: stale config.local.yaml files must not silently
    # supply a key after switching to the single-config setup.
    if not key and cfg.get("credentials_file"):
        credentials = path.parent / cfg["credentials_file"]
        if credentials.is_file():
            local = yaml.safe_load(credentials.read_text(encoding="utf-8-sig")) or {}
            key = local.get("jev", {}).get("api_key", "")
    if not isinstance(key, str) or not key.strip():
        raise ValueError("Jev API key is missing; set jev.api_key in your config or the configured API key environment variable (default TYPESAFE_API_KEY).")
    settings = JevSettings(api_key=key.strip(), **{
        k: cfg[k] for k in (
            "base_url", "model", "timeout_seconds", "max_retries",
            "max_state_question_bytes", "max_request_bytes",
        ) if k in cfg
    })
    # Keep this foundation's credentials on the documented API origin.
    if settings.base_url.rstrip("/") != "https://api.typesafe.ai":
        raise ValueError("This Jev client requires the official https://api.typesafe.ai origin.")
    if settings.timeout_seconds <= 0 or settings.max_retries < 0:
        raise ValueError("Invalid Jev timeout or retry configuration.")
    if not (0 < settings.max_state_question_bytes <= 28000
            and 0 < settings.max_request_bytes <= 56000):
        raise ValueError("Jev byte budgets may be lowered, but not raised above the checked defaults.")
    return settings


def read_source(path):
    """Send decoded file contents, never a file handle or an executable upload."""
    path = Path(path)
    data = path.read_bytes()
    source = data.decode("utf-8-sig")
    if not source.strip() or "\x00" in source:
        raise ValueError("Kernel source must be nonempty UTF-8 text.")
    return {"filename": path.name, "sha256": hashlib.sha256(data).hexdigest(), "source": source}


def build_request(source_path, evidence_path, questions_path, *, model="jev-1.13.0"):
    evidence = json.loads(Path(evidence_path).read_text(encoding="utf-8-sig"))
    questions = json.loads(Path(questions_path).read_text(encoding="utf-8-sig"))
    if not isinstance(evidence, dict) or evidence.get("language") != "en":
        raise ValueError("Prepare reviewed English evidence with language='en' before calling Jev.")
    if "kernel" in evidence:
        raise ValueError("Evidence must not override the kernel read from the source file.")
    request = {"model": model, "state": {**evidence, "kernel": read_source(source_path)},
               "questions": questions}
    assert_english_payload(request)
    return request


def _json_bytes(value):
    # Spaces are retained to make the local budget more conservative.
    return len(json.dumps(value, ensure_ascii=False, allow_nan=False).encode("utf-8"))


def check_budget(request, settings):
    """Conservative byte preflight, NOT the provider's exact tokenizer.

    Jev has both a 64k total and a 32k state+longest-question token limit.
    Leave substantial headroom; record real usage from the response. Never truncate.
    """
    if set(request) != {"model", "state", "questions"}:
        raise ValueError("Jev request must contain exactly model, state and questions.")
    if not isinstance(request["state"], (str, dict, list)):
        raise ValueError("Jev state must be text, a JSON object or an array.")
    questions = request["questions"]
    if not isinstance(questions, dict) or not questions:
        raise ValueError("Jev requires at least one named question.")
    if request["model"] != settings.model:
        raise ValueError("Request model differs from the configured Jev model.")
    sizes = {
        "method": "conservative_utf8_json_bytes_not_exact_tokens",
        "state_bytes": _json_bytes(request["state"]),
        "state_plus_longest_question_bytes": _json_bytes(request["state"]) + max(
            _json_bytes({name: question}) for name, question in questions.items()),
        "request_bytes": _json_bytes(request),
    }
    if sizes["state_plus_longest_question_bytes"] > settings.max_state_question_bytes:
        raise ValueError("Jev state + longest question exceeds local byte budget; prepare a smaller evidence packet.")
    if sizes["request_bytes"] > settings.max_request_bytes:
        raise ValueError("Jev request exceeds local byte budget; shorten questions or split independent requests.")
    return sizes


def evaluate(request, settings):
    """Return the SDK's native JSON answers/usage; do not decide the next stage."""
    started = perf_counter()
    phase = "请求预检"
    try:
        budget = check_budget(request, settings)
        assert_english_payload(request)
        log.debug("[Jev] 开始调用: model=%s, questions=%d, request_bytes=%d",
                  settings.model, len(request["questions"]), budget["request_bytes"])
        phase = "服务调用"
        with TypeSafeClient(
            api_key=settings.api_key, base_url=settings.base_url, model=settings.model,
            timeout=settings.timeout_seconds, retry=RetryPolicy(max_retries=settings.max_retries),
        ) as client:
            result = client.system_one(state=request["state"], questions=request["questions"])
        phase = "响应校验"
        response = result.model_dump(mode="json", exclude_none=True)
        answers = response.get("answers", {})
        if set(answers) != set(request["questions"]):
            raise ValueError("Jev returned an incomplete or unexpected answer set.")
        for name, question in request["questions"].items():
            if answers[name].get("type") != question.get("type"):
                raise ValueError("Jev answer type does not match the question.")
    except Exception as exc:
        # SDK exception messages can contain request data or credentials.
        log.error("[Jev] %s失败: error_type=%s, elapsed=%.2fs",
                  phase, type(exc).__name__, perf_counter() - started)
        raise
    log.info("[Jev] 评分完成: model=%s, questions=%d, elapsed=%.2fs",
             settings.model, len(request["questions"]), perf_counter() - started)
    return response
