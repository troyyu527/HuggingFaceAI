"""Settings from environment variables / Space secrets — the only module that reads them."""

import os
from dataclasses import dataclass

from dotenv import load_dotenv

from persona.models import Endpoint


@dataclass(frozen=True)
class Config:
    llm_endpoints: list[Endpoint]  # main provider first, then the optional fallback
    hf_token: str
    kb_dataset: str
    chunks_path: str


def _models(value: str) -> list[str]:
    return [m.strip() for m in value.split(",") if m.strip()]


def _endpoint(prefix: str, models: str) -> Endpoint | None:
    base_url, api_key = os.getenv(f"{prefix}_BASE_URL", ""), os.getenv(f"{prefix}_API_KEY", "")
    return Endpoint(base_url, api_key, _models(models)) if base_url and _models(models) else None


def load() -> Config:
    """Read the settings; a local .env fills in what the environment doesn't set."""
    load_dotenv()
    # FREE_CHAT_MODELS is the old name of LLM_MODEL, still accepted
    main = _endpoint("LLM", os.getenv("LLM_MODEL") or os.getenv("FREE_CHAT_MODELS", ""))
    fallback = _endpoint("LLM_FALLBACK", os.getenv("LLM_FALLBACK_MODEL", ""))
    return Config(
        llm_endpoints=[e for e in (main, fallback) if e],
        hf_token=os.getenv("HF_TOKEN", ""),
        kb_dataset=os.getenv("KB_DATASET", ""),
        chunks_path=os.getenv("CHUNKS_PATH", ""),
    )
