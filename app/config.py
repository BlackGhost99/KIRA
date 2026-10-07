"""Réglages de KIRA, lus dans les variables d'environnement.

Aucun secret n'est écrit dans le code : clés d'API, mot de passe, jeton GitHub
se saisissent dans Render (Environment) ou dans un fichier .env local.
Les modules lisent toujours ``config.settings`` au moment de l'appel (jamais
une copie faite à l'import), ce qui permet de les modifier dans les tests.
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


def _s(name: str, default: str = "") -> str:
    v = os.environ.get(name)
    return v.strip() if v and v.strip() else default


def _i(name: str, default: int) -> int:
    try:
        return int(_s(name, str(default)))
    except ValueError:
        return default


def _f(name: str, default: float) -> float:
    try:
        return float(_s(name, str(default)))
    except ValueError:
        return default


def _b(name: str, default: bool) -> bool:
    v = _s(name, "1" if default else "0").lower()
    return v in ("1", "true", "yes", "oui", "on")


@dataclass
class Settings:
    # Identité et accès
    owner_name: str = "Brice"
    timezone: str = "Africa/Libreville"
    owner_password: str = ""
    secret_key: str = ""
    cron_token: str = ""
    session_days: int = 30
    access_minutes: int = 15
    cookie_secure: bool = True
    public_origin: str = ""
    allow_legacy_bearer: bool = False
    reauth_seconds: int = 300

    # Base de données : sqlite:///kira.db (local) ou l'adresse Supabase (Session pooler)
    database_url: str = "sqlite:///kira.db"

    # Embeddings en ligne : endpoint dédié, désactivé tant qu'il n'est pas configuré.
    embedding_url: str = ""
    embedding_api_key: str = ""
    embedding_model: str = "@cf/baai/bge-m3"
    embedding_dimensions: int = 1024
    embedding_timeout: int = 5
    memory_semantic_threshold: float = 0.65

    # Fournisseurs d'IA (ordre de priorité, bascule automatique en cas de panne)
    llm_priority: str = "anthropic,openai,deepseek,gemini,groq,openrouter,ollama"
    llm_routing_mode: str = "QUALITY"
    llm_economy_allow_paid: bool = False
    llm_trusted_providers: str = ""
    llm_timeout: int = 45
    gemini_api_key: str = ""
    gemini_model: str = "gemini-2.5-flash"
    gemini_free_tier: bool = False
    groq_free_tier: bool = False
    openrouter_api_key: str = ""
    openrouter_model: str = "openrouter/free"
    anthropic_api_key: str = ""
    anthropic_model: str = "claude-sonnet-5-5"
    anthropic_model_fast: str = "claude-haiku-4-5-20251001"
    anthropic_model_deep: str = "claude-opus-5-5"
    openai_api_key: str = ""
    openai_base_url: str = "https://api.openai.com/v1"
    openai_model: str = "gpt-4o"
    openai_model_fast: str = "gpt-4o-mini"
    openai_model_deep: str = ""
    deepseek_api_key: str = ""
    deepseek_base_url: str = "https://api.deepseek.com"
    deepseek_model: str = "deepseek-chat"
    groq_api_key: str = ""
    groq_base_url: str = "https://api.groq.com/openai/v1"
    groq_model: str = "llama-3.3-70b-versatile"
    ollama_url: str = ""
    ollama_model: str = "mistral:7b"

    # Garde-fous de coût et de boucle
    daily_token_budget: int = 1_500_000
    max_tool_rounds: int = 12

    # Évolution : ouverture de pull requests (jamais de fusion automatique)
    github_token: str = ""
    github_repo: str = "BlackGhost99/KIRA"
    github_base_branch: str = "main"
    evolution_run_tests: bool = True

    # Outils et veille
    tavily_api_key: str = ""
    sandbox_timeout: int = 30
    veille_max_per_source: int = 5
    veille_max_total: int = 30
    veille_auto_validate_min_score: float = 0.0  # 0 = désactivé (validation manuelle)
    interests: str = "physique, mathématiques, informatique, langues, droit"

    @classmethod
    def from_env(cls) -> "Settings":
        return cls(
            owner_name=_s("OWNER_NAME", "Brice"),
            timezone=_s("TIMEZONE", "Africa/Libreville"),
            owner_password=_s("OWNER_PASSWORD"),
            secret_key=_s("SECRET_KEY"),
            cron_token=_s("CRON_TOKEN"),
            session_days=_i("SESSION_DAYS", 30),
            access_minutes=_i("ACCESS_MINUTES", 15),
            cookie_secure=_b("COOKIE_SECURE", True),
            public_origin=_s("PUBLIC_ORIGIN", _s("RENDER_EXTERNAL_URL")),
            allow_legacy_bearer=_b("ALLOW_LEGACY_BEARER", False),
            reauth_seconds=_i("REAUTH_SECONDS", 300),
            database_url=_s("DATABASE_URL", "sqlite:///kira.db"),
            embedding_url=_s("EMBEDDING_URL"),
            embedding_api_key=_s("EMBEDDING_API_KEY"),
            embedding_model=_s("EMBEDDING_MODEL", "@cf/baai/bge-m3"),
            embedding_dimensions=_i("EMBEDDING_DIMENSIONS", 1024),
            embedding_timeout=_i("EMBEDDING_TIMEOUT", 5),
            memory_semantic_threshold=_f("MEMORY_SEMANTIC_THRESHOLD", 0.65),
            llm_priority=_s("LLM_PRIORITY", "anthropic,openai,deepseek,gemini,groq,openrouter,ollama"),
            llm_routing_mode=_s("LLM_ROUTING_MODE", "QUALITY"),
            llm_economy_allow_paid=_b("LLM_ECONOMY_ALLOW_PAID", False),
            llm_trusted_providers=_s("LLM_TRUSTED_PROVIDERS"),
            llm_timeout=_i("LLM_TIMEOUT", 45),
            gemini_api_key=_s("GEMINI_API_KEY"),
            gemini_model=_s("GEMINI_MODEL", "gemini-2.5-flash"),
            gemini_free_tier=_b("GEMINI_FREE_TIER", False),
            groq_free_tier=_b("GROQ_FREE_TIER", False),
            openrouter_api_key=_s("OPENROUTER_API_KEY"),
            openrouter_model=_s("OPENROUTER_MODEL", "openrouter/free"),
            anthropic_api_key=_s("ANTHROPIC_API_KEY"),
            anthropic_model=_s("ANTHROPIC_MODEL", "claude-sonnet-5-5"),
            anthropic_model_fast=_s("ANTHROPIC_MODEL_FAST", "claude-haiku-4-5-20251001"),
            anthropic_model_deep=_s("ANTHROPIC_MODEL_DEEP", "claude-opus-5-5"),
            openai_api_key=_s("OPENAI_API_KEY"),
            openai_base_url=_s("OPENAI_BASE_URL", "https://api.openai.com/v1"),
            openai_model=_s("OPENAI_MODEL", "gpt-4o"),
            openai_model_fast=_s("OPENAI_MODEL_FAST", "gpt-4o-mini"),
            openai_model_deep=_s("OPENAI_MODEL_DEEP"),
            deepseek_api_key=_s("DEEPSEEK_API_KEY"),
            deepseek_base_url=_s("DEEPSEEK_BASE_URL", "https://api.deepseek.com"),
            deepseek_model=_s("DEEPSEEK_MODEL", "deepseek-chat"),
            groq_api_key=_s("GROQ_API_KEY"),
            groq_base_url=_s("GROQ_BASE_URL", "https://api.groq.com/openai/v1"),
            groq_model=_s("GROQ_MODEL", "llama-3.3-70b-versatile"),
            ollama_url=_s("OLLAMA_URL"),
            ollama_model=_s("OLLAMA_MODEL", "mistral:7b"),
            daily_token_budget=_i("DAILY_TOKEN_BUDGET", 1_500_000),
            max_tool_rounds=_i("MAX_TOOL_ROUNDS", 12),
            github_token=_s("GITHUB_TOKEN"),
            github_repo=_s("GITHUB_REPO", "BlackGhost99/KIRA"),
            github_base_branch=_s("GITHUB_BASE_BRANCH", "main"),
            evolution_run_tests=_b("EVOLUTION_RUN_TESTS", True),
            tavily_api_key=_s("TAVILY_API_KEY"),
            sandbox_timeout=_i("SANDBOX_TIMEOUT", 30),
            veille_max_per_source=_i("VEILLE_MAX_PER_SOURCE", 5),
            veille_max_total=_i("VEILLE_MAX_TOTAL", 30),
            veille_auto_validate_min_score=_f("VEILLE_AUTO_VALIDATE_MIN_SCORE", 0.0),
            interests=_s("INTERESTS", "physique, mathématiques, informatique, langues, droit"),
        )


def load_dotenv(path: Path) -> int:
    """Lit un fichier .env (CLE=valeur, une par ligne) sans écraser les variables déjà définies. Sert en local."""
    try:
        text = path.read_text(encoding="utf-8")
    except OSError:
        return 0
    loaded = 0
    for raw in text.splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        key, value = key.strip(), value.strip()
        if len(value) >= 2 and value[0] == value[-1] and value[0] in "\"'":
            value = value[1:-1]
        if key and value and key not in os.environ:
            os.environ[key] = value
            loaded += 1
    return loaded


if not os.environ.get("KIRA_NO_DOTENV"):
    load_dotenv(Path(__file__).resolve().parents[1] / ".env")

settings = Settings.from_env()


def reload_settings() -> Settings:
    global settings
    settings = Settings.from_env()
    return settings
