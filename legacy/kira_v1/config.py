import os

PROPRIETAIRE = "Brice"

DEFAULT_MODELS = [
    "mistral:7b"
]

# Règles fondatrices (non modifiables sans validation explicite)
PRINCIPES = [
    "Souveraineté absolue du propriétaire",
    "Aucune action sans trace",
    "Évolution uniquement par proposition validée",
    "Séparation cognition/exécution"
]

# --- API GROQ (optionnelle et contrôlée) ---
USE_CLOUD = False  # Cloud désactivé par défaut
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")  # Renseignez via variable d'environnement
GROQ_MODEL = "llama-3.3-70b-versatile"  # Recommended replacement for llama3-70b-8192
# Alternatives (per Groq deprecations): "llama-3.1-8b-instant"

# --- LOCAL (Ollama) ---
# Par défaut, on évite les appels locaux pour réduire la latence.
USE_LOCAL = False
# Si True, n'appelle Ollama que si Groq est désactivé ou en échec.
LOCAL_FALLBACK_ONLY = True

# --- LITE (léger, sans LLM lourd) ---
USE_LITE = True
LITE_MAX_OUTPUT_CHARS = 1600
LITE_DISABLE_AUTO_TASKS = False
LITE_DISABLE_VECTOR_MEMORY = True
LITE_DISABLE_OBSERVATION_CONTEXT = False
LITE_DISABLE_EVOLUTION_AUTO = False
LITE_DISABLE_OBSERVATION_AUTO = False
LITE_STORE_FILE = "lite_store.json"
LITE_HISTORY_MAX = 120
LITE_SUMMARY_SENTENCES = 3

# --- MICRO LLM (optionnel, ultra léger) ---
MICRO_LLM_ENABLED = False
MICRO_LLM_MODE = "endpoint"  # endpoint | command
MICRO_LLM_ENDPOINT = "http://127.0.0.1:8081/generate"
MICRO_LLM_COMMAND = ""
MICRO_LLM_TIMEOUT_SECONDS = 3
MICRO_LLM_MAX_OUTPUT_CHARS = 1800
MICRO_LLM_TEMPERATURE = 0.4

# --- PERFORMANCE ---
# fast | balanced | full | max
PERF_MODE = "balanced"
PERF_PRESETS = {
    "fast": {
        "memory_query": False,
        "memory_write": False,
        "memory_items": 2,
        "memory_max_chars": 600,
        "observation_query": False,
        "observation_items": 1,
        "observation_max_chars": 600,
        "auto_tasks": False,
        "synthesis_mode": "heuristic",
        "groq_max_tokens": 512,
        "local_num_predict": 512,
    },
    "balanced": {
        "memory_query": True,
        "memory_write": True,
        "memory_items": 2,
        "memory_max_chars": 800,
        "observation_query": True,
        "observation_items": 1,
        "observation_max_chars": 800,
        "auto_tasks": True,
        "synthesis_mode": "consensus",
        "groq_max_tokens": 800,
        "local_num_predict": 800,
    },
    "full": {
        "memory_query": True,
        "memory_write": True,
        "memory_items": 6,
        "memory_max_chars": 2600,
        "observation_query": True,
        "observation_items": 5,
        "observation_max_chars": 3000,
        "auto_tasks": True,
        "synthesis_mode": "consensus",
        "groq_max_tokens": 2048,
        "local_num_predict": 2048,
    },
    "max": {
        "memory_query": True,
        "memory_write": True,
        "memory_items": 20,
        "memory_max_chars": 0,
        "observation_query": True,
        "observation_items": 10,
        "observation_max_chars": 0,
        "auto_tasks": True,
        "synthesis_mode": "consensus",
        "groq_max_tokens": 4096,
        "local_num_predict": 4096,
    },
}


def get_perf_settings():
    return PERF_PRESETS.get(PERF_MODE, PERF_PRESETS["balanced"])

# --- AUTONOMY ---
AUTONOMY_ENABLED = True
AUTONOMY_SUGGESTION_COUNT = 3

# --- STYLE / LEARNING ---
STYLE_MODE = "natural"  # natural | strict
LEARNING_ENABLED = True
LEARNING_FORCE_RECALL = True
LEARNING_MIN_ITEMS = 1

# --- MODULE EVOLUTION ---
EVOLUTION_AUTO_THRESHOLD = 15  # Analyse auto tous les N dialogues
EVOLUTION_AUTO_ENABLED = True  # Activer les propositions frequentes automatiques

# --- CANAL OBSERVATION ---
OBSERVATION_AUTO_ENABLED = True  # Mode veille automatique
OBSERVATION_AUTO_INTERVAL_HOURS = 24  # Intervalle par défaut (en heures)
OBSERVATION_AUTO_VALIDATE = False
OBSERVATION_AUTO_VALIDATE_MIN_SCORE = 0.8

# --- SYNTHESE REPONSES ---
SYNTHESIS_MODE = "consensus"  # "consensus" ou "heuristic"
SYNTHESIS_MAX_INPUT_CHARS = 20000
SYNTHESIS_MIN_LEN = 80
SYNTHESIS_MAX_LEN = 5000

# --- MEMOIRE ---
MEMORY_SUMMARY_MAX_CHARS = 1200
MEMORY_AUTO_PRUNE = False
MEMORY_MAX_ITEMS = 5000

# --- GROQ ROBUSTESSE ---
GROQ_TIMEOUT_SECONDS = 30
GROQ_MAX_RETRIES = 2
GROQ_RETRY_BACKOFF_SECONDS = 2

# --- GROQ / LOCAL GENERATION LIMITS ---
GROQ_MAX_TOKENS = 4096
LOCAL_NUM_PREDICT = 4096

# --- GROQ CIRCUIT BREAKER ---
GROQ_CIRCUIT_BREAKER_ENABLED = True
GROQ_CIRCUIT_BREAKER_SECONDS = 300
GROQ_DISABLE_RETRIES = True
