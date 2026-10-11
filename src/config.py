import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

BASE_DIR = Path(__file__).resolve().parent.parent

# Storage and Directories
DATA_DIR = Path(os.getenv("DATA_DIR", BASE_DIR / "data"))
GROUND_TRUTH_DIR = DATA_DIR / "ground_truth"
CHROMA_PERSIST_PATH = str(os.getenv("CHROMA_PERSIST_PATH", DATA_DIR / "chroma_db"))
CHROMA_COLLECTION = os.getenv("CHROMA_COLLECTION", "react_docs")

RESULTS_DIR = Path(os.getenv("RESULTS_DIR", BASE_DIR / "results"))
EVAL_RUNS_DIR = RESULTS_DIR / "eval_runs"
DRIFT_LOGS_DIR = RESULTS_DIR / "drift_logs"
MODELS_DIR = Path(os.getenv("MODELS_DIR", BASE_DIR / "models"))
_default_router_path = MODELS_DIR / "strategy_router.pkl"
if not _default_router_path.exists() and (MODELS_DIR / "router.pkl").exists():
    _default_router_path = MODELS_DIR / "router.pkl"
ROUTER_MODEL_PATH = Path(os.getenv("ROUTER_MODEL_PATH", _default_router_path))

# Ensure directories exist
for p in [DATA_DIR, GROUND_TRUTH_DIR, RESULTS_DIR, EVAL_RUNS_DIR, DRIFT_LOGS_DIR, MODELS_DIR]:
    p.mkdir(parents=True, exist_ok=True)

# Models and LLM
LLM_PROVIDER = os.getenv("LLM_PROVIDER", "auto")  # "auto", "ollama", "openai", "extractive"
OLLAMA_HOST = os.getenv("OLLAMA_HOST", "http://localhost:11434")
PRIMARY_MODEL = os.getenv("PRIMARY_MODEL", "llama3.1:8b")
SECONDARY_MODEL = os.getenv("SECONDARY_MODEL", "phi3:mini")
OPENAI_API_BASE = os.getenv("OPENAI_API_BASE", "https://api.openai.com/v1")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
EMBEDDING_MODEL_NAME = os.getenv("EMBEDDING_MODEL", "BAAI/bge-small-en-v1.5")
RERANKER_MODEL_NAME = os.getenv("RERANKER_MODEL", "cross-encoder/ms-marco-MiniLM-L-6-v2")

# Thresholds & Parameters
DRIFT_THRESHOLD = float(os.getenv("DRIFT_THRESHOLD", "0.15"))
SPREAD_CONFIDENCE_THRESHOLD = float(os.getenv("SPREAD_CONFIDENCE_THRESHOLD", "0.35"))
AGREEMENT_CONFIDENCE_THRESHOLD = float(os.getenv("AGREEMENT_CONFIDENCE_THRESHOLD", "0.50"))
RRF_K_CONSTANT = int(os.getenv("RRF_K_CONSTANT", "60"))
DEFAULT_TOP_K = int(os.getenv("DEFAULT_TOP_K", "4"))
SEMANTIC_CACHE_MAX_SIZE = int(os.getenv("SEMANTIC_CACHE_MAX_SIZE", "256"))

# Vector Store Backend & PostgreSQL / pgvector Settings
VECTOR_STORE_BACKEND = os.getenv("VECTOR_STORE_BACKEND", "auto")  # "auto", "pgvector", "chroma"
PG_HOST = os.getenv("PG_HOST", "localhost")
PG_PORT = int(os.getenv("PG_PORT", "5432"))
PG_DATABASE = os.getenv("PG_DATABASE", "apexrag")
PG_USER = os.getenv("PG_USER", "postgres")
PG_PASSWORD = os.getenv("PG_PASSWORD", "postgres_password")
PG_DSN = os.getenv(
    "PG_DSN",
    f"postgresql://{PG_USER}:{PG_PASSWORD}@{PG_HOST}:{PG_PORT}/{PG_DATABASE}",
)

# Concurrency & Non-blocking ThreadPool / Semaphore Settings
MAX_CONCURRENT_EMBEDS = int(os.getenv("MAX_CONCURRENT_EMBEDS", "16"))
MAX_CONCURRENT_RERANKS = int(os.getenv("MAX_CONCURRENT_RERANKS", "4"))
PARALLEL_DUAL_RETRIEVAL = os.getenv("PARALLEL_DUAL_RETRIEVAL", "true").lower() in ("true", "1", "yes")

# Strategy Router & Shannon Entropy Gating Settings
ROUTER_ENTROPY_THRESHOLD = float(os.getenv("ROUTER_ENTROPY_THRESHOLD", "1.25"))
ROUTER_FALLBACK_STRATEGY = os.getenv("ROUTER_FALLBACK_STRATEGY", "rerank")




