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
ROUTER_MODEL_PATH = Path(os.getenv("ROUTER_MODEL_PATH", MODELS_DIR / "strategy_router.pkl"))

# Ensure directories exist
for p in [DATA_DIR, GROUND_TRUTH_DIR, RESULTS_DIR, EVAL_RUNS_DIR, DRIFT_LOGS_DIR, MODELS_DIR]:
    p.mkdir(parents=True, exist_ok=True)

# Models and LLM
OLLAMA_HOST = os.getenv("OLLAMA_HOST", "http://localhost:11434")
PRIMARY_MODEL = os.getenv("PRIMARY_MODEL", "llama3.1:8b")
SECONDARY_MODEL = os.getenv("SECONDARY_MODEL", "phi3:mini")
EMBEDDING_MODEL_NAME = os.getenv("EMBEDDING_MODEL", "BAAI/bge-small-en-v1.5")
RERANKER_MODEL_NAME = os.getenv("RERANKER_MODEL", "cross-encoder/ms-marco-MiniLM-L-6-v2")

# Thresholds & Parameters
DRIFT_THRESHOLD = float(os.getenv("DRIFT_THRESHOLD", "0.15"))
SPREAD_CONFIDENCE_THRESHOLD = float(os.getenv("SPREAD_CONFIDENCE_THRESHOLD", "0.35"))
AGREEMENT_CONFIDENCE_THRESHOLD = float(os.getenv("AGREEMENT_CONFIDENCE_THRESHOLD", "0.50"))
RRF_K_CONSTANT = 60
DEFAULT_TOP_K = 4
