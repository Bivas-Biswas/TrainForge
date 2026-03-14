from pathlib import Path

MODEL_DIR = Path("model_store")
DB_PATH = Path("metadata.db")

TRAINER_COUNT = 2
MODEL_CACHE_SIZE = 32
HEALTH_CHECK_INTERVAL_SEC = 5
TRAINER_TIMEOUT_SEC = 300

MAX_TRAIN_MEMORY_BYTES = 100 * 1024 * 1024 # 100 MB Memory Limit
MAX_TRAIN_CPU_SECONDS = 300 # 5 minutes CPU time limit
