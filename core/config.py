from pathlib import Path

MODEL_DIR = Path("model_store")
DB_PATH = Path("metadata.db")

TRAINER_COUNT = 2   # Number of concurrent trainer threads. Adjust based on expected load and available CPU resources.
MODEL_CACHE_SIZE = 32 # Number of models to keep in memory for inference. Adjust based on available RAM and model sizes.
HEALTH_CHECK_INTERVAL_SEC = 5 # Time limits and cleanup settings
TRAINER_TIMEOUT_SEC = 300 # 5 minutes timeout for training jobs
MODEL_INACTIVE_TTL_SEC = 3600 # 1 hour of inactivity before a model is considered expired
MODEL_CLEANUP_INTERVAL_SEC = 60 # Run cleanup every 60 seconds
MODEL_CLEANUP_BATCH_SIZE = 5 # Number of expired models to clean up in each run
MAX_TRAIN_MEMORY_BYTES = 100 * 1024 * 1024 # 100 MB memory limit for training jobs
MAX_TRAIN_CPU_SECONDS = 300 # 5 minutes CPU time limit for training jobs
