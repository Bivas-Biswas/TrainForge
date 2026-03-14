from pathlib import Path

MODEL_DIR = Path("model_store")
DB_PATH = Path("metadata.db")

# Number of concurrent trainer threads. Adjust based on expected load and available CPU resources.
TRAINER_COUNT = 2

# Number of models to keep in memory for inference. Adjust based on available RAM and model sizes.
MODEL_CACHE_SIZE = 32

# Time limits and cleanup settings
HEALTH_CHECK_INTERVAL_SEC = 5

# 5 minutes timeout for training jobs
TRAINER_TIMEOUT_SEC = 300

# 1 hour of inactivity before a model is considered expired
MODEL_INACTIVE_TTL_SEC = 3600

# Run cleanup every 60 seconds
MODEL_CLEANUP_INTERVAL_SEC = 60 

# Number of expired models to clean up in each run
MODEL_CLEANUP_BATCH_SIZE = 5

# 100 MB memory limit for training jobs
MAX_TRAIN_MEMORY_BYTES = 100 * 1024 * 1024
 
# 5 minutes CPU time limit for training jobs
MAX_TRAIN_CPU_SECONDS = 300

# Default storage limit per client (10 MB). This is a soft limit; clients can exceed it but may face cleanup of their models. Adjust as needed based on expected model sizes and storage capacity.
DEFAULT_MAX_MODELS_STORAGE_SUPPORT_BYTES = 10 * 1024 * 1024
