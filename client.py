import requests
import threading
import time

SERVER = "http://127.0.0.1:8000"
CLIENT_THREAD_COUNT = 5
DEFAULT_DATASET = "data/train.csv"

MODEL_RUN_CONFIGS = [
    {
        "model_type": "svm",
        "params": {"kernel": "linear", "C": 1.0},
    },
    {
        "model_type": "logistic_regression",
        "params": {"max_iter": 300},
    },
    {
        "model_type": "random_forest",
        "params": {"n_estimators": 100, "random_state": 42},
    },
]

def train_model(model_type="svm", params=None, dataset_path=DEFAULT_DATASET, log_prefix=""):
    if params is None:
        params = {}

    payload = {
        "model_type": model_type,
        "dataset_path": dataset_path,
        "params": params,
    }

    r = requests.post(f"{SERVER}/train", json=payload)
    data = r.json()

    if "token" not in data:
        print(f"{log_prefix}Train request failed: {data}")
        return None

    token = data["token"]
    print(f"{log_prefix}Training started. Token: {token} model={model_type}")

    return token


def check_status(token):
    r = requests.get(f"{SERVER}/status/{token}")
    return r.json()


def infer(token, features):
    payload = {
        "token": token,
        "features": features
    }

    r = requests.post(f"{SERVER}/infer", json=payload)
    return r.json()


def main(model_config=None, log_prefix=""):
    model_config = model_config or MODEL_RUN_CONFIGS[0]

    token = train_model(
        model_type=model_config["model_type"],
        params=model_config.get("params", {}),
        log_prefix=log_prefix,
    )

    if token is None:
        return

    print(f"{log_prefix}Waiting for training...")

    while True:
        status = check_status(token)
        print(f"{log_prefix}Status: {status}")

        if status["state"] == "completed":
            break

        if status["state"] == "failed":
            print(f"{log_prefix}Training failed")
            return

        time.sleep(2)

    print(f"{log_prefix}Training finished")

    result = infer(token, [5.1, 3.5, 1.4, 0.2])
    print(f"{log_prefix}Inference result: {result}")


def run_main_in_threads(thread_count=CLIENT_THREAD_COUNT):
    threads = []

    for index in range(thread_count):
        log_prefix = f"[thread-{index + 1}] "
        model_config = MODEL_RUN_CONFIGS[index % len(MODEL_RUN_CONFIGS)]
        thread = threading.Thread(target=main, args=(model_config, log_prefix))
        thread.start()
        threads.append(thread)

    for thread in threads:
        thread.join()


if __name__ == "__main__":
    run_main_in_threads()