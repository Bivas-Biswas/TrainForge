import requests
import time

SERVER = "http://127.0.0.1:8000"

def train_model():
    payload = {
        "model_type": "svm",
        "dataset_path": "data/train.csv",
        "params": {
            "kernel": "linear",
            "C": 1.0
        }
    }

    r = requests.post(f"{SERVER}/train", json=payload)
    data = r.json()

    token = data["token"]
    print("Training started. Token:", token)

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


def main():
    token = train_model()

    print("Waiting for training...")

    while True:
        status = check_status(token)
        print("Status:", status)

        if status["state"] == "completed":
            break

        if status["state"] == "failed":
            print("Training failed")
            return

        time.sleep(2)

    print("Training finished")

    result = infer(token, [5.1, 3.5, 1.4, 0.2])
    print("Inference result:", result)


if __name__ == "__main__":
    main()