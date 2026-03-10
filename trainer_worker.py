import os
import sqlite3
import joblib
import resource
import numpy as np
from sklearn import svm


MODEL_DIR = "model_store"

def set_limits():
    mem = 500 * 1024 * 1024
    resource.setrlimit(resource.RLIMIT_AS, (mem, mem))

    cpu = 60
    resource.setrlimit(resource.RLIMIT_CPU, (cpu, cpu))


def train_model(token, dataset_path, params):
    print(f"Trainer started job {token}")
    
    set_limits()

    data = np.loadtxt(dataset_path, delimiter=",")

    X = data[:, :-1]
    y = data[:, -1]

    model = svm.SVC(**params)

    model.fit(X, y)

    path = f"{MODEL_DIR}/{token}.bin"
    tmp = path + ".tmp"

    joblib.dump(model, tmp)

    os.rename(tmp, path)

    conn = sqlite3.connect("metadata.db")
    c = conn.cursor()

    c.execute(
        "UPDATE models SET status='completed' WHERE token=?",
        (token,)
    )

    conn.commit()
    conn.close()
    

def trainer_worker(job_queue):

    while True:

        job = job_queue.get()

        if job is None:
            break

        token, dataset, params = job

        conn = sqlite3.connect("metadata.db")
        c = conn.cursor()

        c.execute(
            "UPDATE models SET status='training' WHERE token=?",
            (token,)
        )

        conn.commit()
        conn.close()

        try:
            train_model(token, dataset, params)

        except Exception:
            conn = sqlite3.connect("metadata.db")
            c = conn.cursor()

            c.execute(
                "UPDATE models SET status='failed' WHERE token=?",
                (token,)
            )

            conn.commit()
            conn.close()