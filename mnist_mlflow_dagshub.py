"""
Assignment 3 - Part C
Tracking a Deep Learning Experiment with MLflow and DagsHub

Author : Ram Chopra (100995123)
Course : BSc Artificial Intelligence, Durham College

What this script does
---------------------
1. Connects to a DagsHub repository and points MLflow at its tracking server.
2. Turns on TensorFlow/Keras autologging.
3. Loads and preprocesses MNIST.
4. Builds a baseline network (Flatten -> Dense(128, ReLU) -> Dense(10, Softmax)).
5. Trains the baseline (128 nodes, Adam, batch 64, 50 epochs) and logs the
   required hyperparameters.
6. Runs four more experiments that change layers / nodes / optimizer /
   batch size / epochs, each with a meaningful run name.
7. Pulls every run back from MLflow, prints a comparison table, saves it to
   CSV and reports the best run by validation accuracy.

Before running
--------------
    pip install -r requirements.txt
Then set REPO_OWNER / REPO_NAME below to your DagsHub repo. The first time
dagshub.init() runs it opens a browser link to authorise your account
(or set the DAGSHUB_USER_TOKEN environment variable).
"""

import os

import dagshub
import mlflow
import mlflow.tensorflow
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

# ---------------------------------------------------------------------------
# 1. DagsHub + MLflow set-up
# ---------------------------------------------------------------------------
REPO_OWNER = "RamChopra1"                 # your DagsHub username
REPO_NAME = "mnist-mlflow-experiments"    # the DagsHub repository you created
EXPERIMENT_NAME = "MNIST-Dense-NN"

SEED = 42
VALIDATION_SPLIT = 0.1                    # 54,000 train / 6,000 validation

dagshub.init(repo_owner=REPO_OWNER, repo_name=REPO_NAME, mlflow=True)
TRACKING_URI = f"https://dagshub.com/{REPO_OWNER}/{REPO_NAME}.mlflow"
mlflow.set_tracking_uri(TRACKING_URI)
mlflow.set_experiment(EXPERIMENT_NAME)
print(f"MLflow Tracking URI: {mlflow.get_tracking_uri()}")

# ---------------------------------------------------------------------------
# 2. Autologging (epoch-by-epoch metrics, optimizer config, model summary,
#    and the trained model itself)
# ---------------------------------------------------------------------------
mlflow.tensorflow.autolog(log_models=True, log_datasets=False)


# ---------------------------------------------------------------------------
# 3. Load and preprocess MNIST
# ---------------------------------------------------------------------------
def load_mnist():
    (x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()
    # Scale pixel values from 0-255 to 0-1 so training is stable
    x_train = x_train.astype("float32") / 255.0
    x_test = x_test.astype("float32") / 255.0
    # Labels stay as integers 0-9 -> sparse_categorical_crossentropy
    return (x_train, y_train), (x_test, y_test)


# ---------------------------------------------------------------------------
# 4. Model builder. hidden_layers=[128] is the required baseline:
#    Flatten -> Dense(128, relu) -> Dense(10, softmax)
# ---------------------------------------------------------------------------
def build_model(hidden_layers, optimizer_name, learning_rate=None):
    model = keras.Sequential(name="mnist_dense")
    model.add(keras.Input(shape=(28, 28)))
    model.add(layers.Flatten())
    for i, units in enumerate(hidden_layers, start=1):
        model.add(layers.Dense(units, activation="relu", name=f"hidden_{i}"))
    model.add(layers.Dense(10, activation="softmax", name="output"))

    optimizers = {
        "adam": keras.optimizers.Adam,
        "sgd": keras.optimizers.SGD,
        "rmsprop": keras.optimizers.RMSprop,
    }
    opt_cls = optimizers[optimizer_name.lower()]
    optimizer = opt_cls(learning_rate=learning_rate) if learning_rate else opt_cls()

    model.compile(
        optimizer=optimizer,
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


# ---------------------------------------------------------------------------
# 5 + 6. Train one configuration inside its own named MLflow run
# ---------------------------------------------------------------------------
def run_experiment(config, data):
    (x_train, y_train), (x_test, y_test) = data
    tf.keras.utils.set_random_seed(SEED)

    with mlflow.start_run(run_name=config["run_name"]):
        # Parameters required by the assignment (autolog adds more on top)
        mlflow.log_params({
            "num_hidden_layers": len(config["hidden_layers"]),
            "nodes_per_hidden_layer": str(config["hidden_layers"]),
            "optimizer_name": config["optimizer"],
            "batch_size_used": config["batch_size"],
            "epochs_used": config["epochs"],
            "learning_rate_used": config.get("learning_rate") or "default",
            "activation": "relu",
            "output_activation": "softmax",
            "validation_split": VALIDATION_SPLIT,
        })
        mlflow.set_tag("description", config["description"])

        model = build_model(
            config["hidden_layers"], config["optimizer"], config.get("learning_rate")
        )
        model.summary()

        history = model.fit(
            x_train, y_train,
            batch_size=config["batch_size"],
            epochs=config["epochs"],
            validation_split=VALIDATION_SPLIT,
            verbose=2,
        )

        # Final-epoch and best-epoch numbers, logged with clear names so
        # they are easy to compare in the DagsHub/MLflow UI
        h = history.history
        mlflow.log_metrics({
            "final_train_accuracy": h["accuracy"][-1],
            "final_val_accuracy": h["val_accuracy"][-1],
            "final_train_loss": h["loss"][-1],
            "final_val_loss": h["val_loss"][-1],
            "best_val_accuracy": max(h["val_accuracy"]),
            "best_val_loss": min(h["val_loss"]),
        })

        # Held-out test set (never seen during training or model selection)
        test_loss, test_acc = model.evaluate(x_test, y_test, verbose=0)
        mlflow.log_metrics({"test_accuracy": test_acc, "test_loss": test_loss})
        print(f"[{config['run_name']}] val_acc={h['val_accuracy'][-1]:.4f} "
              f"test_acc={test_acc:.4f}")


# ---------------------------------------------------------------------------
# Experiment grid: baseline + 4 variations (assignment asks for >= 3)
# ---------------------------------------------------------------------------
EXPERIMENTS = [
    {
        "run_name": "01_baseline_1x128_adam_bs64_ep50",
        "hidden_layers": [128],
        "optimizer": "adam",
        "batch_size": 64,
        "epochs": 50,
        "description": "Baseline required by the assignment.",
    },
    {
        "run_name": "02_deeper_2x256-128_adam_bs64_ep50",
        "hidden_layers": [256, 128],
        "optimizer": "adam",
        "batch_size": 64,
        "epochs": 50,
        "description": "More capacity: two hidden layers, more neurons.",
    },
    {
        "run_name": "03_sgd_1x128_bs64_ep50",
        "hidden_layers": [128],
        "optimizer": "sgd",
        "learning_rate": 0.01,
        "batch_size": 64,
        "epochs": 50,
        "description": "Same as baseline but plain SGD instead of Adam.",
    },
    {
        "run_name": "04_rmsprop_1x256_bs128_ep20",
        "hidden_layers": [256],
        "optimizer": "rmsprop",
        "batch_size": 128,
        "epochs": 20,
        "description": "Wider layer, RMSprop, bigger batch, fewer epochs.",
    },
    {
        "run_name": "05_wide_3x512-256-128_adam_bs256_ep15",
        "hidden_layers": [512, 256, 128],
        "optimizer": "adam",
        "batch_size": 256,
        "epochs": 15,
        "description": "Deep and wide, large batch, short training.",
    },
]


# ---------------------------------------------------------------------------
# 10 + 11. Compare runs and pick the best by validation accuracy
# ---------------------------------------------------------------------------
def compare_runs():
    runs = mlflow.search_runs(experiment_names=[EXPERIMENT_NAME])
    cols = {
        "tags.mlflow.runName": "run_name",
        "params.num_hidden_layers": "hidden_layers",
        "params.nodes_per_hidden_layer": "nodes",
        "params.optimizer_name": "optimizer",
        "params.batch_size_used": "batch_size",
        "params.epochs_used": "epochs",
        "metrics.final_train_accuracy": "train_acc",
        "metrics.final_val_accuracy": "val_acc",
        "metrics.final_train_loss": "train_loss",
        "metrics.final_val_loss": "val_loss",
        "metrics.best_val_accuracy": "best_val_acc",
        "metrics.best_val_loss": "best_val_loss",
        "metrics.test_accuracy": "test_acc",
    }
    table = runs[[c for c in cols if c in runs.columns]].rename(columns=cols)
    table = table.sort_values("val_acc", ascending=False).reset_index(drop=True)

    pd.set_option("display.width", 200)
    pd.set_option("display.max_columns", 20)
    print("\n=========== Run comparison (sorted by validation accuracy) ===========")
    print(table.round(4).to_string(index=False))
    table.round(4).to_csv("run_comparison.csv", index=False)

    best = table.iloc[0]
    print(f"\nBest run: {best['run_name']}  "
          f"(val_acc={best['val_acc']:.4f}, val_loss={best['val_loss']:.4f})")
    return table


if __name__ == "__main__":
    data = load_mnist()
    print(f"Train: {data[0][0].shape}, Test: {data[1][0].shape}")
    for cfg in EXPERIMENTS:
        run_experiment(cfg, data)
    compare_runs()
    print(f"\nOpen the runs at: https://dagshub.com/{REPO_OWNER}/{REPO_NAME}/experiments")
