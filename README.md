# MNIST Experiment Tracking with MLflow + DagsHub

Assignment 3, Part C: Ram Chopra (100995123), Durham College.

Trains a dense neural network on MNIST (a baseline plus 4 variations) and tracks
every run on DagsHub's hosted MLflow server using Keras autologging.

| Run | Hidden layers | Optimizer | Batch | Epochs |
|---|---|---|---|---|
| 01_baseline_1x128_adam_bs64_ep50 | 128 | Adam | 64 | 50 |
| 02_deeper_2x256-128_adam_bs64_ep50 | 256, 128 | Adam | 64 | 50 |
| 03_sgd_1x128_bs64_ep50 | 128 | SGD (lr 0.01) | 64 | 50 |
| 04_rmsprop_1x256_bs128_ep20 | 256 | RMSprop | 128 | 20 |
| 05_wide_3x512-256-128_adam_bs256_ep15 | 512, 256, 128 | Adam | 256 | 15 |

## Run

```bash
pip install -r requirements.txt
python mnist_mlflow_dagshub.py
```

On the first run `dagshub.init()` prints a link to authorise your DagsHub account.

- MLflow Tracking URI: https://dagshub.com/RamChopra1/mnist-mlflow-experiments.mlflow
- Experiments: https://dagshub.com/RamChopra1/mnist-mlflow-experiments/experiments
