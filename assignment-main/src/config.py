from pathlib import Path

# abosolout root path
ROOT = Path(__file__).resolve().parents[1]

# paths
RAW = ROOT / "data" / "raw"
PROCESSED = ROOT / "data" / "processed"
MODELS = ROOT / "outputs" / "models"
FIGS = ROOT / "outputs" / "figures"

# make the directly if they are not existent
for p in (PROCESSED, MODELS, FIGS):
    p.mkdir(parents=True, exist_ok=True)

DATASETS = ["ETTh1", "ETTh2", "ETTm1", "ETTm2"]
# guess the oil tempature
TARGET = "OT"
# check the tempature by 1h or 15mins
HORIZON = 1

TRAIN_RATIO = 0.6
VAL_RATIO = 0.1

# the model always predicts the change from the last known OT (OT(t) - OT(t-HORIZON))
# and adds it back. predicting the OT value directly lost to the naive baseline (see report p.12)
# disabled because the training period is about 1 year. reconsider if the data covers 2+ years
USE_MONTH = False

# rolling labels
LAGS = {
    "h": [1, 2, 3, 6, 12, 24, 48, 168],
    "m": [1, 2, 3, 4, 8, 24 ,48, 96, 192, 672],
}

ROLLS = {"h": [24, 168], "m": [96, 672]}

LGB_PARAMS = dict(
    n_estimators=3000, learning_rate=0.01, num_leaves=31,
    subsample=0.8, subsample_freq=1, colsample_bytree=0.8,
    random_state=42, verbose=-1,
)
