import pandas as pd
from config import (RAW, PROCESSED, DATASETS, TARGET, HORIZON, LAGS, ROLLS, TRAIN_RATIO, VAL_RATIO,
                    USE_MONTH)

LOAD_COLS = ["HUFL", "HULL", "MUFL", "MULL", "LUFL", "LULL"]

# figure out time frequency ("h" = 1 hour, "m" = 15 min)
def freq_of(name):
    return "m" if name.startswith("ETTm") else "h"


def make_features(df, freq):
    # make the date the index and drop the date column
    df = df.set_index(pd.to_datetime(df["date"])).drop(columns="date")
    # new empty data frame for the LightGBM features
    X = pd.DataFrame(index=df.index)

    # calendar features
    X["hour"] = df.index.hour
    X["dayofweek"] = df.index.dayofweek
    if USE_MONTH:
        X["month"] = df.index.month
    if freq == "m":
        X["minute"] = df.index.minute

    # lag features: OT from N steps ago (only lags >= HORIZON, no future leakage)
    lags = sorted({HORIZON, *[l for l in LAGS[freq] if l >= HORIZON]})
    for lag in lags:
        X[f"OT_lag{lag}"] = df[TARGET].shift(lag)

    # rolling average and volatility, shifted first so the current OT is not included
    past = df[TARGET].shift(HORIZON)
    for w in ROLLS[freq]:
        X[f"OT_roll_mean{w}"] = past.rolling(w).mean()
        X[f"OT_roll_std{w}"] = past.rolling(w).std()

    # last known load values
    for c in LOAD_COLS:
        X[f"{c}_lag{HORIZON}"] = df[c].shift(HORIZON)

    # answer column, and drop the first rows that have no history
    X[TARGET] = df[TARGET]
    return X.dropna()


# time-ordered split: train 60% / val 10% / test 30%, no shuffle
def split(df):
    n = len(df)
    i_tr = int(n * TRAIN_RATIO)
    i_va = int(n * (TRAIN_RATIO + VAL_RATIO))
    return df.iloc[:i_tr], df.iloc[i_tr:i_va], df.iloc[i_va:]


# runs only with "uv run python src/preprocess.py", not on import
if __name__ == "__main__":
    for name in DATASETS:
        raw = pd.read_csv(RAW / f"{name}.csv")
        feat = make_features(raw, freq_of(name))
        feat.to_csv(PROCESSED / f"{name.lower()}.csv")
        print(f"{name}: {raw.shape} -> {feat.shape}")
