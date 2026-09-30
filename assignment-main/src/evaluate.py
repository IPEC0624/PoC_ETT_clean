import joblib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import mean_absolute_error, mean_squared_error
from config import ROOT, PROCESSED, MODELS, FIGS, DATASETS, TARGET, HORIZON
from preprocess import split, freq_of

# applies seaborn's style to every matplotlib graph
sns.set_theme(style="whitegrid")
rows = []

# read processed csv files, split them and keep only the test part
for name in DATASETS:
    df = pd.read_csv(PROCESSED / f"{name.lower()}.csv", index_col=0, parse_dates=True)
    _, _, te = split(df)
    X_te, y_te = te.drop(columns=TARGET), te[TARGET]

    # load the models which finished training and validation and predict OT
    model = joblib.load(MODELS / f"lgb_{name.lower()}.pkl")
    pred = model.predict(X_te)

    # naive baseline: predict the last known OT. LightGBM must get a lower error (MAE/RMSE) than this
    naive = X_te[f"OT_lag{HORIZON}"]

    # two models show MAE, RMSE and then need to compare
    for label, p in [("LightGBM", pred), ("Naive", naive)]:
        rows.append(dict(
            dataset=name, model=label,
            MAE=mean_absolute_error(y_te, p),
            RMSE=np.sqrt(mean_squared_error(y_te, p)),
        ))

    # number of rows in 7 days (168 for h, 672 for m)
    steps = 7 * 24 * (4 if freq_of(name) == "m" else 1)
    # this creates a figure and axes
    fig, ax = plt.subplots(figsize=(12, 4))
    # plot actual vs predicted for the last 7 days
    ax.plot(te.index[-steps:], y_te.iloc[-steps:], label="Actual", linewidth=1.5)
    ax.plot(te.index[-steps:], pred[-steps:], label="LightGBM", linewidth=1.2)
    ax.set(title=f"{name}: OT actual vs predicted (last 7 days of test)",
           xlabel="Date", ylabel="OT (°C)")
    ax.legend()
    # adjust margins, save as PNG, and free memory
    fig.tight_layout()
    fig.savefig(FIGS / f"{name.lower()}_pred.png", dpi=150)
    plt.close(fig)

# save the metrics table (full precision) and show it rounded to 4 decimals
metrics = pd.DataFrame(rows)
metrics.to_csv(ROOT / "outputs" / "metrics.csv", index=False)
print(metrics.round(4).to_string(index=False))
