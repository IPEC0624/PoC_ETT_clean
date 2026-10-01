import joblib
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from config import ROOT, MODELS, FIGS, DATASETS

# same colors and style as eda.py
BLUE, ORANGE = "#2a78d6", "#eb6834"
INK, MUTED, GRID = "#0b0b0b", "#52514e", "#e6e5e0"
sns.set_theme(style="whitegrid", rc={
    "axes.edgecolor": GRID, "grid.color": GRID, "axes.labelcolor": MUTED,
    "xtick.color": MUTED, "ytick.color": MUTED, "text.color": INK,
    "axes.titleweight": "bold", "axes.titlesize": 12,
})
OUT = FIGS / "report"
OUT.mkdir(parents=True, exist_ok=True)

# 1. test MAE: LightGBM vs naive (reads the metrics saved by evaluate.py)
metrics = pd.read_csv(ROOT / "outputs" / "metrics.csv")
fig, ax = plt.subplots(figsize=(9, 4))
sns.barplot(data=metrics, x="dataset", y="MAE", hue="model", hue_order=["Naive", "LightGBM"],
            palette=[MUTED, BLUE], width=0.6, gap=0.15, ax=ax)
for c in ax.containers:
    ax.bar_label(c, fmt="%.3f", padding=3, color=MUTED, fontsize=9)
ax.set(title="Test MAE (°C): LightGBM vs naive (last value)", xlabel="", ylabel="MAE (°C)")
ax.legend(title="", frameon=False, loc="upper right")
fig.tight_layout()
fig.savefig(OUT / "test_mae.png", dpi=200)
plt.close(fig)

# 2. feature importance (gain), top 10, ETTh1 vs ETTh2
fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
for ax, name in zip(axes, ["ETTh1", "ETTh2"]):
    model = joblib.load(MODELS / f"lgb_{name.lower()}.pkl")
    imp = pd.Series(model.booster_.feature_importance("gain"), index=model.feature_name_)
    imp = (imp / imp.sum()).sort_values(ascending=False).head(10)
    ax.barh(imp.index, imp.values, color=BLUE, height=0.6)
    for y, v in enumerate(imp.values):
        ax.text(v + 0.005, y, f"{v:.0%}", va="center", color=MUTED, fontsize=9)
    ax.invert_yaxis()
    ax.set(title=f"{name}: feature importance (gain share, top 10)", xlabel="share of total gain",
           xlim=(0, imp.values.max() * 1.18))
fig.tight_layout()
fig.savefig(OUT / "feature_importance.png", dpi=200)
plt.close(fig)

print(f"saved report figures to {OUT}")
