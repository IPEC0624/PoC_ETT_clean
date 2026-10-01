import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from config import RAW, FIGS, DATASETS, TARGET, TRAIN_RATIO, VAL_RATIO

LOAD_COLS = ["HUFL", "HULL", "MUFL", "MULL", "LUFL", "LULL"]
EDA = FIGS / "eda"
EDA.mkdir(parents=True, exist_ok=True)

# colors: one hue per role, same across all figures
BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
INK, MUTED, GRID = "#0b0b0b", "#52514e", "#e6e5e0"

sns.set_theme(style="whitegrid", rc={
    "axes.edgecolor": GRID, "grid.color": GRID, "axes.labelcolor": MUTED,
    "xtick.color": MUTED, "ytick.color": MUTED, "text.color": INK,
    "axes.titleweight": "bold", "axes.titlesize": 12, "lines.linewidth": 2,
})


def load(name):
    return pd.read_csv(RAW / f"{name}.csv", parse_dates=["date"]).set_index("date")


def period_bounds(df):
    n = len(df)
    return df.index[int(n * TRAIN_RATIO)], df.index[int(n * (TRAIN_RATIO + VAL_RATIO))]


def save(fig, file):
    fig.tight_layout()
    fig.savefig(EDA / file, dpi=200)
    plt.close(fig)


# 1. OT over time with train / val / test periods (daily mean for readability)
fig, axes = plt.subplots(2, 1, figsize=(12, 5.5), sharex=True)
for ax, name in zip(axes, ["ETTh1", "ETTh2"]):
    df = load(name)
    t_val, t_test = period_bounds(df)
    daily = df[TARGET].resample("D").mean()
    ax.axvspan(t_val, t_test, color=AQUA, alpha=0.12, lw=0)
    ax.axvspan(t_test, df.index[-1], color=ORANGE, alpha=0.10, lw=0)
    ax.plot(daily.index, daily.values, color=BLUE, lw=1.5)
    ax.set(title=f"{name}: daily mean OT", ylabel="OT (°C)")
    top = ax.get_ylim()[1]
    for x, label in [(df.index[0], "train 60%"), (t_val, "val 10%"), (t_test, "test 30%")]:
        ax.text(x, top, f" {label}", va="top", ha="left", color=MUTED, fontsize=9)
save(fig, "eda_ot_timeline.png")

# 2. OT distribution: train vs test (distribution shift)
rows = []
for name in DATASETS:
    df = load(name)
    t_val, t_test = period_bounds(df)
    rows.append(pd.DataFrame({"dataset": name, "period": "train", "OT": df.loc[:t_val, TARGET]}))
    rows.append(pd.DataFrame({"dataset": name, "period": "test", "OT": df.loc[t_test:, TARGET]}))
dist = pd.concat(rows)
fig, ax = plt.subplots(figsize=(9, 4))
sns.boxplot(data=dist, x="dataset", y="OT", hue="period", palette=[BLUE, ORANGE],
            fliersize=0, width=0.6, linewidth=1.2, gap=0.15, ax=ax)
ax.set(title="OT distribution: train vs test period", xlabel="", ylabel="OT (°C)")
ax.legend(title="", frameon=False, loc="upper right")
save(fig, "eda_ot_shift.png")

# 3. lag-1 autocorrelation: OT itself vs its 1-step change
acf = pd.DataFrame([
    {"dataset": n, "series": s, "acf1": v}
    for n in DATASETS
    for s, v in [("OT", load(n)[TARGET].autocorr(1)),
                 ("ΔOT (1-step change)", load(n)[TARGET].diff().autocorr(1))]
])
fig, ax = plt.subplots(figsize=(9, 4))
sns.barplot(data=acf, x="dataset", y="acf1", hue="series", palette=[BLUE, ORANGE],
            width=0.6, gap=0.15, ax=ax)
for c in ax.containers:
    ax.bar_label(c, fmt="%.2f", padding=3, color=MUTED, fontsize=9)
ax.axhline(0, color=MUTED, lw=1)
ax.set(title="Lag-1 autocorrelation", xlabel="", ylabel="autocorrelation", ylim=(-0.1, 1.12))
ax.legend(title="", frameon=False, loc="upper center", ncol=2, bbox_to_anchor=(0.5, -0.08))
save(fig, "eda_autocorr.png")

# 4. correlation of each load with OT
fig, axes = plt.subplots(1, 2, figsize=(10, 3.8), sharey=True)
for ax, name in zip(axes, ["ETTh1", "ETTh2"]):
    corr = load(name).corr()[TARGET].drop(TARGET).reindex(LOAD_COLS)
    ax.barh(corr.index, corr.values, color=BLUE, height=0.6)
    for y, v in enumerate(corr.values):
        ax.text(v + (0.01 if v >= 0 else -0.01), y, f"{v:.2f}", va="center",
                ha="left" if v >= 0 else "right", color=MUTED, fontsize=9)
    ax.axvline(0, color=MUTED, lw=1)
    ax.set(title=f"{name}: correlation with OT", xlim=(-0.3, 0.65), xlabel="Pearson r")
    ax.invert_yaxis()
save(fig, "eda_load_corr.png")

# 5. daily cycle: mean OT by hour, as deviation from each day's mean
fig, ax = plt.subplots(figsize=(9, 3.8))
for name, color in [("ETTh1", BLUE), ("ETTh2", ORANGE)]:
    ot = load(name)[TARGET]
    dev = ot - ot.groupby(ot.index.date).transform("mean")
    prof = dev.groupby(dev.index.hour).mean()
    ax.plot(prof.index, prof.values, color=color, marker="o", ms=4, label=name)
    ax.text(23.3, prof.values[-1], name, color=INK, va="center", fontsize=9)
ax.axhline(0, color=MUTED, lw=1)
ax.set(title="Daily cycle: OT minus daily mean, by hour", xlabel="hour of day",
       ylabel="°C", xticks=range(0, 24, 3), xlim=(-0.5, 25))
ax.legend(frameon=False, loc="upper left")
save(fig, "eda_daily_cycle.png")

print(f"saved EDA figures to {EDA}")
