import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error
from config import PROCESSED, MODELS, DATASETS, TARGET, HORIZON, LGB_PARAMS
from preprocess import split

for name in DATASETS:
    df = pd.read_csv(PROCESSED / f"{name.lower()}.csv", index_col=0, parse_dates=True)
    tr, va, _ = split(df)
    # other datas, OT datas in training
    X_tr, y_tr = tr.drop(columns=TARGET), tr[TARGET]
    # other datas, OT datas in validataion
    X_va, y_va = va.drop(columns=TARGET), va[TARGET]

    # last known OT: the naive forecast, and the base the change is added to
    lag = f"OT_lag{HORIZON}"
    # learn the change from the last known OT instead of the OT value itself
    fit_tr, fit_va = y_tr - X_tr[lag], y_va - X_va[lag]

    model = lgb.LGBMRegressor(**LGB_PARAMS)
    model.fit(
        X_tr, fit_tr,
        eval_set=[(X_va, fit_va)],
        eval_metric="l1",
        callbacks=[lgb.early_stopping(100), lgb.log_evaluation(200)],
    )

    # store the model, the extension of pkl does not have meaning and it is convention in python
    joblib.dump(model, MODELS / f"lgb_{name.lower()}.pkl")
    # show the best number of  trees(more than 3000, increase n_estimators, increase learning_rate)
    print(f"{name}: best iteration = {model.best_iteration_}")

    # validation scores (use these, not the test set, to tune parameters)
    # change -> OT value
    pred_va = model.predict(X_va) + X_va[lag].to_numpy()
    naive_va = X_va[lag]
    for label, p in [("LightGBM", pred_va), ("Naive", naive_va)]:
        print(f"{name} val {label}: "
              f"MAE={mean_absolute_error(y_va, p):.4f} "
              f"RMSE={np.sqrt(mean_squared_error(y_va, p)):.4f}")
