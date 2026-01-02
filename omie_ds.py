import warnings
warnings.filterwarnings("ignore")

import pandas as pd
import numpy as np
from pathlib import Path

from prophet import Prophet
from xgboost import XGBRegressor, XGBClassifier
from sklearn.metrics import mean_absolute_error, mean_squared_error, roc_auc_score
from sklearn.model_selection import train_test_split



BASE_DIR = Path(r" your_dir ") #imput your dir
OUTPUT_DIR = BASE_DIR / "outputs"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

DAILY_PATH = BASE_DIR / "pdbcpt_daily_2023_onwards.csv"
QH_PATH = BASE_DIR / "pdbcpt_quarter_hour_2023_onwards.csv"

HORIZON_DAYS = 90
INTRADAY_TEST_DAYS = 7
PEAK_QUANTILE = 0.80  



def require_cols(df: pd.DataFrame, cols: list, name: str):
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise ValueError(f"[{name}] Missing required columns: {missing}. Found: {list(df.columns)}")

def load_data():
    if not DAILY_PATH.exists():
        raise FileNotFoundError(f"Daily file not found: {DAILY_PATH}")
    if not QH_PATH.exists():
        raise FileNotFoundError(f"Quarter-hour file not found: {QH_PATH}")

    daily = pd.read_csv(DAILY_PATH, parse_dates=["date"])
    qh = pd.read_csv(QH_PATH, parse_dates=["date", "datetime"])

    require_cols(daily, ["date", "daily_avg_price", "daily_min_price", "daily_max_price"], "DAILY")
    require_cols(qh, ["date", "datetime", "period", "price_eur_mwh"], "QH")

    daily = daily.sort_values("date").reset_index(drop=True)
    qh = qh.sort_values("datetime").reset_index(drop=True)

    for c in ["daily_avg_price", "daily_min_price", "daily_max_price"]:
        daily[c] = pd.to_numeric(daily[c], errors="coerce")

    qh["price_eur_mwh"] = pd.to_numeric(qh["price_eur_mwh"], errors="coerce")
    qh["period"] = pd.to_numeric(qh["period"], errors="coerce")

    daily["date"] = pd.to_datetime(daily["date"])
    qh["datetime"] = pd.to_datetime(qh["datetime"])
    qh["date"] = pd.to_datetime(qh["date"]).dt.date

    daily = daily.dropna(subset=["date", "daily_avg_price", "daily_min_price", "daily_max_price"]).reset_index(drop=True)
    qh = qh.dropna(subset=["datetime", "date", "period", "price_eur_mwh"]).reset_index(drop=True)

    return daily, qh



def rmse(y_true, y_pred):
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))

def print_reg_metrics(tag, y_true, y_pred):
    mae = float(mean_absolute_error(y_true, y_pred))
    _rmse = rmse(y_true, y_pred)
    print(f"[{tag}] MAE: {mae:.2f} €/MWh | RMSE: {_rmse:.2f} €/MWh")
    return mae, _rmse



def build_daily_prophet_model(df_daily: pd.DataFrame, horizon_days: int = 90):
    df_p = df_daily[["date", "daily_avg_price"]].rename(columns={"date": "ds", "daily_avg_price": "y"}).copy()
    df_p["ds"] = pd.to_datetime(df_p["ds"])

    if len(df_p) <= horizon_days + 60:
        raise ValueError("Not enough data for Prophet horizon. Reduce horizon_days or load more history.")

    train = df_p.iloc[:-horizon_days].copy()
    test = df_p.iloc[-horizon_days:].copy()

    m = Prophet(
        yearly_seasonality=True,
        weekly_seasonality=True,
        daily_seasonality=False,
        seasonality_mode="additive",
    )
    m.fit(train)

    future = m.make_future_dataframe(periods=horizon_days)
    forecast = m.predict(future)

    test_merged = test.merge(forecast[["ds", "yhat"]], on="ds", how="left")
    print_reg_metrics("Prophet Daily Avg", test_merged["y"], test_merged["yhat"])

    return m, forecast, test_merged



def add_daily_features(df_daily: pd.DataFrame) -> pd.DataFrame:
    df = df_daily[["date", "daily_avg_price", "daily_min_price", "daily_max_price"]].copy()
    df["date"] = pd.to_datetime(df["date"])

    df["dow"] = df["date"].dt.weekday
    df["is_weekend"] = (df["dow"] >= 5).astype(int)
    df["month"] = df["date"].dt.month
    df["quarter"] = df["date"].dt.quarter
    df["day"] = df["date"].dt.day

    m = df["month"]
    df["season"] = np.select(
        [m.isin([12, 1, 2]), m.isin([3, 4, 5]), m.isin([6, 7, 8]), m.isin([9, 10, 11])],
        [1, 2, 3, 4],
        default=0
    )

    df["price_range"] = df["daily_max_price"] - df["daily_min_price"]
    df["mid_price"] = (df["daily_max_price"] + df["daily_min_price"]) / 2.0

    df["avg_delta_1d"] = df["daily_avg_price"].diff(1)
    df["avg_return_1d"] = df["daily_avg_price"].pct_change(1)

    df["vol_7d"] = df["daily_avg_price"].rolling(7).std()
    df["vol_30d"] = df["daily_avg_price"].rolling(30).std()
    df["range_avg_7d"] = df["price_range"].rolling(7).mean()
    df["range_avg_30d"] = df["price_range"].rolling(30).mean()
    df["avg_ma_7"] = df["daily_avg_price"].rolling(7).mean()
    df["avg_ma_30"] = df["daily_avg_price"].rolling(30).mean()

    for lag in [1, 2, 7, 14, 30]:
        df[f"avg_lag_{lag}"] = df["daily_avg_price"].shift(lag)
        df[f"range_lag_{lag}"] = df["price_range"].shift(lag)
        df[f"max_lag_{lag}"] = df["daily_max_price"].shift(lag)
        df[f"min_lag_{lag}"] = df["daily_min_price"].shift(lag)

    df = df.dropna().reset_index(drop=True)
    return df


def build_daily_xgb_forecast(df_daily: pd.DataFrame, horizon_days: int = 90):
    df = add_daily_features(df_daily)

    if len(df) <= horizon_days + 60:
        raise ValueError(f"Not enough daily rows after feature engineering. Rows={len(df)}")

    target = "daily_avg_price"
    feature_cols = [c for c in df.columns if c not in ["date", target]]

    train = df.iloc[:-horizon_days].copy()
    test = df.iloc[-horizon_days:].copy()

    X_train, y_train = train[feature_cols], train[target]
    X_test, y_test = test[feature_cols], test[target]

    model = XGBRegressor(
        n_estimators=900,
        max_depth=8,
        learning_rate=0.03,
        subsample=0.9,
        colsample_bytree=0.9,
        objective="reg:squarederror",
        random_state=42,
        n_jobs=-1,
    )
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    print_reg_metrics("XGB Daily Avg (Improved)", y_test, y_pred)

    test_result = test[["date", target]].copy()
    test_result["y_pred"] = y_pred
    test_result["abs_error"] = (test_result[target] - test_result["y_pred"]).abs()
    test_result["error"] = test_result[target] - test_result["y_pred"]

    return model, test_result, feature_cols



def prepare_intraday_subset(df_qh: pd.DataFrame) -> pd.DataFrame:
    """
    Keep ONLY numeric + required columns to avoid object dtypes (e.g., day_name).
    """
    df = df_qh[["datetime", "date", "period", "price_eur_mwh"]].copy()

    df["datetime"] = pd.to_datetime(df["datetime"])
    df["date"] = pd.to_datetime(df["date"]).dt.date
    df["period"] = pd.to_numeric(df["period"], errors="coerce")
    df["price_eur_mwh"] = pd.to_numeric(df["price_eur_mwh"], errors="coerce")

    df = df.dropna(subset=["datetime", "date", "period", "price_eur_mwh"]).reset_index(drop=True)

    counts = df.groupby("date")["period"].nunique()
    full_days = counts[counts == 96].index
    subset = df[df["date"].isin(full_days)].copy()
    subset = subset.sort_values("datetime").reset_index(drop=True)

    subset["t_in_day"] = subset.groupby("date").cumcount()
    subset["t_norm"] = subset["t_in_day"] / subset["t_in_day"].max()

    subset["sin_time"] = np.sin(2 * np.pi * subset["t_norm"])
    subset["cos_time"] = np.cos(2 * np.pi * subset["t_norm"])

    for lag in [1, 2, 4, 96]:
        subset[f"price_lag_{lag}"] = subset["price_eur_mwh"].shift(lag)

    subset["dow"] = subset["datetime"].dt.weekday
    subset["is_weekend"] = (subset["dow"] >= 5).astype(int)
    subset["hour"] = subset["datetime"].dt.hour

    subset = subset.dropna().reset_index(drop=True)
    return subset


def build_intraday_xgb_forecast(df_qh: pd.DataFrame, test_size_days: int = 7):
    subset = prepare_intraday_subset(df_qh)

    unique_days = pd.Series(subset["date"].unique()).sort_values().reset_index(drop=True)
    if len(unique_days) <= test_size_days + 5:
        raise ValueError("Not enough full 96-period days for intraday test.")

    split_date = unique_days.iloc[-test_size_days]

    train = subset[subset["date"] < split_date].copy()
    test = subset[subset["date"] >= split_date].copy()

    target = "price_eur_mwh"

    
    feature_cols = [c for c in subset.columns if c not in ["date", "datetime", target]]

    X_train = train[feature_cols]
    X_test = test[feature_cols]

    
    obj_cols = X_train.select_dtypes(include=["object"]).columns.tolist()
    if obj_cols:
        X_train = X_train.drop(columns=obj_cols)
        X_test = X_test.drop(columns=obj_cols)
        feature_cols = [c for c in feature_cols if c not in obj_cols]

    
    X_train = X_train.astype(float)
    X_test = X_test.astype(float)

    y_train, y_test = train[target], test[target]

    model = XGBRegressor(
        n_estimators=600,
        max_depth=8,
        learning_rate=0.05,
        subsample=0.9,
        colsample_bytree=0.9,
        objective="reg:squarederror",
        random_state=42,
        n_jobs=-1,
    )
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    print_reg_metrics("XGB 15-min", y_test, y_pred)

    result = test[["datetime", "date", target]].copy()
    result["y_pred"] = y_pred
    result["abs_error"] = (result[target] - result["y_pred"]).abs()
    result["error"] = result[target] - result["y_pred"]

    return model, result, feature_cols



def seasonality_summaries(df_daily: pd.DataFrame, df_qh: pd.DataFrame):
    d = df_daily[["date", "daily_avg_price"]].copy()
    d["date"] = pd.to_datetime(d["date"])

    monthly = (
        d.groupby(d["date"].dt.to_period("M"))["daily_avg_price"]
        .mean()
        .reset_index()
        .rename(columns={"date": "year_month", "daily_avg_price": "monthly_avg"})
    )
    monthly["year_month"] = monthly["year_month"].astype(str)

    d["day_of_week"] = d["date"].dt.weekday
    dow = (
        d.groupby("day_of_week")["daily_avg_price"]
        .agg(["mean", "min", "max"])
        .reset_index()
    )

    q = df_qh.copy()
    q["hour"] = pd.to_datetime(q["datetime"]).dt.hour
    intraday = (
        q.groupby("hour")["price_eur_mwh"]
        .agg(["mean", "min", "max"])
        .reset_index()
    )

    return monthly, dow, intraday



def build_peak_probability_model(df_daily: pd.DataFrame, quantile: float = 0.80):
    df_feat = add_daily_features(df_daily)

    threshold = df_feat["daily_max_price"].quantile(quantile)
    df_feat["is_peak_day"] = (df_feat["daily_max_price"] >= threshold).astype(int)

    feature_cols = [c for c in df_feat.columns if c not in ["date", "is_peak_day", "daily_max_price"]]

    X = df_feat[feature_cols]
    y = df_feat["is_peak_day"]

    if y.nunique() < 2:
        raise ValueError("Peak label has only one class. Lower quantile or check data.")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, shuffle=True, stratify=y, random_state=42
    )

    pos = int(y.sum())
    neg = int(len(y) - pos)
    scale_pos_weight = (neg / pos) if pos > 0 else 1.0

    model = XGBClassifier(
        n_estimators=650,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.9,
        colsample_bytree=0.9,
        eval_metric="logloss",
        scale_pos_weight=scale_pos_weight,
        random_state=42,
        n_jobs=-1,
    )
    model.fit(X_train, y_train)

    y_proba = model.predict_proba(X_test)[:, 1]
    auc = float(roc_auc_score(y_test, y_proba))

    print(f"[Peak Model] q={quantile:.2f} | threshold≈{threshold:.2f} €/MWh | ROC-AUC={auc:.3f} | peak_rate={y.mean()*100:.1f}%")

    df_last = df_feat.sort_values("date").tail(90).copy()
    df_last["peak_probability"] = model.predict_proba(df_last[feature_cols])[:, 1]
    peak_result = df_last[["date", "is_peak_day", "peak_probability"]].copy()

    imp_gain = model.get_booster().get_score(importance_type="gain")
    df_imp = (
        pd.DataFrame({"feature": list(imp_gain.keys()), "importance": list(imp_gain.values())})
        .sort_values("importance", ascending=False)
        .reset_index(drop=True)
    )

    return model, peak_result, feature_cols, df_imp



def export_all(
    daily_xgb_test: pd.DataFrame,
    intraday_test: pd.DataFrame,
    peak_result: pd.DataFrame,
    monthly: pd.DataFrame,
    dow: pd.DataFrame,
    intrah: pd.DataFrame,
    prophet_test: pd.DataFrame | None = None,
    prophet_full: pd.DataFrame | None = None,
    peak_importance: pd.DataFrame | None = None,
):
    daily_xgb_test.to_csv(OUTPUT_DIR / "daily_forecast_xgb.csv", index=False)
    intraday_test.to_csv(OUTPUT_DIR / "intraday_forecast_xgb.csv", index=False)
    peak_result.to_csv(OUTPUT_DIR / "peak_probabilities.csv", index=False)

    monthly.to_csv(OUTPUT_DIR / "seasonality_monthly.csv", index=False)
    dow.to_csv(OUTPUT_DIR / "seasonality_day_of_week.csv", index=False)
    intrah.to_csv(OUTPUT_DIR / "seasonality_intraday_hour.csv", index=False)

    if prophet_test is not None:
        prophet_test.to_csv(OUTPUT_DIR / "daily_forecast_prophet_test.csv", index=False)
    if prophet_full is not None:
        prophet_full.to_csv(OUTPUT_DIR / "daily_forecast_prophet_full.csv", index=False)

    if peak_importance is not None and not peak_importance.empty:
        peak_importance.to_csv(OUTPUT_DIR / "peak_feature_importance.csv", index=False)

    print(f"✔️ Outputs exportados para: {OUTPUT_DIR}")



if __name__ == "__main__":
    print(">>> Loading data")
    daily, qh = load_data()

    print("\n>>> Daily forecast with Prophet")
    prophet_model = None
    prophet_forecast = None
    prophet_test = None
    try:
        prophet_model, prophet_forecast, prophet_test = build_daily_prophet_model(daily, horizon_days=HORIZON_DAYS)
    except Exception as e:
        print(f"⚠️ Prophet failed: {e}")

    print("\n>>> Daily forecast with XGBoost (Improved)")
    xgb_daily_model, xgb_daily_test, daily_features = build_daily_xgb_forecast(daily, horizon_days=HORIZON_DAYS)

    print("\n>>> Intraday (15-min) forecast with XGBoost")
    xgb_intraday_model, intraday_result, intraday_features = build_intraday_xgb_forecast(qh, test_size_days=INTRADAY_TEST_DAYS)

    print("\n>>> Seasonality summaries")
    monthly_seasonality, dow_seasonality, intraday_seasonality = seasonality_summaries(daily, qh)

    print("\n>>> Peak day probability model + explainability")
    peak_model, peak_result, peak_features, peak_importance = build_peak_probability_model(daily, quantile=PEAK_QUANTILE)

    print("\n>>> Exporting outputs")
    export_all(
        daily_xgb_test=xgb_daily_test,
        intraday_test=intraday_result,
        peak_result=peak_result,
        monthly=monthly_seasonality,
        dow=dow_seasonality,
        intrah=intraday_seasonality,
        prophet_test=prophet_test,
        prophet_full=prophet_forecast,
        peak_importance=peak_importance,
    )

    print("\nAll done.")

