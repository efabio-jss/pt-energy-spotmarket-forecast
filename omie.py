import os
import datetime as dt
from pathlib import Path

import requests
import pandas as pd


START_DATE = dt.date(2023, 1, 1)
END_DATE = dt.date.today() - dt.timedelta(days=1)



BASE_DIR = Path(
    r" your_dir"
) #imput your dir
RAW_DIR = BASE_DIR / "raw"
PROCESSED_DIR = BASE_DIR / "processed"

RAW_DIR.mkdir(parents=True, exist_ok=True)
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)


DOWNLOAD_BASE_URL = "https://www.omie.es/pt/file-download"

HEADERS = {
    "User-Agent": "Mozilla/5.0 (compatible; Fabio-OMIE-Scraper/1.0; +https://www.omie.es)"
}




def daterange(start_date: dt.date, end_date: dt.date):
    """Gera todas as datas entre start_date e end_date (incluindo ambas)."""
    current = start_date
    while current <= end_date:
        yield current
        current += dt.timedelta(days=1)


def build_filename_for_date(d: dt.date) -> str:
    """Constrói o nome do ficheiro OMIE para uma dada data."""
    return f"marginalpdbcpt_{d:%Y%m%d}.1"


def download_file_for_date(d: dt.date, overwrite: bool = False) -> Path | None:
    """
    Faz download do ficheiro para a data d.
    Se não existir ficheiro (404), devolve None.
    """
    filename = build_filename_for_date(d)
    dest_path = RAW_DIR / filename

    if dest_path.exists() and not overwrite:
        
        print(f"   ✔️ Já existe localmente: {filename}")
        return dest_path

    params = {
        "parents": "marginalpdbcpt",
        "filename": filename,
    }

    print(f"   ⬇️  A descarregar {filename} ...")
    resp = requests.get(DOWNLOAD_BASE_URL, headers=HEADERS, params=params, timeout=60)

    if resp.status_code == 404:
        print(f"   ⚠️ Ficheiro não encontrado na OMIE para {d} (404).")
        return None

    resp.raise_for_status()

    with open(dest_path, "wb") as f:
        f.write(resp.content)

    return dest_path


def download_all_files(start: dt.date, end: dt.date):
    """Faz download de todos os ficheiros no intervalo de datas."""
    downloaded_paths = []

    print(f"➜ A descarregar ficheiros de {start} a {end}...")
    for d in daterange(start, end):
        try:
            path = download_file_for_date(d)
            if path is not None:
                downloaded_paths.append(path)
        except Exception as e:
            print(f"   ❌ Erro ao descarregar {d}: {e}")

    print(f"✔️ Download concluído. {len(downloaded_paths)} ficheiros disponíveis.")
    return downloaded_paths




def parse_pdbcpt_file(path: Path) -> pd.DataFrame:
    """
    Lê um ficheiro marginalpdbcpt_YYYYMMDD.1 e devolve DataFrame com:
    - datetime (timestamp quarto-horário)
    - price_eur_mwh
    - várias features temporais
    """

    

    df = pd.read_csv(
        path,
        sep=";",
        header=None,
        skiprows=1,        
        skipfooter=1,      
        engine="python",   
        encoding="latin1"
    )

    
    df = df.rename(
        columns={
            0: "year",
            1: "month",
            2: "day",
            3: "period",            
            4: "price_eur_mwh",
            5: "price2_eur_mwh",    
            6: "empty"              
        }
    )

    
    if "empty" in df.columns:
        df = df.drop(columns=["empty"])

    
    df["date"] = pd.to_datetime(df[["year", "month", "day"]])

    
    df["minutes_since_midnight"] = (df["period"] - 1) * 15
    df["datetime"] = df["date"] + pd.to_timedelta(df["minutes_since_midnight"], unit="m")

    
    df["year"] = df["datetime"].dt.year
    df["month"] = df["datetime"].dt.month
    df["day"] = df["datetime"].dt.day
    df["hour"] = df["datetime"].dt.hour
    df["minute"] = df["datetime"].dt.minute
    df["quarter_in_hour"] = (df["period"] - 1) % 4  
    df["day_of_week"] = df["datetime"].dt.weekday  
    df["day_name"] = df["datetime"].dt.day_name()
    df["week_of_year"] = df["datetime"].dt.isocalendar().week.astype(int)
    df["is_weekend"] = df["day_of_week"] >= 5

    
    df["is_peak_hour"] = df["hour"].between(8, 19)      
    df["is_offpeak_hour"] = ~df["is_peak_hour"]
    df["is_night"] = df["hour"].between(0, 6)
    df["is_evening"] = df["hour"].between(18, 23)

    return df




def build_quarter_hour_dataset(file_paths: list[Path]) -> pd.DataFrame:
    """Lê todos os ficheiros e devolve um único DataFrame quarto-horário."""
    all_dfs = []

    print("➜ A processar ficheiros PDBCPT...")
    for path in sorted(file_paths):
        try:
            print(f"   🧾 {path.name}")
            df = parse_pdbcpt_file(path)
            if not df.empty:
                all_dfs.append(df)
        except Exception as e:
            print(f"   ❌ Erro ao processar {path.name}: {e}")

    if not all_dfs:
        raise RuntimeError("Não foi possível construir o dataset (todos os ficheiros falharam).")

    full_df = pd.concat(all_dfs, ignore_index=True)
    full_df = full_df.sort_values("datetime").reset_index(drop=True)

    return full_df




def create_daily_metrics(df: pd.DataFrame) -> pd.DataFrame:
    """
    Cria tabela diária com KPIs e features para previsão.
    """
    
    df["price_peak"] = df["price_eur_mwh"].where(df["is_peak_hour"])
    df["price_offpeak"] = df["price_eur_mwh"].where(df["is_offpeak_hour"])
    df["price_night"] = df["price_eur_mwh"].where(df["is_night"])
    df["price_evening"] = df["price_eur_mwh"].where(df["is_evening"])

    daily = (
        df.groupby("date")
        .agg(
            year=("year", "first"),
            month=("month", "first"),
            week_of_year=("week_of_year", "first"),
            day_of_week=("day_of_week", "first"),
            day_name=("day_name", "first"),
            is_weekend=("is_weekend", "first"),

            
            daily_avg_price=("price_eur_mwh", "mean"),
            daily_min_price=("price_eur_mwh", "min"),
            daily_max_price=("price_eur_mwh", "max"),
            daily_std_price=("price_eur_mwh", "std"),
            daily_median_price=("price_eur_mwh", "median"),

            
            daily_peak_avg=("price_peak", "mean"),
            daily_offpeak_avg=("price_offpeak", "mean"),
            daily_night_avg=("price_night", "mean"),
            daily_evening_avg=("price_evening", "mean"),

            
            trading_periods=("price_eur_mwh", "count"),
        )
        .reset_index()
    )

    
    daily["daily_price_range"] = daily["daily_max_price"] - daily["daily_min_price"]
    daily["peak_offpeak_spread"] = daily["daily_peak_avg"] - daily["daily_offpeak_avg"]

    
    daily = daily.sort_values("date").reset_index(drop=True)

    daily["daily_avg_change_pct"] = daily["daily_avg_price"].pct_change() * 100
    daily["daily_max_change_pct"] = daily["daily_max_price"].pct_change() * 100

    daily["avg_price_rolling_7d"] = daily["daily_avg_price"].rolling(7).mean()
    daily["avg_price_rolling_30d"] = daily["daily_avg_price"].rolling(30).mean()

    daily["volatility_rolling_7d"] = daily["daily_std_price"].rolling(7).mean()
    daily["volatility_rolling_30d"] = daily["daily_std_price"].rolling(30).mean()

    
    daily["is_above_30d_avg"] = daily["daily_avg_price"] > daily["avg_price_rolling_30d"]
    daily["is_extreme_high_day"] = daily["daily_max_price"] > (
        daily["daily_avg_price"] + 2 * daily["daily_std_price"]
    )
    daily["is_extreme_low_day"] = daily["daily_min_price"] < (
        daily["daily_avg_price"] - 2 * daily["daily_std_price"]
    )

    return daily


def create_monthly_metrics(daily: pd.DataFrame) -> pd.DataFrame:
    """
    Cria tabela mensal agregada, ideal para dashboards executivos.
    """
    daily["year_month"] = daily["date"].dt.to_period("M")

    monthly = (
        daily.groupby("year_month")
        .agg(
            year=("year", "first"),
            month=("month", "first"),

            
            monthly_avg_daily_price=("daily_avg_price", "mean"),
            monthly_min_daily_price=("daily_min_price", "min"),
            monthly_max_daily_price=("daily_max_price", "max"),
            monthly_avg_peak_price=("daily_peak_avg", "mean"),
            monthly_avg_offpeak_price=("daily_offpeak_avg", "mean"),

            
            monthly_avg_daily_volatility=("daily_std_price", "mean"),
            monthly_price_range_avg=("daily_price_range", "mean"),

            
            trading_days=("date", "count"),
        )
        .reset_index()
    )

    
    monthly["month_start_date"] = monthly["year_month"].dt.to_timestamp()

    
    monthly = monthly.sort_values("month_start_date").reset_index(drop=True)
    monthly["mom_change_avg_price_pct"] = (
        monthly["monthly_avg_daily_price"].pct_change() * 100
    )
    monthly["mom_change_peak_price_pct"] = (
        monthly["monthly_avg_peak_price"].pct_change() * 100
    )

    return monthly


def enrich_quarter_hour_with_daily(df: pd.DataFrame, daily: pd.DataFrame) -> pd.DataFrame:
    """
    Faz merge dos KPIs diários no dataset quarto-horário,
    para poderes usar tudo no mesmo modelo/relatório se quiseres.
    """
    daily_cols_to_merge = [
        "date",
        "daily_avg_price",
        "daily_min_price",
        "daily_max_price",
        "daily_median_price",
        "daily_std_price",
        "daily_price_range",
        "daily_peak_avg",
        "daily_offpeak_avg",
        "daily_night_avg",
        "daily_evening_avg",
        "daily_avg_change_pct",
        "avg_price_rolling_7d",
        "avg_price_rolling_30d",
        "volatility_rolling_7d",
        "volatility_rolling_30d",
        "is_above_30d_avg",
        "is_extreme_high_day",
        "is_extreme_low_day",
    ]

    df = df.merge(
        daily[daily_cols_to_merge],
        on="date",
        how="left",
        validate="many_to_one"
    )

    
    drop_cols = [c for c in df.columns if c.startswith("price_") and c not in ("price_eur_mwh", "price2_eur_mwh")]
    df = df.drop(columns=drop_cols, errors="ignore")

    return df




def export_datasets(df_qh: pd.DataFrame, df_daily: pd.DataFrame, df_monthly: pd.DataFrame):
    """
    Grava datasets em CSV e Parquet, organizados para análise,
    modelos e dashboards (Power BI / Streamlit).
    """
    qh_csv = PROCESSED_DIR / "pdbcpt_quarter_hour_2023_onwards.csv"
    qh_parquet = PROCESSED_DIR / "pdbcpt_quarter_hour_2023_onwards.parquet"

    daily_csv = PROCESSED_DIR / "pdbcpt_daily_2023_onwards.csv"
    daily_parquet = PROCESSED_DIR / "pdbcpt_daily_2023_onwards.parquet"

    monthly_csv = PROCESSED_DIR / "pdbcpt_monthly_2023_onwards.csv"
    monthly_parquet = PROCESSED_DIR / "pdbcpt_monthly_2023_onwards.parquet"

    df_qh.to_csv(qh_csv, index=False)
    df_qh.to_parquet(qh_parquet, index=False)

    df_daily.to_csv(daily_csv, index=False)
    df_daily.to_parquet(daily_parquet, index=False)

    df_monthly.to_csv(monthly_csv, index=False)
    df_monthly.to_parquet(monthly_parquet, index=False)

    print("✔️ Ficheiros exportados em:")
    print(f"   - Quarter-hour: {qh_csv}")
    print(f"   - Daily       : {daily_csv}")
    print(f"   - Monthly     : {monthly_csv}")




if __name__ == "__main__":
    
    file_paths = download_all_files(START_DATE, END_DATE)

    if not file_paths:
        raise SystemExit("Nenhum ficheiro disponível. Verifica datas e ligação.")

    
    df_qh = build_quarter_hour_dataset(file_paths)

    
    df_daily = create_daily_metrics(df_qh)
    df_monthly = create_monthly_metrics(df_daily)

    
    df_qh = enrich_quarter_hour_with_daily(df_qh, df_daily)

    
    export_datasets(df_qh, df_daily, df_monthly)

    print("✅ Pipeline OMIE PDBCPT concluído com sucesso.")
