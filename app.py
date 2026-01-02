import datetime as dt
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

try:
    import altair as alt
    ALTAIR_OK = True
except Exception:
    ALTAIR_OK = False



st.set_page_config(
    page_title="OMIE Intelligence",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded",
)


st.markdown(
    """
<style>
html, body, [class*="css"]  { font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Inter, Roboto, Arial, sans-serif; }
:root {
  --bg: #070A12;
  --panel: rgba(255,255,255,0.06);
  --panel2: rgba(255,255,255,0.09);
  --text: rgba(255,255,255,0.92);
  --muted: rgba(255,255,255,0.62);
  --accent: #7C5CFF;
  --accent2: #21D4FD;
  --good: #2EE59D;
  --warn: #FFB020;
  --bad: #FF4D6D;
  --border: rgba(255,255,255,0.12);
}

.stApp {
  background: radial-gradient(1200px 700px at 15% 10%, rgba(124,92,255,0.22), transparent 55%),
              radial-gradient(900px 600px at 90% 15%, rgba(33,212,253,0.18), transparent 55%),
              radial-gradient(900px 700px at 40% 100%, rgba(46,229,157,0.10), transparent 55%),
              var(--bg);
  color: var(--text);
}

section[data-testid="stSidebar"] {
  background: rgba(255,255,255,0.04);
  border-right: 1px solid var(--border);
}
section[data-testid="stSidebar"] * { color: var(--text) !important; }

h1, h2, h3 { letter-spacing: -0.02em; }
small, .muted { color: var(--muted); }

.card {
  background: linear-gradient(180deg, rgba(255,255,255,0.08), rgba(255,255,255,0.05));
  border: 1px solid var(--border);
  border-radius: 18px;
  padding: 18px 18px;
  box-shadow: 0 10px 30px rgba(0,0,0,0.25);
}
.card-title { color: var(--muted); font-size: 12px; text-transform: uppercase; letter-spacing: .12em; }
.card-value { font-size: 28px; font-weight: 650; margin-top: 2px; }
.card-sub { color: var(--muted); font-size: 13px; margin-top: 8px; }

.hr { height: 1px; background: var(--border); margin: 12px 0 18px 0; }

.badge {
  display:inline-block;
  padding: 6px 10px;
  border-radius: 999px;
  border: 1px solid var(--border);
  background: rgba(255,255,255,0.05);
  color: var(--muted);
  font-size: 12px;
}

[data-testid="stDataFrame"] { border: 1px solid var(--border); border-radius: 14px; overflow: hidden; }

.stButton button {
  border-radius: 12px;
  border: 1px solid var(--border);
  background: rgba(255,255,255,0.06);
}

</style>
""",
    unsafe_allow_html=True,
)


def eur_mwh(x):
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return "—"
    return f"{x:,.2f} €/MWh".replace(",", "X").replace(".", ",").replace("X", ".")

def pct(x):
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return "—"
    return f"{x*100:.1f}%"

def kpi_card(title, value, subtitle=""):
    st.markdown(
        f"""
<div class="card">
  <div class="card-title">{title}</div>
  <div class="card-value">{value}</div>
  <div class="card-sub">{subtitle}</div>
</div>
""",
        unsafe_allow_html=True,
    )

def safe_read_csv(path: Path, parse_dates=None):
    if not path.exists():
        return None
    try:
        return pd.read_csv(path, parse_dates=parse_dates)
    except Exception as e:
        st.error(f"Falha a ler: {path}\n{e}")
        return None

@st.cache_data(show_spinner=False)
def load_all(data_dir: Path):
    out = {}
    out["daily_xgb"] = safe_read_csv(data_dir / "daily_forecast_xgb.csv", parse_dates=["date"])
    out["daily_prophet_test"] = safe_read_csv(data_dir / "daily_forecast_prophet_test.csv", parse_dates=["ds"])
    out["intraday_xgb"] = safe_read_csv(data_dir / "intraday_forecast_xgb.csv", parse_dates=["datetime", "date"])
    out["peaks"] = safe_read_csv(data_dir / "peak_probabilities.csv", parse_dates=["date"])
    out["s_monthly"] = safe_read_csv(data_dir / "seasonality_monthly.csv")
    out["s_dow"] = safe_read_csv(data_dir / "seasonality_day_of_week.csv")
    out["s_intrah"] = safe_read_csv(data_dir / "seasonality_intraday_hour.csv")
    
    out["peak_importance"] = safe_read_csv(data_dir / "peak_feature_importance.csv")
    return out

def ensure_altair():
    if not ALTAIR_OK:
        return
    alt.themes.enable("default")

def layered_real_vs_pred(df, x, real_col, pred_col, title, height=340):
    if not ALTAIR_OK:
        st.line_chart(df.set_index(x)[[real_col, pred_col]])
        return
    ensure_altair()
    d = df[[x, real_col, pred_col]].copy()
    d = d.rename(columns={real_col: "Real", pred_col: "Previsto"})
    d_long = d.melt(x, var_name="Serie", value_name="Preço")
    chart = alt.Chart(d_long).mark_line().encode(
        x=alt.X(x, title=None),
        y=alt.Y("Preço:Q", title=None),
        color=alt.Color("Serie:N", legend=alt.Legend(title=None)),
        tooltip=[x, "Serie", alt.Tooltip("Preço:Q", format=".2f")]
    ).properties(height=height, title=title)
    st.altair_chart(chart, use_container_width=True)

def bar_chart(df, x, y, title, height=320):
    if not ALTAIR_OK:
        st.bar_chart(df.set_index(x)[y])
        return
    ensure_altair()
    chart = alt.Chart(df).mark_bar().encode(
        x=alt.X(x, title=None),
        y=alt.Y(y, title=None),
        tooltip=[x, alt.Tooltip(y, format=".4f")]
    ).properties(height=height, title=title)
    st.altair_chart(chart, use_container_width=True)

def line_chart(df, x, y, title, height=320):
    if not ALTAIR_OK:
        st.line_chart(df.set_index(x)[y])
        return
    ensure_altair()
    chart = alt.Chart(df).mark_line().encode(
        x=alt.X(x, title=None),
        y=alt.Y(y, title=None),
        tooltip=[x, alt.Tooltip(y, format=".2f")]
    ).properties(height=height, title=title)
    st.altair_chart(chart, use_container_width=True)

def hist_chart(df, col, title, height=320, bins=50):
    if not ALTAIR_OK:
        st.bar_chart(np.histogram(df[col].dropna(), bins=bins)[0])
        return
    ensure_altair()
    chart = alt.Chart(df).mark_bar().encode(
        alt.X(col, bin=alt.Bin(maxbins=bins), title=None),
        y=alt.Y("count()", title=None),
        tooltip=[alt.Tooltip("count()", title="Count")]
    ).properties(height=height, title=title)
    st.altair_chart(chart, use_container_width=True)

def download_csv_button(df: pd.DataFrame, filename: str, label: str):
    csv_bytes = df.to_csv(index=False).encode("utf-8")
    st.download_button(
        label=label,
        data=csv_bytes,
        file_name=filename,
        mime="text/csv",
        use_container_width=True,
    )


st.sidebar.markdown("## ⚡ OMIE Intelligence")
st.sidebar.markdown('<div class="badge">Apple Night • Executive-grade</div>', unsafe_allow_html=True)
st.sidebar.markdown("")

default_dir = Path(r" your_dir ") #imput your dir
data_dir_txt = st.sidebar.text_input("Pasta dos outputs", value=str(default_dir))
DATA_DIR = Path(data_dir_txt)

data = load_all(DATA_DIR)

page = st.sidebar.radio("Navegação", ["Exec", "Spot Market", "Forecast"], index=0)
st.sidebar.markdown("---")
st.sidebar.caption("Dark-only • sem Plotly • Altair opcional")


monthly = data.get("s_monthly")
dow = data.get("s_dow")
intrah = data.get("s_intrah")
peaks = data.get("peaks")
intraday = data.get("intraday_xgb")
daily_xgb = data.get("daily_xgb")
prophet_test = data.get("daily_prophet_test")
peak_imp = data.get("peak_importance")

today = dt.date.today()

if monthly is not None:
    monthly["year_month"] = monthly["year_month"].astype(str)
    monthly["year"] = monthly["year_month"].str.slice(0, 4).astype(int)
    monthly["month"] = monthly["year_month"].str.slice(5, 7).astype(int)


def executive_narrative(monthly_df, intraday_df, peaks_df):
    lines = []
    
    if monthly_df is not None and not monthly_df.empty:
        latest = monthly_df.sort_values(["year", "month"]).tail(1).iloc[0]
        y, m = int(latest["year"]), int(latest["month"])
        v = float(latest["monthly_avg"])

        prev = monthly_df.sort_values(["year", "month"]).tail(2)
        mom = None
        if len(prev) >= 2:
            prev_v = float(prev.iloc[0]["monthly_avg"])
            mom = (v / prev_v - 1) if prev_v else np.nan

        
        yoy = None
        same_last = monthly_df[(monthly_df["year"] == y - 1) & (monthly_df["month"] == m)]
        if not same_last.empty:
            yoy_v = float(same_last["monthly_avg"].iloc[0])
            yoy = (v / yoy_v - 1) if yoy_v else np.nan

        lines.append(f"**Regime mensal:** último mês ({y}-{m:02d}) com média de **{eur_mwh(v)}**.")
        if mom is not None and not np.isnan(mom):
            lines.append(f"• **MoM:** {pct(mom)} vs mês anterior.")
        if yoy is not None and not np.isnan(yoy):
            lines.append(f"• **YoY:** {pct(yoy)} vs mesmo mês do ano anterior.")

        
        year_df = monthly_df[monthly_df["year"] == y]
        if not year_df.empty:
            best = year_df.sort_values("monthly_avg").head(1).iloc[0]
            worst = year_df.sort_values("monthly_avg", ascending=False).head(1).iloc[0]
            lines.append(f"• **Melhor mês {y}:** {int(best['month']):02d} com **{eur_mwh(float(best['monthly_avg']))}**.")
            lines.append(f"• **Mês de stress {y}:** {int(worst['month']):02d} com **{eur_mwh(float(worst['monthly_avg']))}**.")

    
    if intraday_df is not None and not intraday_df.empty:
        d = intraday_df.copy()
        d["datetime"] = pd.to_datetime(d["datetime"])
        max_dt = d["datetime"].max()
        last7_start = max_dt - pd.Timedelta(days=7)
        prev7_start = max_dt - pd.Timedelta(days=14)

        w1 = d[(d["datetime"] >= last7_start)]
        w0 = d[(d["datetime"] >= prev7_start) & (d["datetime"] < last7_start)]

        if not w1.empty:
            avg1 = float(w1["price_eur_mwh"].mean())
            p951 = float(w1["price_eur_mwh"].quantile(0.95))
            max1 = float(w1["price_eur_mwh"].max())
            min1 = float(w1["price_eur_mwh"].min())

            lines.append("")
            lines.append(f"**Últimos 7 dias (15-min):** média **{eur_mwh(avg1)}**, P95 **{eur_mwh(p951)}**, range **{eur_mwh(min1)} → {eur_mwh(max1)}**.")

            if not w0.empty:
                avg0 = float(w0["price_eur_mwh"].mean())
                delta = (avg1 / avg0 - 1) if avg0 else np.nan
                if not np.isnan(delta):
                    lines.append(f"• **Δ semana:** {pct(delta)} vs semana anterior.")

    
    if peaks_df is not None and not peaks_df.empty:
        r = peaks_df.copy()
        r["date"] = pd.to_datetime(r["date"]).dt.date
        recent = r.sort_values("date").tail(14)
        if not recent.empty:
            high = recent[recent["peak_probability"] >= 0.7]
            lines.append("")
            lines.append(f"**Risco (14 dias):** {len(high)} dias com probabilidade ≥ **70%**.")
            if len(high) > 0:
                top = high.sort_values("peak_probability", ascending=False).head(3)
                tops = ", ".join([f"{row['date']} ({row['peak_probability']:.0%})" for _, row in top.iterrows()])
                lines.append(f"• **Top alertas:** {tops}.")

    if not lines:
        return "Sem dados suficientes para gerar narrative."
    return "\n".join(lines)


if page == "Exec":
    st.markdown("# Executive Overview")
    st.markdown('<span class="muted">KPIs anuais/mensais + narrativa automática (mês/semana)</span>', unsafe_allow_html=True)
    st.markdown('<div class="hr"></div>', unsafe_allow_html=True)

    if monthly is None:
        st.error("Não encontrei seasonality_monthly.csv. Preciso dele para a página Exec.")
        st.stop()

    years = sorted(monthly["year"].unique().tolist())
    colA, colB = st.columns([1.0, 1.0])
    with colA:
        year_sel = st.selectbox("Ano", years, index=len(years) - 1)
    with colB:
        compare_year = st.selectbox("Comparar com", years, index=max(0, len(years) - 2))

    def year_stats(y):
        d = monthly[monthly["year"] == y]
        return {
            "avg": float(d["monthly_avg"].mean()),
            "min": float(d["monthly_avg"].min()),
            "max": float(d["monthly_avg"].max()),
            "vol": float(d["monthly_avg"].std()),
        }

    s1 = year_stats(year_sel)
    s0 = year_stats(compare_year)
    yoy = (s1["avg"] / s0["avg"] - 1.0) if s0["avg"] else np.nan

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        kpi_card(f"Média anual ({year_sel})", eur_mwh(s1["avg"]), f"vs {compare_year}: {pct(yoy)}")
    with c2:
        kpi_card(f"Mínimo mensal ({year_sel})", eur_mwh(s1["min"]), "melhor mês (preço mais baixo)")
    with c3:
        kpi_card(f"Máximo mensal ({year_sel})", eur_mwh(s1["max"]), "mês de maior stress")
    with c4:
        kpi_card(f"Volatilidade mensal ({year_sel})", eur_mwh(s1["vol"]), "desvio padrão (mês a mês)")

    st.markdown("## Executive Narrative (auto)")
    narrative = executive_narrative(monthly, intraday, peaks)
    st.markdown(narrative)

    st.markdown("## Comparação mensal (2023/24/25)")
    df_cmp = monthly[monthly["year"].isin([compare_year, year_sel])].copy()
    df_cmp["Ano"] = df_cmp["year"].astype(str)
    df_cmp["Mes"] = df_cmp["month"].astype(int)

    if ALTAIR_OK:
        ensure_altair()
        chart = alt.Chart(df_cmp).mark_line(point=True).encode(
            x=alt.X("Mes:O", title=None),
            y=alt.Y("monthly_avg:Q", title=None),
            color=alt.Color("Ano:N", legend=alt.Legend(title=None)),
            tooltip=["Ano", "Mes", alt.Tooltip("monthly_avg:Q", format=".2f")]
        ).properties(height=360, title="Média mensal (€/MWh)")
        st.altair_chart(chart, use_container_width=True)
    else:
        pivot = df_cmp.pivot_table(index="Mes", columns="Ano", values="monthly_avg")
        st.line_chart(pivot)

    st.markdown("## Download")
    download_csv_button(
        df_cmp.sort_values(["Mes", "Ano"]),
        filename=f"exec_monthly_compare_{year_sel}_vs_{compare_year}.csv",
        label="⬇️ Download comparação mensal (CSV)"
    )


elif page == "Spot Market":
    st.markdown("# Spot Market Intelligence")
    st.markdown('<span class="muted">Day-of-week, intraday e distribuição (com download filtrado)</span>', unsafe_allow_html=True)
    st.markdown('<div class="hr"></div>', unsafe_allow_html=True)

    st.markdown("## Day-of-Week Pattern")
    if dow is None:
        st.warning("Não encontrei seasonality_day_of_week.csv.")
    else:
        dow2 = dow.copy()
        map_pt = {0:"Seg",1:"Ter",2:"Qua",3:"Qui",4:"Sex",5:"Sáb",6:"Dom"}
        dow2["DOW"] = dow2["day_of_week"].map(map_pt)
        c1, c2 = st.columns([1.25, 0.9])
        with c1:
            if ALTAIR_OK:
                ensure_altair()
                chart = alt.Chart(dow2).mark_bar().encode(
                    x=alt.X("DOW:N", sort=["Seg","Ter","Qua","Qui","Sex","Sáb","Dom"], title=None),
                    y=alt.Y("mean:Q", title=None),
                    tooltip=["DOW", alt.Tooltip("mean:Q", format=".2f")]
                ).properties(height=320, title="Preço médio por dia da semana (€/MWh)")
                st.altair_chart(chart, use_container_width=True)
            else:
                st.bar_chart(dow2.set_index("DOW")["mean"])
        with c2:
            cheapest = dow2.sort_values("mean").head(1).iloc[0]
            priciest = dow2.sort_values("mean", ascending=False).head(1).iloc[0]
            kpi_card("Dia mais barato", str(cheapest["DOW"]), eur_mwh(float(cheapest["mean"])))
            st.write("")
            kpi_card("Dia mais caro", str(priciest["DOW"]), eur_mwh(float(priciest["mean"])))

        download_csv_button(dow2[["day_of_week","DOW","mean","min","max"]], "spot_day_of_week.csv", "⬇️ Download day-of-week (CSV)")

    st.markdown("## Intraday Pattern (Hourly)")
    if intrah is None:
        st.warning("Não encontrei seasonality_intraday_hour.csv.")
    else:
        c1, c2 = st.columns([1.3, 0.9])
        with c1:
            line_chart(intrah, "hour", "mean", "Perfil intradiário médio (€/MWh)", height=340)
        with c2:
            low = intrah.sort_values("mean").head(1).iloc[0]
            high = intrah.sort_values("mean", ascending=False).head(1).iloc[0]
            kpi_card("Hora mais barata (média)", f"{int(low['hour']):02d}:00", eur_mwh(float(low["mean"])))
            st.write("")
            kpi_card("Hora mais cara (média)", f"{int(high['hour']):02d}:00", eur_mwh(float(high["mean"])))

        download_csv_button(intrah, "spot_intraday_hour.csv", "⬇️ Download intraday hourly (CSV)")

    st.markdown("## Distribuição (15-min) + Download filtrado")
    if intraday is None:
        st.warning("Não encontrei intraday_forecast_xgb.csv.")
    else:
        d = intraday.copy()
        d["datetime"] = pd.to_datetime(d["datetime"])
        d = d.sort_values("datetime")

        col1, col2, col3 = st.columns([1.0, 1.0, 1.2])
        with col1:
            last_days = st.slider("Janela (últimos dias)", 3, 90, 14, 1)
        with col2:
            cut_outliers = st.checkbox("Cortar outliers (1%-99%)", value=True)
        with col3:
            bins = st.slider("Bins histograma", 20, 120, 60, 5)

        max_dt = d["datetime"].max()
        min_dt = max_dt - pd.Timedelta(days=int(last_days))
        df_f = d[d["datetime"] >= min_dt].copy()

        if cut_outliers:
            lo, hi = df_f["price_eur_mwh"].quantile([0.01, 0.99]).values
            df_f = df_f[(df_f["price_eur_mwh"] >= lo) & (df_f["price_eur_mwh"] <= hi)].copy()

        c1, c2 = st.columns([1.2, 0.9])
        with c1:
            hist_chart(df_f, "price_eur_mwh", f"Distribuição 15-min — últimos {last_days} dias", height=340, bins=bins)
        with c2:
            kpi_card("Média", eur_mwh(float(df_f["price_eur_mwh"].mean())), "janela selecionada")
            st.write("")
            kpi_card("P95", eur_mwh(float(df_f["price_eur_mwh"].quantile(0.95))), "top 5% (stress)")
            st.write("")
            kpi_card("Min / Max", f"{eur_mwh(float(df_f['price_eur_mwh'].min()))} / {eur_mwh(float(df_f['price_eur_mwh'].max()))}", "range")

        download_csv_button(df_f, f"spot_intraday_filtered_last_{last_days}d.csv", "⬇️ Download intraday filtrado (CSV)")


else:
    st.markdown("# Forecast & Risk")
    st.markdown('<span class="muted">Real vs previsto, erro, alertas e explainability</span>', unsafe_allow_html=True)
    st.markdown('<div class="hr"></div>', unsafe_allow_html=True)

    
    st.markdown("## Alertas (Peak Risk ≥ 70%)")

    if peaks is None:
        st.warning("Não encontrei peak_probabilities.csv.")
    else:
        r = peaks.copy()
        r["date"] = pd.to_datetime(r["date"]).dt.date
        r = r.sort_values("date")

        
        future = r[r["date"] > today].copy()
        if not future.empty:
            alerts = future[future["peak_probability"] >= 0.7].sort_values("peak_probability", ascending=False)
            if alerts.empty:
                st.success("✅ Sem alertas críticos (≥70%) nos próximos dias disponíveis.")
            else:
                st.error(f"⚠️ {len(alerts)} alertas críticos (≥70%) nos próximos dias disponíveis.")
                st.dataframe(alerts[["date","peak_probability","is_peak_day"]], use_container_width=True)
                download_csv_button(alerts, "alerts_peak_risk_future.csv", "⬇️ Download alertas futuros (CSV)")
        else:
            
            last = r.tail(14).copy()
            alerts = last[last["peak_probability"] >= 0.7].sort_values("peak_probability", ascending=False)
            if alerts.empty:
                st.success("✅ Sem alertas críticos (≥70%) nos últimos 14 dias.")
            else:
                st.warning(f"⚠️ {len(alerts)} dias com risco ≥70% nos últimos 14 dias (não há datas futuras no ficheiro).")
                st.dataframe(alerts[["date","peak_probability","is_peak_day"]], use_container_width=True)
                download_csv_button(alerts, "alerts_peak_risk_last14d.csv", "⬇️ Download alertas (últimos 14 dias)")

    st.markdown("")

    
    st.markdown("## Daily Forecast (XGBoost)")
    if daily_xgb is None:
        st.warning("Não encontrei daily_forecast_xgb.csv.")
    else:
        d = daily_xgb.copy()
        d["date"] = pd.to_datetime(d["date"])
        d = d.sort_values("date")
        d["abs_error"] = (d["daily_avg_price"] - d["y_pred"]).abs()
        d["error"] = d["daily_avg_price"] - d["y_pred"]

        c1, c2, c3, c4 = st.columns(4)
        with c1:
            kpi_card("MAE", eur_mwh(float(d["abs_error"].mean())), "erro médio absoluto")
        with c2:
            rmse = float(np.sqrt(np.mean(d["error"] ** 2)))
            kpi_card("RMSE", eur_mwh(rmse), "penaliza erros grandes")
        with c3:
            kpi_card("P95 |Erro|", eur_mwh(float(d["abs_error"].quantile(0.95))), "dias difíceis")
        with c4:
            kpi_card("Std previsão", eur_mwh(float(d["y_pred"].std())), "se ~0 → previsão constante")

        layered_real_vs_pred(
            d,
            x="date",
            real_col="daily_avg_price",
            pred_col="y_pred",
            title="Real vs Previsto (XGB) — janela de teste",
            height=360,
        )

        
        st.markdown("### Download")
        download_csv_button(d, "forecast_daily_xgb.csv", "⬇️ Download daily forecast (CSV)")

    
    st.markdown("## Intraday Forecast (15-min) — (forte)")
    if intraday is None:
        st.warning("Não encontrei intraday_forecast_xgb.csv.")
    else:
        d = intraday.copy()
        d["datetime"] = pd.to_datetime(d["datetime"])
        d = d.sort_values("datetime")
        d["abs_error"] = (d["price_eur_mwh"] - d["y_pred"]).abs()
        d["error"] = d["price_eur_mwh"] - d["y_pred"]

        unique_dates = sorted(pd.to_datetime(d["date"]).dt.date.unique().tolist())
        col1, col2 = st.columns([1.2, 1.0])
        with col1:
            pick = st.selectbox("Dia para inspeção (15-min)", unique_dates, index=len(unique_dates) - 1)
        with col2:
            show_error = st.checkbox("Mostrar erro", value=False)

        dd = d[pd.to_datetime(d["date"]).dt.date == pick].copy()

        c1, c2, c3, c4 = st.columns(4)
        with c1:
            kpi_card("MAE (dia)", eur_mwh(float(dd["abs_error"].mean())), "qualidade intradiária")
        with c2:
            rmse = float(np.sqrt(np.mean(dd["error"] ** 2)))
            kpi_card("RMSE (dia)", eur_mwh(rmse), "picos")
        with c3:
            kpi_card("P95 |Erro|", eur_mwh(float(dd["abs_error"].quantile(0.95))), "stress")
        with c4:
            kpi_card("Max real", eur_mwh(float(dd["price_eur_mwh"].max())), "stress intradiário")

        layered_real_vs_pred(
            dd,
            x="datetime",
            real_col="price_eur_mwh",
            pred_col="y_pred",
            title=f"Real vs Previsto — {pick} (15-min)",
            height=360,
        )

        if show_error:
            line_chart(dd, "datetime", "error", "Erro intradiário (Real - Previsto)", height=220)

        st.markdown("### Download")
        download_csv_button(dd, f"forecast_intraday_{pick}.csv", f"⬇️ Download intraday ({pick})")

    
    st.markdown("## Peak-Day Probabilities (90d) + Download")
    if peaks is None:
        st.warning("Não encontrei peak_probabilities.csv.")
    else:
        r = peaks.copy()
        r["date"] = pd.to_datetime(r["date"])
        r = r.sort_values("date")

        c1, c2, c3, c4 = st.columns(4)
        with c1:
            kpi_card("Taxa de pico", pct(float(r["is_peak_day"].mean())), "janela do ficheiro")
        with c2:
            kpi_card("Prob. média", pct(float(r["peak_probability"].mean())), "nível médio")
        with c3:
            kpi_card("Top 5% risco", pct(float(r["peak_probability"].quantile(0.95))), "alerta")
        with c4:
            alerts = int((r["peak_probability"] >= 0.5).sum())
            kpi_card("Alertas (p≥0.5)", f"{alerts}", "no ficheiro")

        if ALTAIR_OK:
            ensure_altair()
            base = alt.Chart(r).encode(x=alt.X("date:T", title=None))
            line = base.mark_line().encode(
                y=alt.Y("peak_probability:Q", title=None),
                tooltip=["date:T", alt.Tooltip("peak_probability:Q", format=".3f"), "is_peak_day:Q"]
            )
            st.altair_chart((line).properties(height=320, title="Probabilidade de pico"), use_container_width=True)
        else:
            st.line_chart(r.set_index("date")["peak_probability"])

        download_csv_button(r, "peak_probabilities.csv", "⬇️ Download peak probabilities (CSV)")

    
    st.markdown("## Explainability (Peak Model)")
    if peak_imp is None or peak_imp.empty:
        st.info(
            "Ainda não encontrei `peak_feature_importance.csv` na pasta outputs.\n\n"
            "✅ Quando exportares esse ficheiro no treino, vou mostrar aqui um ranking executivo."
        )
    else:
        imp = peak_imp.copy()
        
        cols = [c.lower() for c in imp.columns]
        if "feature" not in cols or "importance" not in cols:
            st.warning("Formato esperado: colunas `feature` e `importance`.")
        else:
            
            imp.columns = [c.lower() for c in imp.columns]
            imp = imp.sort_values("importance", ascending=False).head(20)

            c1, c2 = st.columns([1.2, 0.9])
            with c1:
                bar_chart(imp, "feature", "importance", "Top drivers do risco de pico (XGBoost importance)", height=420)
            with c2:
                top1 = imp.iloc[0]
                top2 = imp.iloc[1] if len(imp) > 1 else None
                kpi_card("Driver #1", str(top1["feature"]), f"importance={float(top1['importance']):.4f}")
                st.write("")
                if top2 is not None:
                    kpi_card("Driver #2", str(top2["feature"]), f"importance={float(top2['importance']):.4f}")

            st.markdown("### Download")
            download_csv_button(imp, "peak_feature_importance_top20.csv", "⬇️ Download top 20 importance (CSV)")

st.markdown("")
st.caption("© OMIE Intelligence — Dark-only Apple Night • Executive Dashboard")
