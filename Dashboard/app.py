"""
Precursor — Market Signal Intelligence
=======================================
Financial ML dashboard tracking insider trades,
macro signals, and model predictions across S&P 500.

Tables used:
  precursor.gold.features
  precursor.gold.predictions
  precursor.gold.backtest
  precursor.gold.findings
  precursor.gold.agreement
  precursor.bronze.sec_clean
  precursor.silver.joined
"""

import json
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import date, datetime, timedelta
from databricks import sql
from typing import Optional
import requests
import time

# ── Page config ───────────────────────────────────────────────

st.set_page_config(
    page_title="Precursor Market Intelligence",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Global CSS ────────────────────────────────────────────────

st.html("""
<link href="https://fonts.googleapis.com/css2?family=DM+Mono:ital,wght@0,300;0,400;0,500;1,300&family=Syne:wght@400;500;600;700;800&display=swap" rel="stylesheet">
<style>
  html, body, [class*="css"] {
    font-family: 'DM Mono', monospace;
    background-color: #080c14;
    color: #e2e8f0;
  }
  .main { background-color: #080c14; }
  .block-container {
    padding: 2rem 2.5rem 4rem 2.5rem !important;
    max-width: 1400px;
  }
  #MainMenu, footer, header { visibility: hidden; }

  section[data-testid="stSidebar"] {
    background: #0d1420;
    border-right: 1px solid #1a2744;
  }

  div[data-testid="metric-container"] {
    background: #0d1420;
    border: 1px solid #1a2744;
    border-radius: 10px;
    padding: 18px 20px;
  }
  div[data-testid="metric-container"] label {
    font-family: 'DM Mono', monospace !important;
    font-size: 10px !important;
    color: #4a6080 !important;
    text-transform: uppercase;
    letter-spacing: 1.5px;
  }
  div[data-testid="stMetricValue"] {
    font-family: 'Syne', sans-serif !important;
    font-size: 30px !important;
    font-weight: 700 !important;
    color: #00d4ff !important;
  }

  div[data-testid="stDataFrame"] {
    border: 1px solid #1a2744;
    border-radius: 10px;
    overflow: hidden;
  }

  hr { border-color: #1a2744 !important; margin: 28px 0 !important; }

  .stSelectbox label, .stRadio label {
    font-family: 'DM Mono', monospace !important;
    font-size: 11px !important;
    color: #4a6080 !important;
    text-transform: uppercase;
    letter-spacing: 1px;
  }

  div[role="radiogroup"] label {
    font-family: 'DM Mono', monospace !important;
    font-size: 13px !important;
    color: #94a3b8 !important;
    padding: 8px 12px !important;
    border-radius: 6px !important;
  }
  div[role="radiogroup"] label:hover {
    color: #e2e8f0 !important;
    background: #1a2744 !important;
  }
</style>
""" )

# ── Color palette ─────────────────────────────────────────────

C = {
    "bg":       "#080c14",
    "card":     "#0d1420",
    "border":   "#1a2744",
    "blue":     "#00d4ff",
    "green":    "#00ff88",
    "red":      "#ff4444",
    "orange":   "#ff8800",
    "muted":    "#4a6080",
    "text":     "#e2e8f0",
    "subtext":  "#94a3b8",
}

PLOTLY_BASE = dict(
    paper_bgcolor=C["bg"],
    plot_bgcolor=C["card"],
    font=dict(color=C["text"], family="DM Mono"),
    margin=dict(t=60, b=40, l=50, r=30),
    xaxis=dict(gridcolor="#1a2744", showgrid=True,
               zeroline=False, tickfont=dict(size=11)),
    yaxis=dict(gridcolor="#1a2744", showgrid=True,
               zeroline=False, tickfont=dict(size=11)),
    legend=dict(
        bgcolor=C["card"],
        bordercolor=C["border"],
        borderwidth=1,
        font=dict(size=11),
    ),
    hoverlabel=dict(
        bgcolor=C["card"],
        bordercolor=C["border"],
        font=dict(family="DM Mono", size=12),
    ),
)

# ── Database connection ───────────────────────────────────────

HOSTNAME     = "dbc-b7ee8514-f214.cloud.databricks.com"
WAREHOUSE_ID = "189351f1633e6859"
HTTP_PATH    = f"/sql/1.0/warehouses/{WAREHOUSE_ID}"

def ensure_warehouse_running() -> None:
    token   = st.secrets["DATABRICKS_TOKEN"]
    headers = {"Authorization": f"Bearer {token}"}
    state   = requests.get(
        f"https://{HOSTNAME}/api/2.0/sql/warehouses/{WAREHOUSE_ID}",
        headers=headers,
    ).json().get("state", "UNKNOWN")
    if state == "RUNNING":
        return
    requests.post(
        f"https://{HOSTNAME}/api/2.0/sql/warehouses/{WAREHOUSE_ID}/start",
        headers=headers,
    )
    for _ in range(90):
        time.sleep(1)
        state = requests.get(
            f"https://{HOSTNAME}/api/2.0/sql/warehouses/{WAREHOUSE_ID}",
            headers=headers,
        ).json().get("state", "UNKNOWN")
        if state == "RUNNING":
            return

@st.cache_data(ttl=3600)
def run_query(sql_str: str) -> pd.DataFrame:
    try:
        ensure_warehouse_running()
        with sql.connect(
            server_hostname=HOSTNAME,
            http_path=HTTP_PATH,
            access_token=st.secrets["DATABRICKS_TOKEN"],
            _retry_stop_after_attempts_count=3,
            _retry_delay_min=5,
            _retry_delay_max=15,
        ) as connection:
            with connection.cursor() as cursor:
                cursor.execute(sql_str)
                rows = cursor.fetchall()
                cols = [d[0] for d in cursor.description]
                return pd.DataFrame(rows, columns=cols)
    except Exception as exc:
        st.error(f"Query failed: {exc}")
        return pd.DataFrame()


# ── Cached data loaders ───────────────────────────────────────

@st.cache_data(ttl=3600)
def load_insider_monthly() -> pd.DataFrame:
    return run_query("""
        SELECT DATE_TRUNC('month', transaction_date) AS month,
               COUNT(*) AS filing_count
        FROM precursor.bronze.sec_clean
        GROUP BY 1 ORDER BY 1
    """)


@st.cache_data(ttl=3600)
def load_spy_price() -> pd.DataFrame:
    return run_query("""
        SELECT date, close
        FROM precursor.gold.features
        WHERE ticker = 'SPY'
        ORDER BY date
    """)


@st.cache_data(ttl=3600)
def load_findings() -> dict:
    df = run_query("SELECT findings FROM precursor.gold.findings LIMIT 1")
    if df.empty:
        return {}
    try:
        return json.loads(df.iloc[0]["findings"])
    except Exception:
        return {}


@st.cache_data(ttl=3600)
def load_sector_filings() -> pd.DataFrame:
    return run_query("""
        SELECT f.sector, COUNT(*) AS filings
        FROM precursor.bronze.sec_clean s
        JOIN precursor.gold.features f
          ON s.ticker = f.ticker
         AND s.transaction_date = f.date
        WHERE f.sector IS NOT NULL
        GROUP BY f.sector
        ORDER BY filings DESC
    """)


@st.cache_data(ttl=3600)
def load_recent_filings(ticker: str = "All", limit: int = 100) -> pd.DataFrame:
    where = "" if ticker == "All" else f"AND ticker = '{ticker}'"
    return run_query(f"""
        SELECT ticker, transaction_date, filing_date,
               days_to_file, is_late_filing
        FROM precursor.bronze.sec_clean
        WHERE 1=1 {where}
        ORDER BY transaction_date DESC
        LIMIT {limit}
    """)


@st.cache_data(ttl=3600)
def load_backtest() -> pd.DataFrame:
    return run_query("""
        SELECT date, model, horizon,
               daily_return, cumulative_return,
               benchmark_return, daily_accuracy, trades_taken
        FROM precursor.gold.backtest
        ORDER BY date
    """)


@st.cache_data(ttl=3600)
def load_ticker_list() -> list[str]:
    df = run_query("""
        SELECT DISTINCT ticker
        FROM precursor.gold.predictions
        WHERE dataset = 'inference'
        ORDER BY ticker
    """)
    return df["ticker"].tolist() if not df.empty else []


@st.cache_data(ttl=3600)
def load_predictions_for_ticker(
    ticker: str, model: str, start: str, end: str
) -> pd.DataFrame:
    return run_query(f"""
        SELECT p.date, p.model, p.horizon,
               p.prediction, p.probability, p.confidence,
               f.target_1d, f.target_21d,
               CASE
                 WHEN p.horizon = '1d'
                      AND p.prediction = f.target_1d  THEN 1
                 WHEN p.horizon = '21d'
                      AND p.prediction = f.target_21d THEN 1
                 ELSE 0
               END AS correct
        FROM precursor.gold.predictions p
        LEFT JOIN precursor.gold.features f
          ON p.ticker = f.ticker AND p.date = f.date
        WHERE p.ticker = '{ticker}'
          AND p.model  = '{model}'
          AND p.date BETWEEN '{start}' AND '{end}'
        ORDER BY p.date DESC
        LIMIT 90
    """)


@st.cache_data(ttl=3600)
def load_ohlcv(ticker: str, start: str, end: str) -> pd.DataFrame:
    return run_query(f"""
        SELECT date, open, high, low, close, volume,
               rsi_14, macd, bb_position, volatility_21d,
               volume_zscore_21d, filing_count,
               DFF, T10Y2Y, VIXCLS, UNRATE, CPIAUCSL, M2SL
        FROM precursor.silver.joined
        WHERE ticker = '{ticker}'
          AND date BETWEEN '{start}' AND '{end}'
        ORDER BY date
    """)


@st.cache_data(ttl=3600)
def load_top_picks(model: str, horizon: str) -> pd.DataFrame:
    return run_query(f"""
        SELECT p.ticker, p.probability, p.confidence,
               p.prediction,
               f.sector,
               f.rsi_14,
               f.price_vs_52w_high
        FROM precursor.gold.predictions p
        JOIN precursor.gold.features f
          ON p.ticker = f.ticker AND p.date = f.date
        WHERE p.dataset  = 'inference'
          AND p.model    = '{model}'
          AND p.horizon  = '{horizon}'
          AND p.prediction = 1
        ORDER BY p.confidence DESC
        LIMIT 20
    """)


@st.cache_data(ttl=3600)
def load_data_freshness() -> Optional[datetime]:
    df = run_query("""
        SELECT MAX(predicted_at) AS latest
        FROM precursor.gold.predictions
    """)
    if df.empty or df.iloc[0]["latest"] is None:
        return None
    return pd.to_datetime(df.iloc[0]["latest"])


# ── UI helpers ────────────────────────────────────────────────

def section_header(title: str, subtitle: str = "") -> None:
    sub_html = (
        f'<div style="font-size:13px;color:{C["subtext"]};'
        f'margin-top:5px;font-family:DM Mono;">{subtitle}</div>'
        if subtitle else ""
    )
    st.html(f"""
    <div style="margin:36px 0 20px 0;
                padding-bottom:14px;
                border-bottom:1px solid {C['border']};">
      <div style="font-family:Syne,sans-serif;
                  font-size:20px;
                  font-weight:700;
                  color:{C['text']};
                  letter-spacing:-0.3px;">
        {title}
      </div>
      {sub_html}
    </div>
    """ )


def insight_card(text: str, color: str = "#00d4ff") -> None:
    st.html(f"""
    <div style="background:{C['card']};
                border-left:3px solid {color};
                border-radius:0 8px 8px 0;
                padding:16px 20px;
                margin:16px 0;
                font-size:13px;
                color:{C['subtext']};
                font-family:DM Mono;
                line-height:1.7;">
      {text}
    </div>
    """ )


def stat_card(value: str, label: str,
              color: str = "#00d4ff") -> str:
    return f"""
    <div style="background:{C['card']};
                border:1px solid {C['border']};
                border-radius:12px;
                padding:24px 20px;
                text-align:center;">
      <div style="font-family:Syne,sans-serif;
                  font-size:38px;
                  font-weight:800;
                  color:{color};
                  line-height:1;">
        {value}
      </div>
      <div style="font-size:10px;
                  color:{C['muted']};
                  margin-top:10px;
                  text-transform:uppercase;
                  letter-spacing:1.5px;
                  font-family:DM Mono;">
        {label}
      </div>
    </div>
    """


def prediction_badge(prediction: int,
                     probability: float,
                     horizon: str) -> str:
    label  = "↑ UP" if prediction == 1 else "↓ DOWN"
    color  = C["green"] if prediction == 1 else C["red"]
    pct    = f"{probability*100:.1f}%"
    return f"""
    <div style="background:{C['card']};
                border:2px solid {color};
                border-radius:12px;
                padding:20px;
                text-align:center;">
      <div style="font-family:Syne,sans-serif;
                  font-size:32px;
                  font-weight:800;
                  color:{color};">
        {label}
      </div>
      <div style="font-size:22px;
                  color:{color};
                  margin-top:4px;
                  font-family:DM Mono;">
        {pct}
      </div>
      <div style="font-size:10px;
                  color:{C['muted']};
                  margin-top:8px;
                  text-transform:uppercase;
                  letter-spacing:1px;
                  font-family:DM Mono;">
        {horizon} prediction
      </div>
    </div>
    """


# ── Sidebar ───────────────────────────────────────────────────

with st.sidebar:
    st.html(f"""
    <div style="padding:8px 4px 20px 4px;">
      <div style="font-family:Syne,sans-serif;
                  font-size:26px;
                  font-weight:800;
                  color:{C['blue']};
                  letter-spacing:-0.5px;">
        PRECURSOR
      </div>
      <div style="font-size:11px;
                  color:{C['muted']};
                  margin-top:4px;
                  text-transform:uppercase;
                  letter-spacing:2px;
                  font-family:DM Mono;">
        Market Signal Intelligence
      </div>
      <div style="font-size:10px;
                  color:{C['muted']};
                  margin-top:8px;
                  font-family:DM Mono;
                  opacity:0.7;">
        613 stocks · 2020–2026 · S&P 500
      </div>
    </div>
    <hr style="border-color:{C['border']};margin:0 0 20px 0;">
    """ )

    page = st.radio(
        "Navigation",
        [
            "🔴  The Insider Trades",
            "📊  Market Insights",
            "🔍  Stock Explorer",
            "🏆  Today's Picks",
        ],
        label_visibility="collapsed",
    )

    st.html("<hr>" )

    selected_model_label = st.selectbox(
        "Model",
        ["XGBoost (21-day)", "TFT (1-day)"],
        help="XGBoost predicts 21-day direction. TFT predicts tomorrow.",
    )
    model_key   = "xgboost" if "XGBoost" in selected_model_label else "tft"
    horizon_key = "21d"     if "XGBoost" in selected_model_label else "1d"

    date_range = st.date_input(
        "Date range",
        value=(date(2024, 1, 1), date.today()),
        min_value=date(2020, 4, 1),
        max_value=date.today(),
    )
    start_str = str(date_range[0]) if len(date_range) > 0 else "2024-01-01"
    end_str   = str(date_range[1]) if len(date_range) > 1 else str(date.today())

    st.html("<hr>" )

    # Data freshness
    freshness = load_data_freshness()
    if freshness:
        age_hrs = (datetime.now(freshness.tzinfo) - freshness).total_seconds() / 3600
        color   = C["green"] if age_hrs < 24 else C["red"]
        label   = freshness.strftime("%b %d %H:%M UTC")
        st.html(f"""
        <div style="font-size:10px;color:{C['muted']};
                    font-family:DM Mono;text-transform:uppercase;
                    letter-spacing:1px;margin-bottom:4px;">
          Last updated
        </div>
        <div style="font-size:12px;color:{color};font-family:DM Mono;">
          ● {label}
        </div>
        """ )

    st.html(f"""
    <div style="margin-top:24px;font-size:10px;
                color:{C['muted']};font-family:DM Mono;
                line-height:1.6;opacity:0.8;">
      ⚠️ Not financial advice.<br>
      For research purposes only.
    </div>
    """ )


# ══════════════════════════════════════════════════════════════
# PAGE 1 — THE INSIDER TRADES
# ══════════════════════════════════════════════════════════════

if "Insider" in page:

    # Hero hook
    st.html(f"""
    <div style="padding:48px 0 36px 0;text-align:center;">
      <div style="font-family:Syne,sans-serif;
                  font-size:52px;
                  font-weight:800;
                  color:{C['text']};
                  letter-spacing:-1.5px;
                  line-height:1.1;">
        The stock market is rigged.
      </div>
      <div style="font-family:DM Mono;
                  font-size:18px;
                  color:{C['blue']};
                  margin-top:16px;">
        I have 225,346 legal documents proving it.
      </div>
      <div style="font-family:DM Mono;
                  font-size:13px;
                  color:{C['muted']};
                  margin-top:10px;">
        All public. All legal. All hiding in plain sight on the SEC website.
      </div>
    </div>
    """ )

    # Stat cards
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.html(stat_card("225,346", "Form 4 Filings") )
    with c2:
        st.html(stat_card("613", "Companies Tracked") )
    with c3:
        st.html(stat_card("6 yrs", "2020 → 2026", color=C["green"]) )
    with c4:
        st.html(stat_card("48 hrs", "SEC Disclosure Window", color=C["orange"]) )

    st.html("<br>" )

    # ── Hero chart ────────────────────────────────────────────
    section_header(
        "They Bought The Bottom. Both Times.",
        "Monthly insider filing activity vs S&P 500 price (2020–2026)"
    )

    with st.spinner("Loading insider trading data..."):
        insider_monthly = load_insider_monthly()
        spy_price       = load_spy_price()

    if not insider_monthly.empty and not spy_price.empty:
        insider_monthly["month"] = pd.to_datetime(insider_monthly["month"])
        spy_price["date"]        = pd.to_datetime(spy_price["date"])

        fig = make_subplots(specs=[[{"secondary_y": True}]])

        fig.add_trace(
            go.Bar(
                x=insider_monthly["month"],
                y=insider_monthly["filing_count"],
                name="Insider Filings",
                marker=dict(
                    color=C["blue"],
                    opacity=0.65,
                    line=dict(width=0),
                ),
                hovertemplate="%{x|%b %Y}<br>%{y:,} filings<extra></extra>",
            ),
            secondary_y=False,
        )

        fig.add_trace(
            go.Scatter(
                x=spy_price["date"],
                y=spy_price["close"],
                name="S&P 500",
                line=dict(color="white", width=2),
                hovertemplate="%{x|%b %d %Y}<br>S&P 500: $%{y:.0f}<extra></extra>",
            ),
            secondary_y=True,
        )

        # Annotations for key moments
        for x_date, label in [
            ("2020-03-23", "Insiders bought aggressively<br>Market bottomed"),
            ("2022-10-12", "Insider buying spiked again<br>Market bottomed exactly then"),
        ]:
            fig.add_vline(
                x=x_date,
                line_dash="dash",
                line_color=C["red"],
                line_width=1.5,
            )
            fig.add_annotation(
                x=x_date, y=1, yref="paper",
                text=label,
                showarrow=False,
                font=dict(color=C["red"], size=11, family="DM Mono"),
                bgcolor=C["card"],
                bordercolor=C["red"],
                borderwidth=1,
                xanchor="left",
                yanchor="top",
                xshift=8,
            )

        layout = {**PLOTLY_BASE}
        layout.update(
            height=480,
            showlegend=False,
            title=None,
            xaxis=dict(gridcolor=C["border"], showgrid=False,
                       tickfont=dict(size=11)),
            yaxis=dict(
                gridcolor=C["border"], showgrid=True,
                title="Monthly Filings",
                title_font=dict(color=C["blue"], size=11),
                tickfont=dict(size=11),
            ),
            yaxis2=dict(
                title="S&P 500 Price",
                title_font=dict(color="white", size=11),
                tickfont=dict(size=11),
                gridcolor=C["border"],
                showgrid=False,
            ),
        )
        fig.update_layout(**layout)
        st.plotly_chart(fig, use_container_width=True)

        insight_card(
            "Executive stock purchases must be disclosed to the SEC within 48 hours "
            "via Form 4 filings. These 225,346 documents are public record on "
            "edgar.sec.gov. The pattern above is not a coincidence — insiders "
            "bought aggressively at both major market bottoms.",
            color=C["red"],
        )

    # ── Sector breakdown ──────────────────────────────────────
    section_header(
        "Which Sectors Had The Most Insider Activity?",
        "Cumulative Form 4 filings by GICS sector · 2020–2026"
    )

    col_left, col_right = st.columns(2)

    with col_left:
        with st.spinner("Loading sector data..."):
            sector_df = load_sector_filings()

        if not sector_df.empty:
            max_filings = sector_df["filings"].max()
            colors = [
                C["blue"] if v == max_filings
                else f"rgba(0, 212, 255, {0.3 + 0.5 * v / max_filings})"
                for v in sector_df["filings"]
            ]
            fig2 = go.Figure(go.Bar(
                x=sector_df["filings"],
                y=sector_df["sector"],
                orientation="h",
                marker=dict(color=colors, line=dict(width=0)),
                hovertemplate="%{y}<br>%{x:,} filings<extra></extra>",
            ))
            layout2 = {**PLOTLY_BASE}
            layout2.update(
                height=360,
                title=None,
                xaxis=dict(gridcolor=C["border"], title="Total Filings",
                           tickfont=dict(size=11)),
                yaxis=dict(gridcolor=C["border"], showgrid=False,
                           tickfont=dict(size=11)),
            )
            fig2.update_layout(**layout2)
            st.plotly_chart(fig2, use_container_width=True)

    with col_right:
        findings = load_findings()
        edge     = findings.get("insider_activity_edge", {})

        if edge:
            spike_acc    = edge.get("accuracy_with_spike", 0.5)
            no_spike_acc = edge.get("accuracy_without_spike", 0.5)
            finding_text = edge.get("finding", "")

            fig3 = go.Figure(go.Bar(
                x=["Normal Activity", "Spike Detected"],
                y=[no_spike_acc * 100, spike_acc * 100],
                marker=dict(
                    color=[C["muted"], C["green"] if spike_acc >= no_spike_acc else C["red"]],
                    line=dict(width=0),
                ),
                hovertemplate="%{x}<br>Accuracy: %{y:.1f}%<extra></extra>",
            ))
            fig3.add_hline(
                y=50, line_dash="dash",
                line_color=C["muted"], line_width=1,
                annotation_text="Random baseline",
                annotation_font=dict(color=C["muted"], size=10),
            )
            layout3 = {**PLOTLY_BASE}
            layout3.update(
                height=280,
                title=dict(text="Does Insider Activity Add Signal?",
                           font=dict(size=14, color=C["text"]), x=0),
                yaxis=dict(gridcolor=C["border"], range=[45, 55],
                           title="Model Accuracy %", tickfont=dict(size=11)),
                xaxis=dict(showgrid=False, tickfont=dict(size=11)),
            )
            fig3.update_layout(**layout3)
            st.plotly_chart(fig3, use_container_width=True)
            if finding_text:
                insight_card(finding_text, color=C["green"])

    # ── Raw filings table ─────────────────────────────────────
    section_header(
        "The Evidence — Recent Form 4 Filings",
        "Every row is a legal disclosure. Public record. Source: SEC EDGAR"
    )

    tickers_list = load_ticker_list()
    ticker_filter = st.selectbox(
        "Filter by ticker",
        ["All"] + tickers_list,
        key="sec_ticker_filter",
    )

    with st.spinner("Loading filings..."):
        filings_df = load_recent_filings(ticker_filter)

    if not filings_df.empty:
        filings_display = filings_df.rename(columns={
            "transaction_date": "Trade Date",
            "filing_date":      "Filed With SEC",
            "days_to_file":     "Days to Disclose",
            "is_late_filing":   "Late?",
        })
        st.dataframe(
            filings_display,
            use_container_width=True,
            hide_index=True,
        )
        st.caption("Source: SEC EDGAR (edgar.sec.gov) — public domain")


# ══════════════════════════════════════════════════════════════
# PAGE 2 — MARKET INSIGHTS
# ══════════════════════════════════════════════════════════════

elif "Insights" in page:

    section_header(
        "What Did The Models Actually Learn?",
        "Research findings across 613 stocks · 2020–2026 · 42 features"
    )

    # Pipeline stats
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.metric("Training Rows", "762K")
    with c2:
        st.metric("Features", "42")
    with c3:
        st.metric("XGBoost AUC (21d)", "0.533")
    with c4:
        st.metric("Stocks", "613")

    # ── TFT attention weights ─────────────────────────────────
    section_header(
        "TFT Attention Weights — What The AI Paid Attention To",
        "Temporal Fusion Transformer variable importance from training"
    )

    # Hardcoded from training output
    encoder_data = {
        "return_63d":       135,
        "return_1d":         70,
        "volatility_21d":    42,
        "rsi_14":            32,
        "atr_pct":           28,
        "volume_zscore_21d": 25,
        "return_5d":         24,
        "macd":              22,
        "bb_position":       20,
        "return_21d":        18,
    }
    decoder_data = {
        "inflation_mom":        155,
        "unemployment_delta_63d": 115,
        "fed_rate_delta_63d":    68,
        "yield_curve_change_21d":50,
        "vix_level":             45,
        "yield_curve_level":     32,
        "fed_rate_delta_21d":    24,
        "m2_growth_63d":         12,
        "vix_zscore_63d":         8,
        "macro_regime":           6,
    }

    col_enc, col_dec = st.columns(2)

    with col_enc:
        enc_features = list(encoder_data.keys())
        enc_vals     = list(encoder_data.values())
        max_enc      = max(enc_vals)
        enc_colors   = [
            C["blue"] if v >= sorted(enc_vals)[-3] else "#1e4a6e"
            for v in enc_vals
        ]
        fig_enc = go.Figure(go.Bar(
            x=enc_vals, y=enc_features,
            orientation="h",
            marker=dict(color=enc_colors, line=dict(width=0)),
            hovertemplate="%{y}<br>Weight: %{x}<extra></extra>",
        ))
        layout_enc = {**PLOTLY_BASE}
        layout_enc.update(
            height=360,
            title=dict(
                text="Encoder — Historical Patterns",
                font=dict(size=14, color=C["text"]), x=0,
            ),
            xaxis=dict(gridcolor=C["border"], title="Attention Weight",
                       tickfont=dict(size=10)),
            yaxis=dict(showgrid=False, tickfont=dict(size=11),
                       autorange="reversed"),
        )
        fig_enc.update_layout(**layout_enc)
        st.plotly_chart(fig_enc, use_container_width=True)
        st.caption("Quarterly momentum dominates historical pattern recognition")

    with col_dec:
        dec_features = list(decoder_data.keys())
        dec_vals     = list(decoder_data.values())
        dec_colors   = [
            C["green"] if v >= sorted(dec_vals)[-3] else "#1a4a2e"
            for v in dec_vals
        ]
        fig_dec = go.Figure(go.Bar(
            x=dec_vals, y=dec_features,
            orientation="h",
            marker=dict(color=dec_colors, line=dict(width=0)),
            hovertemplate="%{y}<br>Weight: %{x}<extra></extra>",
        ))
        layout_dec = {**PLOTLY_BASE}
        layout_dec.update(
            height=360,
            title=dict(
                text="Decoder — Future Context",
                font=dict(size=14, color=C["text"]), x=0,
            ),
            xaxis=dict(gridcolor=C["border"], title="Attention Weight",
                       tickfont=dict(size=10)),
            yaxis=dict(showgrid=False, tickfont=dict(size=11),
                       autorange="reversed"),
        )
        fig_dec.update_layout(**layout_dec)
        st.plotly_chart(fig_dec, use_container_width=True)
        st.caption("Inflation and unemployment dominate macro context")

    insight_card(
        "The TFT separated HOW a stock has moved (encoder) from "
        "WHAT MACRO ENVIRONMENT it is in (decoder). "
        "Quarterly momentum drives historical recognition. "
        "Inflation and unemployment drive future context. "
        "This mirrors how professional portfolio managers think about markets.",
        color=C["blue"],
    )

    # ── Sector predictability ─────────────────────────────────
    section_header(
        "Which Sectors Are Most Predictable?",
        "XGBoost accuracy by GICS sector · test set 2024–2026"
    )

    findings = load_findings()
    sector_ranking = findings.get("sector_predictability", {}).get("ranking", [])

    if sector_ranking:
        sec_df = pd.DataFrame(sector_ranking)
        sec_df["accuracy_pct"] = sec_df["accuracy"] * 100
        sec_df["color"] = sec_df["accuracy"].apply(
            lambda x: C["green"] if x > 0.51 else C["red"]
        )
        fig_sec = go.Figure(go.Bar(
            x=sec_df["accuracy_pct"],
            y=sec_df["sector"],
            orientation="h",
            marker=dict(color=sec_df["color"].tolist(), line=dict(width=0)),
            hovertemplate="%{y}<br>Accuracy: %{x:.1f}%<extra></extra>",
            text=sec_df["accuracy_pct"].apply(lambda x: f"{x:.1f}%"),
            textposition="outside",
            textfont=dict(size=11, color=C["subtext"]),
        ))
        fig_sec.add_vline(
            x=50, line_dash="dash",
            line_color=C["muted"], line_width=1.5,
            annotation_text="Random baseline (50%)",
            annotation_font=dict(color=C["muted"], size=10),
        )
        layout_sec = {**PLOTLY_BASE}
        layout_sec.update(
            height=380,
            title=None,
            xaxis=dict(
                gridcolor=C["border"], range=[46, 57],
                title="Accuracy %", tickfont=dict(size=11),
            ),
            yaxis=dict(showgrid=False, tickfont=dict(size=11),
                       autorange="reversed"),
        )
        fig_sec.update_layout(**layout_sec)
        st.plotly_chart(fig_sec, use_container_width=True)

        finding_text = findings.get("sector_predictability", {}).get("finding", "")
        if finding_text:
            insight_card(finding_text, color=C["orange"])

    # ── Macro regime analysis ─────────────────────────────────
    section_header(
        "Macro Regime Analysis",
        "Does the economic environment affect model accuracy?"
    )

    col_regime, col_vix = st.columns(2)

    with col_regime:
        macro_data = findings.get("macro_regime_accuracy", {})
        accs       = macro_data.get("accuracies", {})
        if accs:
            labels = ["Risk Off", "Neutral", "Risk On"]
            values = [
                (accs.get("risk_off") or 0.5) * 100,
                (accs.get("neutral")  or 0.5) * 100,
                (accs.get("risk_on")  or 0.5) * 100,
            ]
            fig_regime = go.Figure(go.Bar(
                x=labels, y=values,
                marker=dict(
                    color=[C["red"], C["orange"], C["green"]],
                    line=dict(width=0),
                ),
                hovertemplate="%{x}<br>Accuracy: %{y:.1f}%<extra></extra>",
            ))
            fig_regime.add_hline(
                y=50, line_dash="dash",
                line_color=C["muted"], line_width=1,
            )
            layout_regime = {**PLOTLY_BASE}
            layout_regime.update(
                height=300,
                title=dict(text="Accuracy by Macro Regime",
                           font=dict(size=14), x=0),
                yaxis=dict(gridcolor=C["border"], range=[45, 56],
                           title="Accuracy %", tickfont=dict(size=11)),
                xaxis=dict(showgrid=False, tickfont=dict(size=12)),
            )
            fig_regime.update_layout(**layout_regime)
            st.plotly_chart(fig_regime, use_container_width=True)
            finding_text = macro_data.get("finding", "")
            if finding_text:
                insight_card(finding_text, color=C["orange"])

    with col_vix:
        vix_data = findings.get("vix_regime_accuracy", {})
        if vix_data:
            low_vix  = (vix_data.get("low_vix_accuracy")  or 0.5) * 100
            high_vix = (vix_data.get("high_vix_accuracy") or 0.5) * 100
            fig_vix = go.Figure(go.Bar(
                x=["Low VIX (<15)", "High VIX (>25)"],
                y=[low_vix, high_vix],
                marker=dict(
                    color=[C["green"], C["red"]],
                    line=dict(width=0),
                ),
                hovertemplate="%{x}<br>Accuracy: %{y:.1f}%<extra></extra>",
            ))
            fig_vix.add_hline(
                y=50, line_dash="dash",
                line_color=C["muted"], line_width=1,
            )
            layout_vix = {**PLOTLY_BASE}
            layout_vix.update(
                height=300,
                title=dict(text="Accuracy by Market Fear (VIX)",
                           font=dict(size=14), x=0),
                yaxis=dict(gridcolor=C["border"], range=[45, 56],
                           title="Accuracy %", tickfont=dict(size=11)),
                xaxis=dict(showgrid=False, tickfont=dict(size=12)),
            )
            fig_vix.update_layout(**layout_vix)
            st.plotly_chart(fig_vix, use_container_width=True)
            vix_finding = vix_data.get("finding", "")
            if vix_finding:
                insight_card(vix_finding, color=C["blue"])

    # ── Backtest ──────────────────────────────────────────────
    section_header(
        "Strategy Performance vs S&P 500",
        "Confidence-filtered trades · 0.01% commission + 0.05% slippage"
    )

    with st.spinner("Loading backtest..."):
        bt_df = load_backtest()

    if not bt_df.empty:
        bt_df["date"] = pd.to_datetime(bt_df["date"])

        fig_bt = go.Figure()

        for mdl, color, label in [
            ("xgboost", C["blue"],  "XGBoost (21d)"),
            ("tft",     C["green"], "TFT (1d)"),
        ]:
            sub = bt_df[bt_df["model"] == mdl]
            if sub.empty:
                continue
            fig_bt.add_trace(go.Scatter(
                x=sub["date"],
                y=sub["cumulative_return"] * 100,
                name=label,
                line=dict(color=color, width=2),
                hovertemplate="%{x|%b %Y}<br>Return: %{y:.1f}%<extra>" + label + "</extra>",
            ))

        # SPY benchmark
        sub_spy = bt_df[bt_df["model"] == "xgboost"]
        if not sub_spy.empty:
            fig_bt.add_trace(go.Scatter(
                x=sub_spy["date"],
                y=sub_spy["benchmark_return"] * 100,
                name="S&P 500 (benchmark)",
                line=dict(color=C["muted"], width=1.5, dash="dot"),
                hovertemplate="%{x|%b %Y}<br>SPY: %{y:.1f}%<extra>Benchmark</extra>",
            ))

        # 2022 bear market shading
        fig_bt.add_vrect(
            x0="2022-01-01", x1="2022-12-31",
            fillcolor=C["red"], opacity=0.06,
            layer="below", line_width=0,
            annotation_text="2022 Bear Market",
            annotation_position="top left",
            annotation_font=dict(color=C["muted"], size=10),
        )

        layout_bt = {**PLOTLY_BASE}
        layout_bt.update(
            height=400,
            title=None,
            xaxis=dict(gridcolor=C["border"], showgrid=False,
                       tickfont=dict(size=11)),
            yaxis=dict(gridcolor=C["border"],
                       title="Cumulative Return %",
                       tickfont=dict(size=11)),
        )
        fig_bt.update_layout(**layout_bt)
        st.plotly_chart(fig_bt, use_container_width=True)

        # Backtest summary metrics
        xgb_bt = bt_df[bt_df["model"] == "xgboost"]
        if not xgb_bt.empty:
            final_ret   = xgb_bt["cumulative_return"].iloc[-1] * 100
            spy_ret     = xgb_bt["benchmark_return"].iloc[-1] * 100
            avg_acc     = xgb_bt["daily_accuracy"].mean() * 100
            total_trades = xgb_bt["trades_taken"].sum()

            m1, m2, m3, m4 = st.columns(4)
            m1.metric("Strategy Return", f"{final_ret:.1f}%")
            m2.metric("S&P 500 Return",  f"{spy_ret:.1f}%")
            m3.metric("Avg Daily Accuracy", f"{avg_acc:.1f}%")
            m4.metric("Total Trades", f"{int(total_trades):,}")

        insight_card(
            "Model accuracy of ~53% on 21-day direction is consistent with "
            "academic literature on stock prediction. The Efficient Market "
            "Hypothesis is real. The more interesting finding is in the "
            "attention weights — the model learned that macro regime separates "
            "from price momentum in a theoretically consistent way.",
            color=C["muted"],
        )


# ══════════════════════════════════════════════════════════════
# PAGE 3 — STOCK EXPLORER
# ══════════════════════════════════════════════════════════════

elif "Explorer" in page:

    section_header("Stock Explorer", "Deep dive into any S&P 500 stock")

    tickers_list = load_ticker_list()
    if not tickers_list:
        st.warning("No tickers available — run the predict pipeline first.")
        st.stop()

    # Stock selector row
    col_sel, col_info, col_pred_xgb, col_pred_tft = st.columns([1, 1, 1, 1])

    with col_sel:
        selected_ticker = st.selectbox(
            "Select ticker",
            tickers_list,
            index=tickers_list.index("AAPL") if "AAPL" in tickers_list else 0,
        )

    # Latest price info
    latest_price = run_query(f"""
        SELECT close, sector, date
        FROM precursor.silver.joined
        WHERE ticker = '{selected_ticker}'
        ORDER BY date DESC LIMIT 1
    """)

    with col_info:
        if not latest_price.empty:
            row = latest_price.iloc[0]
            st.html(stat_card(
                f"${row['close']:.2f}",
                f"{row['sector']} · {row['date']}",
                color=C["text"],
            ) )

    # Predictions for this ticker
    for col, mdl, hrz, lbl in [
        (col_pred_xgb, "xgboost", "21d", "XGBoost 21d"),
        (col_pred_tft, "tft",     "1d",  "TFT 1d"),
    ]:
        pred_row = run_query(f"""
            SELECT prediction, probability, confidence
            FROM precursor.gold.predictions
            WHERE ticker   = '{selected_ticker}'
              AND model    = '{mdl}'
              AND horizon  = '{hrz}'
              AND dataset  = 'inference'
            ORDER BY date DESC LIMIT 1
        """)
        with col:
            if not pred_row.empty:
                r = pred_row.iloc[0]
                st.html(
                    prediction_badge(
                        int(r["prediction"]),
                        float(r["probability"]),
                        lbl,
                    ),
                    unsafe_allow_html=True,
                )

    st.html("<br>" )

    # ── Price chart ───────────────────────────────────────────
    section_header(
        f"{selected_ticker} — Price & Signals",
        f"{start_str} to {end_str}"
    )

    with st.spinner(f"Loading {selected_ticker}..."):
        ohlcv = load_ohlcv(selected_ticker, start_str, end_str)
        preds = load_predictions_for_ticker(
            selected_ticker, model_key, start_str, end_str
        )

    if not ohlcv.empty:
        ohlcv["date"] = pd.to_datetime(ohlcv["date"])

        fig_price = make_subplots(
            rows=2, cols=1,
            shared_xaxes=True,
            row_heights=[0.75, 0.25],
            vertical_spacing=0.04,
        )

        # Candlestick
        fig_price.add_trace(go.Candlestick(
            x=ohlcv["date"],
            open=ohlcv["open"], high=ohlcv["high"],
            low=ohlcv["low"],   close=ohlcv["close"],
            increasing=dict(line=dict(color=C["green"]),
                            fillcolor=C["green"]),
            decreasing=dict(line=dict(color=C["red"]),
                            fillcolor=C["red"]),
            name="Price",
        ), row=1, col=1)

        # Prediction markers
        if not preds.empty:
            preds["date"] = pd.to_datetime(preds["date"])
            up_preds   = preds[preds["prediction"] == 1].merge(
                ohlcv[["date", "high"]], on="date", how="left"
            )
            down_preds = preds[preds["prediction"] == 0].merge(
                ohlcv[["date", "low"]], on="date", how="left"
            )

            if not up_preds.empty:
                fig_price.add_trace(go.Scatter(
                    x=up_preds["date"],
                    y=up_preds["high"] * 1.01,
                    mode="markers",
                    marker=dict(
                        symbol="triangle-up",
                        size=8,
                        color=C["green"],
                    ),
                    name="Predicted UP",
                    hovertemplate="%{x|%b %d}<br>↑ UP<extra></extra>",
                ), row=1, col=1)

            if not down_preds.empty:
                fig_price.add_trace(go.Scatter(
                    x=down_preds["date"],
                    y=down_preds["low"] * 0.99,
                    mode="markers",
                    marker=dict(
                        symbol="triangle-down",
                        size=8,
                        color=C["red"],
                    ),
                    name="Predicted DOWN",
                    hovertemplate="%{x|%b %d}<br>↓ DOWN<extra></extra>",
                ), row=1, col=1)

        # Volume
        vol_colors = [
            C["green"] if c >= o else C["red"]
            for c, o in zip(ohlcv["close"], ohlcv["open"])
        ]
        fig_price.add_trace(go.Bar(
            x=ohlcv["date"], y=ohlcv["volume"],
            marker=dict(color=vol_colors, opacity=0.5, line=dict(width=0)),
            name="Volume",
            hovertemplate="%{x|%b %d}<br>Vol: %{y:,.0f}<extra></extra>",
        ), row=2, col=1)

        fig_price.update_layout(
            height=550,
            paper_bgcolor=C["bg"],
            plot_bgcolor=C["card"],
            font=dict(color=C["text"], family="DM Mono"),
            margin=dict(t=20, b=40, l=50, r=30),
            showlegend=True,
            legend=dict(
                bgcolor=C["card"], bordercolor=C["border"],
                borderwidth=1, font=dict(size=11),
            ),
            xaxis_rangeslider_visible=False,
            xaxis2=dict(gridcolor=C["border"], showgrid=False),
            yaxis=dict(gridcolor=C["border"]),
            yaxis2=dict(gridcolor=C["border"], showgrid=False,
                        title="Volume"),
        )
        st.plotly_chart(fig_price, use_container_width=True)

    # ── Technical indicators ──────────────────────────────────
    if not ohlcv.empty:
        col_rsi, col_macd = st.columns(2)

        with col_rsi:
            fig_rsi = go.Figure()
            fig_rsi.add_trace(go.Scatter(
                x=ohlcv["date"], y=ohlcv["rsi_14"],
                line=dict(color=C["blue"], width=1.5),
                name="RSI 14",
                hovertemplate="%{x|%b %d}<br>RSI: %{y:.1f}<extra></extra>",
            ))
            fig_rsi.add_hline(y=70, line_dash="dash",
                               line_color=C["red"], line_width=1,
                               annotation_text="Overbought (70)",
                               annotation_font=dict(color=C["red"], size=10))
            fig_rsi.add_hline(y=30, line_dash="dash",
                               line_color=C["green"], line_width=1,
                               annotation_text="Oversold (30)",
                               annotation_font=dict(color=C["green"], size=10))
            fig_rsi.add_hrect(y0=70, y1=100,
                              fillcolor=C["red"], opacity=0.05, layer="below")
            fig_rsi.add_hrect(y0=0, y1=30,
                              fillcolor=C["green"], opacity=0.05, layer="below")
            layout_rsi = {**PLOTLY_BASE}
            layout_rsi.update(
                height=260, title=dict(text="RSI (14-day)",
                font=dict(size=13), x=0),
                yaxis=dict(gridcolor=C["border"], range=[0, 100],
                           tickfont=dict(size=10)),
                xaxis=dict(gridcolor=C["border"], showgrid=False,
                           tickfont=dict(size=10)),
                showlegend=False,
            )
            fig_rsi.update_layout(**layout_rsi)
            st.plotly_chart(fig_rsi, use_container_width=True)

        with col_macd:
            fig_macd = go.Figure()
            fig_macd.add_trace(go.Scatter(
                x=ohlcv["date"], y=ohlcv["macd"],
                line=dict(color=C["blue"], width=1.5),
                name="MACD",
            ))
            fig_macd.add_hline(y=0, line_color=C["muted"],
                                line_width=1)
            layout_macd = {**PLOTLY_BASE}
            layout_macd.update(
                height=260, title=dict(text="MACD",
                font=dict(size=13), x=0),
                yaxis=dict(gridcolor=C["border"], tickfont=dict(size=10)),
                xaxis=dict(gridcolor=C["border"], showgrid=False,
                           tickfont=dict(size=10)),
                showlegend=False,
            )
            fig_macd.update_layout(**layout_macd)
            st.plotly_chart(fig_macd, use_container_width=True)

    # ── Prediction history table ──────────────────────────────
    section_header(
        f"Prediction History — {selected_ticker}",
        f"Last 90 {model_key.upper()} predictions"
    )

    if not preds.empty:
        preds_display = preds[[
            "date", "model", "horizon",
            "prediction", "probability", "confidence", "correct"
        ]].copy()
        preds_display["probability"] = (
            preds_display["probability"] * 100
        ).round(1).astype(str) + "%"
        preds_display["confidence"] = (
            preds_display["confidence"] * 100
        ).round(1).astype(str) + "%"
        preds_display["prediction"] = preds_display["prediction"].map(
            {1: "↑ UP", 0: "↓ DOWN"}
        )
        preds_display["correct"] = preds_display["correct"].map(
            {1: "✅", 0: "❌", None: "—"}
        )
        preds_display.columns = [
            "Date", "Model", "Horizon",
            "Direction", "Probability", "Confidence", "Correct?"
        ]
        st.dataframe(preds_display, use_container_width=True, hide_index=True)

        # Recent accuracy
        valid = preds[preds["correct"].notna()]
        if not valid.empty:
            recent_acc = valid["correct"].mean() * 100
            st.caption(
                f"Recent accuracy ({len(valid)} predictions): {recent_acc:.1f}%"
            )


# ══════════════════════════════════════════════════════════════
# PAGE 4 — TODAY'S PICKS
# ══════════════════════════════════════════════════════════════

elif "Picks" in page:

    section_header(
        "Today's Top Picks",
        f"Highest confidence predictions · {model_key.upper()} · {horizon_key} horizon"
    )

    st.html(f"""
    <div style="background:{C['card']};
                border:1px solid {C['orange']};
                border-radius:10px;
                padding:14px 18px;
                margin-bottom:24px;
                font-size:12px;
                color:{C['muted']};
                font-family:DM Mono;">
      ⚠️ These are model outputs for research purposes only.
      Not financial advice. Past performance does not guarantee future results.
    </div>
    """ )

    with st.spinner("Loading today's picks..."):
        picks = load_top_picks(model_key, horizon_key)

    if picks.empty:
        st.info("No inference predictions available. Run the predict pipeline first.")
    else:
        picks["probability_pct"] = (picks["probability"] * 100).round(1)
        picks["confidence_pct"]  = (picks["confidence"]  * 100).round(1)

        # Hero metrics
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Stocks Predicting UP", str(len(picks)))
        c2.metric("Avg Confidence",
                  f"{picks['confidence_pct'].mean():.1f}%")
        c3.metric("Avg Probability",
                  f"{picks['probability_pct'].mean():.1f}%")
        c4.metric("Model", model_key.upper())

        st.html("<br>" )

        # Confidence bar chart
        fig_picks = go.Figure(go.Bar(
            x=picks["ticker"],
            y=picks["confidence_pct"],
            marker=dict(
                color=picks["confidence_pct"],
                colorscale=[[0, "#1a2744"], [0.5, C["blue"]],
                             [1, C["green"]]],
                showscale=False,
                line=dict(width=0),
            ),
            text=picks["probability_pct"].apply(lambda x: f"{x:.0f}%"),
            textposition="outside",
            textfont=dict(size=10, color=C["subtext"]),
            hovertemplate=(
                "%{x}<br>"
                "Confidence: %{y:.1f}%<br>"
                "<extra></extra>"
            ),
        ))
        layout_picks = {**PLOTLY_BASE}
        layout_picks.update(
            height=380,
            title=dict(
                text="Top Picks by Confidence",
                font=dict(size=15, color=C["text"]), x=0,
            ),
            xaxis=dict(gridcolor=C["border"], showgrid=False,
                       tickfont=dict(size=11)),
            yaxis=dict(gridcolor=C["border"],
                       title="Confidence %", tickfont=dict(size=11)),
            showlegend=False,
        )
        fig_picks.update_layout(**layout_picks)
        st.plotly_chart(fig_picks, use_container_width=True)

        # Full table
        section_header("Full Rankings")
        display_picks = picks[[
            "ticker", "sector", "probability_pct",
            "confidence_pct", "rsi_14", "price_vs_52w_high"
        ]].copy()
        display_picks.columns = [
            "Ticker", "Sector", "Probability %",
            "Confidence %", "RSI (14)", "vs 52W High"
        ]
        display_picks["vs 52W High"] = (
            display_picks["vs 52W High"] * 100
        ).round(1).astype(str) + "%"
        display_picks["RSI (14)"] = display_picks["RSI (14)"].round(1)
        st.dataframe(display_picks, use_container_width=True,
                     hide_index=True)
