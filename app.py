import pandas as pd
import streamlit as st
from datetime import datetime
import plotly.express as px

# ─────────────────────────────
# データ読み込み
# ─────────────────────────────
df = pd.read_csv("data/sample_sales.csv", parse_dates=["date"])

# ─────────────────────────────
# UI ― フィルター類
# ─────────────────────────────
st.title("📊 Sample Sales Dashboard")

min_date = df["date"].min().to_pydatetime()
max_date = df["date"].max().to_pydatetime()

date_range = st.slider(
    "期間を選択",
    min_value=min_date,
    max_value=max_date,
    value=(min_date, max_date),
    format="YYYY-MM-DD",
)

cats = st.multiselect(
    "カテゴリを選択（複数可）",
    options=df["category"].unique().tolist(),
    default=df["category"].unique().tolist(),
)
regions = st.multiselect(
    "地域を選択（複数可）",
    options=df["region"].unique().tolist(),
    default=df["region"].unique().tolist(),
)
channels = st.multiselect(
    "チャネルを選択（複数可）",
    options=df["sales_channel"].unique().tolist(),
    default=df["sales_channel"].unique().tolist(),
)

# ─────────────────────────────
# フィルタリング
# ─────────────────────────────
start_dt = pd.to_datetime(date_range[0])
end_dt   = pd.to_datetime(date_range[1])

df_filt = df[
    (df["date"].between(start_dt, end_dt))
    & (df["category"].isin(cats))
    & (df["region"].isin(regions))
    & (df["sales_channel"].isin(channels))
]

# ─────────────────────────────
# KPI
# ─────────────────────────────
total_revenue  = int(df_filt["revenue"].sum())
total_units    = int(df_filt["units"].sum())
avg_unit_price = int(df_filt["unit_price"].mean()) if not df_filt.empty else 0

col1, col2, col3 = st.columns(3)
col1.metric("売上合計 (円)", f"{total_revenue:,.0f}")
col2.metric("販売数量 (個)", f"{total_units:,}")
col3.metric("平均単価 (円)", f"{avg_unit_price:,.0f}")

st.divider()

# ─────────────────────────────
# Plotly でチャート描画
# ─────────────────────────────

# 1) 日別売上推移
revenue_daily = (
    df_filt.groupby("date", as_index=False)["revenue"].sum().sort_values("date")
)
fig_daily = px.line(
    revenue_daily,
    x="date",
    y="revenue",
    markers=True,
    labels={"date": "日付", "revenue": "売上 (円)"},
    title="🗓️ 日別売上推移",
)
fig_daily.update_layout(height=350, hovermode="x unified")
st.plotly_chart(fig_daily, use_container_width=True)

# 2) カテゴリ別売上
revenue_by_cat = (
    df_filt.groupby("category", as_index=False)["revenue"].sum().sort_values("revenue")
)
fig_cat = px.bar(
    revenue_by_cat,
    x="category",
    y="revenue",
    text_auto=".2s",
    labels={"category": "カテゴリ", "revenue": "売上 (円)"},
    title="🏷️ カテゴリ別売上",
)
fig_cat.update_layout(height=350)
st.plotly_chart(fig_cat, use_container_width=True)

# 3) 地域別売上
revenue_by_region = (
    df_filt.groupby("region", as_index=False)["revenue"].sum().sort_values("revenue")
)
fig_region = px.bar(
    revenue_by_region,
    x="region",
    y="revenue",
    text_auto=".2s",
    labels={"region": "地域", "revenue": "売上 (円)"},
    title="🌎 地域別売上",
)
fig_region.update_layout(height=350)
st.plotly_chart(fig_region, use_container_width=True)

st.divider()

# ─────────────────────────────
# 明細テーブル
# ─────────────────────────────
with st.expander("📄 フィルタ後データを表示"):
    st.dataframe(df_filt.reset_index(drop=True), use_container_width=True)
