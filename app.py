import pandas as pd
import streamlit as st
from datetime import datetime
import plotly.express as px
from api import get_weather_data

st.title("天気予報ダッシュボード")

city = st.selectbox("都市を選択", ["Tokyo", "Osaka", "Kyoto", "Yokohama"])

if st.button("天気データを取得"):
    with st.spinner("データを取得中..."):
        try:
            times, temps = get_weather_data(city)
            
            df = pd.DataFrame({
                'time': pd.to_datetime(times),
                'temperature': temps
            })
            
            st.success(f"{city}の天気データを取得しました！")
            
            st.subheader("温度の時系列グラフ")
            fig = px.line(df, x='time', y='temperature', 
                         title=f'{city}の時間別気温予報',
                         labels={'time': '時間', 'temperature': '気温 (°C)'})
            st.plotly_chart(fig, use_container_width=True)
            
            st.subheader("統計情報")
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("最高気温", f"{max(temps):.1f}°C")
            with col2:
                st.metric("最低気温", f"{min(temps):.1f}°C")
            with col3:
                st.metric("平均気温", f"{sum(temps)/len(temps):.1f}°C")
            with col4:
                st.metric("データ数", len(temps))
            
            st.subheader("詳細データ")
            st.dataframe(df)
            
        except Exception as e:
            st.error(f"エラーが発生しました: {e}")

st.divider()

# ─────────────────────────────
# 元のダッシュボード
# ─────────────────────────────
st.title("📊 Sample Sales Dashboard")

# ─────────────────────────────
# データ読み込み
# ─────────────────────────────
df_sales = pd.read_csv("data/sample_sales.csv", parse_dates=["date"])

# ─────────────────────────────
# UI ― フィルター類
# ─────────────────────────────

min_date = df_sales["date"].min().to_pydatetime()
max_date = df_sales["date"].max().to_pydatetime()

date_range = st.slider(
    "期間を選択",
    min_value=min_date,
    max_value=max_date,
    value=(min_date, max_date),
    format="YYYY-MM-DD",
)

cats = st.multiselect(
    "カテゴリを選択（複数可）",
    options=df_sales["category"].unique().tolist(),
    default=df_sales["category"].unique().tolist(),
)
regions = st.multiselect(
    "地域を選択（複数可）",
    options=df_sales["region"].unique().tolist(),
    default=df_sales["region"].unique().tolist(),
)
channels = st.multiselect(
    "チャネルを選択（複数可）",
    options=df_sales["sales_channel"].unique().tolist(),
    default=df_sales["sales_channel"].unique().tolist(),
)

# ─────────────────────────────
# フィルタリング
# ─────────────────────────────
start_dt = pd.to_datetime(date_range[0])
end_dt   = pd.to_datetime(date_range[1])

df_filt = df_sales[
    (df_sales["date"].between(start_dt, end_dt))
    & (df_sales["category"].isin(cats))
    & (df_sales["region"].isin(regions))
    & (df_sales["sales_channel"].isin(channels))
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
