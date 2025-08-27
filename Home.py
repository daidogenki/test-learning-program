import streamlit as st
import pandas as pd
import plotly.express as px

st.set_page_config(page_title="Streamlit BI x Claude Code Starter", layout="wide")

st.title("Streamlit BI x Claude Code Starter")

@st.cache_data
def load_data():
    try:
        orders_df = pd.read_csv("sample_data/orders.csv")
        users_df = pd.read_csv("sample_data/users.csv")
        return orders_df, users_df
    except FileNotFoundError as e:
        st.error(f"ファイルが見つかりません: {e}")
        return None, None

@st.cache_data
def process_monthly_data(orders_df):
    if orders_df is None:
        return None, None
    
    orders_df['created_at'] = pd.to_datetime(orders_df['created_at'])
    orders_df['year_month'] = orders_df['created_at'].dt.to_period('M').astype(str)
    
    monthly_summary = orders_df.groupby('year_month').agg({
        'order_id': 'count',
        'status': lambda x: (x == 'Cancelled').sum()
    }).rename(columns={'order_id': 'total_orders', 'status': 'cancelled_orders'})
    
    monthly_summary['cancellation_rate'] = (
        monthly_summary['cancelled_orders'] / monthly_summary['total_orders'] * 100
    )
    
    return monthly_summary, orders_df

@st.cache_data
def process_regional_data(orders_df, users_df):
    if orders_df is None or users_df is None:
        return None
    
    # データを結合
    merged_df = orders_df.merge(users_df, left_on='user_id', right_on='id', how='inner')
    
    # 地域別のキャンセル率を計算
    regional_summary = merged_df.groupby('state').agg({
        'order_id': 'count',
        'status': lambda x: (x == 'Cancelled').sum()
    }).rename(columns={'order_id': 'total_orders', 'status': 'cancelled_orders'})
    
    regional_summary['cancellation_rate'] = (
        regional_summary['cancelled_orders'] / regional_summary['total_orders'] * 100
    )
    
    # 注文数でソート（降順）
    regional_summary = regional_summary.sort_values('total_orders', ascending=False)
    
    return regional_summary

orders_df, users_df = load_data()

if orders_df is not None and users_df is not None:
    monthly_summary, processed_orders = process_monthly_data(orders_df)
    regional_summary = process_regional_data(orders_df, users_df)
    
    if monthly_summary is not None:
        st.header("月別分析")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("月別オーダー数")
            fig_orders = px.bar(
                x=monthly_summary.index,
                y=monthly_summary['total_orders'],
                title="月別オーダー数推移",
                labels={'x': '年月', 'y': 'オーダー数'},
                color=monthly_summary['total_orders'],
                color_continuous_scale='Blues'
            )
            fig_orders.update_layout(
                xaxis_title="年月",
                yaxis_title="オーダー数",
                showlegend=False
            )
            st.plotly_chart(fig_orders, use_container_width=True)
        
        with col2:
            st.subheader("月別キャンセル率")
            fig_cancel = px.line(
                x=monthly_summary.index,
                y=monthly_summary['cancellation_rate'],
                title="月別キャンセル率推移",
                labels={'x': '年月', 'y': 'キャンセル率 (%)'},
                markers=True
            )
            fig_cancel.update_traces(line_color='red', marker_color='red')
            fig_cancel.update_layout(
                xaxis_title="年月",
                yaxis_title="キャンセル率 (%)"
            )
            st.plotly_chart(fig_cancel, use_container_width=True)
    
    if regional_summary is not None:
        st.header("地域別分析")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("地域別キャンセル率")
            # 上位20地域のみ表示
            top_regions = regional_summary.head(20)
            fig_regional = px.bar(
                x=top_regions['cancellation_rate'],
                y=top_regions.index,
                orientation='h',
                title="地域別キャンセル率 (上位20地域)",
                labels={'x': 'キャンセル率 (%)', 'y': '地域'},
                color=top_regions['cancellation_rate'],
                color_continuous_scale='Reds'
            )
            fig_regional.update_layout(
                height=600,
                xaxis_title="キャンセル率 (%)",
                yaxis_title="地域",
                showlegend=False
            )
            st.plotly_chart(fig_regional, use_container_width=True)
        
        with col2:
            st.subheader("地域別オーダー数")
            top_regions_orders = regional_summary.head(15)
            fig_regional_orders = px.bar(
                x=top_regions_orders.index,
                y=top_regions_orders['total_orders'],
                title="地域別オーダー数 (上位15地域)",
                labels={'x': '地域', 'y': 'オーダー数'},
                color=top_regions_orders['total_orders'],
                color_continuous_scale='Greens'
            )
            fig_regional_orders.update_layout(
                xaxis_title="地域",
                yaxis_title="オーダー数",
                showlegend=False,
                xaxis={'tickangle': 45}
            )
            st.plotly_chart(fig_regional_orders, use_container_width=True)
    
    st.header("Orders Data (Top 10 rows)")
    st.dataframe(orders_df.head(10))
    
    st.header("Users Data (Top 10 rows)")
    st.dataframe(users_df.head(10))
else:
    st.error("データの読み込みに失敗しました。ファイルパスを確認してください。")