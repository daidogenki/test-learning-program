import streamlit as st
import pandas as pd
import duckdb
import json
import plotly.express as px
import plotly.graph_objects as go
from openai import OpenAI
import time
import re
from typing import Optional, Dict, Any, Tuple, List

# OpenAIクライアント初期化
client = OpenAI()

# アプリタイトル
st.title('SQL自動生成データ分析チャットボット')

# 1. データローダー（CSV→DuckDB VIEW）
@st.cache_resource
def setup_duckdb():
    """CSVデータの読み込みとDuckDB VIEW作成"""
    try:
        # DuckDB接続
        conn = duckdb.connect(':memory:')
        
        # CSV読み込み
        df = pd.read_csv('data/sample_sales.csv')
        
        # 前処理
        df['date'] = pd.to_datetime(df['date'])
        df['revenue'] = df['revenue'].fillna(df['units'] * df['unit_price'])
        
        # 派生列追加
        df['month'] = df['date'].dt.to_period("M").dt.to_timestamp()
        df['quarter'] = df['date'].dt.to_period("Q").dt.start_time
        
        # DuckDBにDataFrame登録
        conn.register('sales_df', df)
        
        # VIEWを作成（固定スキーマ）
        conn.execute("""
            CREATE OR REPLACE VIEW sales AS
            SELECT 
                date,
                month,
                quarter,
                category,
                units,
                unit_price,
                region,
                sales_channel,
                customer_segment,
                revenue
            FROM sales_df
        """)
        
        return conn, df
    except Exception as e:
        st.error(f"データ読み込みエラー: {e}")
        return None, None

# 2. LLM SQL生成（厳密なプロンプト）
def generate_sql(question: str) -> Optional[str]:
    """自然言語からSQL文を生成"""
    
    system_prompt = """あなたはデータ分析のための SQL 生成器です。次のルールを厳守して SQL だけを出力してください。

[データスキーマ]
- テーブル: sales
- 列: date, month, quarter, category, units, unit_price, region, sales_channel, customer_segment, revenue
- すべて SELECT のみ。DDL/DML/PRAGMA/ATTACH/COPY/設定変更は禁止。
- 外部テーブルやサブクエリでの書き込みは使用禁止。
- 返す行数は必ず LIMIT 1000 以内。

[表現と規約]
- 時系列の集計は、time 列を返す: 
  - 例: SELECT month AS time, SUM(revenue) AS value FROM sales GROUP BY 1 ORDER BY 1 LIMIT 1000
- 名義集計は、key/value の2列を返す:
  - 例: SELECT region AS key, SUM(revenue) AS value FROM sales GROUP BY 1 ORDER BY 2 DESC LIMIT 1000
- 軸が2つ以上必要な場合（例: 月×カテゴリ）は、time, <dimension>, value の形で返す:
  - 例: SELECT month AS time, category, SUM(revenue) AS value FROM sales GROUP BY 1,2 ORDER BY 1,2 LIMIT 1000
- 合計の指標が曖昧な場合は SUM(revenue) を既定とする。
- 期間指定やフィルタが自然文に含まれる場合、WHERE 句に正しく反映する。
- 句の順序は SELECT → FROM → WHERE → GROUP BY → HAVING → ORDER BY → LIMIT とする。

[テンプレ例]
-- 月×カテゴリ別の売上合計
SELECT month AS time, category, SUM(revenue) AS value
FROM sales
GROUP BY 1,2
ORDER BY 1,2
LIMIT 1000;

-- 地域ごとの売上合計
SELECT region AS key, SUM(revenue) AS value
FROM sales
GROUP BY 1
ORDER BY 2 DESC
LIMIT 1000;

-- South 地域の四半期別売上推移
SELECT quarter AS time, SUM(revenue) AS value
FROM sales
WHERE region = 'South'
GROUP BY 1
ORDER BY 1
LIMIT 1000;

出力は SQL のみ。説明文、コードブロック、補足は禁止。"""

    try:
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": question}
            ],
            temperature=0
        )
        
        sql = response.choices[0].message.content.strip()
        
        # SQLコードブロックから抽出
        if "```sql" in sql:
            sql = sql.split("```sql")[1].split("```")[0].strip()
        elif "```" in sql:
            sql = sql.split("```")[1].strip()
        
        # 末尾セミコロン削除
        sql = sql.rstrip(';')
        
        return sql
        
    except Exception as e:
        st.error(f"SQL生成エラー: {e}")
        return None

# 3. SQL安全ガード
def is_safe_sql(sql: str) -> Tuple[bool, str]:
    """SQLの安全性をチェック"""
    if not sql:
        return False, "SQL文が空です"
    
    # 危険な句をチェック
    dangerous_keywords = [
        'insert', 'update', 'delete', 'drop', 'create', 'alter', 
        'attach', 'copy', 'pragma', 'vacuum', 'set', 'load', 'call'
    ]
    
    sql_lower = sql.lower()
    for keyword in dangerous_keywords:
        if keyword in sql_lower:
            return False, f"禁止されたSQL句が含まれています: {keyword.upper()}"
    
    # SELECT文で始まることを確認
    if not sql_lower.strip().startswith('select'):
        return False, "SELECT文のみが許可されています"
    
    return True, ""

def enforce_limit(sql: str, max_rows: int = 1000) -> str:
    """LIMIT句を強制"""
    sql_lower = sql.lower()
    if 'limit' not in sql_lower:
        sql += f" LIMIT {max_rows}"
    return sql

def precheck_sql(conn, sql: str) -> Tuple[bool, str]:
    """SQLの事前チェック（EXPLAIN実行）"""
    try:
        conn.execute(f"EXPLAIN {sql}")
        return True, ""
    except Exception as e:
        return False, f"SQL構文エラー: {str(e)}"

# 4. DuckDB実行エンジン
def run_sql(conn, sql: str) -> Tuple[Optional[pd.DataFrame], Dict[str, Any], str]:
    """SQLを実行して結果と統計を返す"""
    try:
        start_time = time.time()
        
        # 実行
        result = conn.execute(sql).fetchdf()
        
        elapsed_ms = int((time.time() - start_time) * 1000)
        
        # 統計情報
        stats = {
            "rows": len(result),
            "columns": len(result.columns),
            "elapsed_ms": elapsed_ms,
            "schema": [(col, str(result[col].dtype)) for col in result.columns]
        }
        
        return result, stats, ""
        
    except Exception as e:
        return None, {}, f"SQL実行エラー: {str(e)}"

# 5. 可視化（自動推定）
def infer_chart(df: pd.DataFrame) -> Dict[str, Any]:
    """DataFrameからチャート仕様を自動推定"""
    if df.empty:
        return {"type": "table"}
    
    cols = set(df.columns.str.lower())
    
    # time/value規約（時系列）
    if {'time', 'value'}.issubset(cols):
        other_cols = [col for col in df.columns if col.lower() not in {'time', 'value'}]
        
        chart_spec = {
            "type": "line",
            "x": "time",
            "y": "value",
            "title": "時系列推移"
        }
        
        # 系列軸がある場合
        if other_cols:
            chart_spec["color"] = other_cols[0]
            chart_spec["title"] = f"{other_cols[0]}別時系列推移"
        
        return chart_spec
    
    # key/value規約（名義集計）
    elif {'key', 'value'}.issubset(cols):
        return {
            "type": "bar",
            "x": "key", 
            "y": "value",
            "title": "集計結果"
        }
    
    # その他は表のみ
    else:
        return {"type": "table"}

def create_chart(df: pd.DataFrame, chart_spec: Dict[str, Any]):
    """チャート仕様に基づいてPlotlyチャートを生成"""
    try:
        if chart_spec["type"] == "line":
            if "color" in chart_spec:
                fig = px.line(df, x=chart_spec["x"], y=chart_spec["y"], 
                             color=chart_spec["color"], title=chart_spec["title"],
                             markers=True)
            else:
                fig = px.line(df, x=chart_spec["x"], y=chart_spec["y"], 
                             title=chart_spec["title"], markers=True)
        
        elif chart_spec["type"] == "bar":
            fig = px.bar(df, x=chart_spec["x"], y=chart_spec["y"], 
                        title=chart_spec["title"])
        
        else:
            return None
        
        # 数値フォーマット
        fig.update_layout(
            xaxis_title=chart_spec["x"],
            yaxis_title=chart_spec["y"],
            showlegend=True if "color" in chart_spec else False
        )
        
        return fig
    
    except Exception:
        return None

# 6. エラーハンドリング
def generate_error_response(error_msg: str, question: str) -> str:
    """エラーメッセージと代替案を生成"""
    
    # 一般的な代替案
    alternatives = [
        "• 期間を指定してみてください（例：2025年1月の売上）",
        "• 粒度を変えてみてください（日次→月次、月次→四半期）", 
        "• 地域やカテゴリで絞り込んでみてください",
        "• 利用可能な列: category, region, sales_channel, customer_segment",
        "• 利用可能な指標: revenue（売上）, units（数量）, unit_price（単価）"
    ]
    
    # エラー別の具体的な提案
    if "未知" in error_msg or "存在しない" in error_msg:
        alternatives.insert(0, "• 列名をご確認ください。利用可能な列名は上記をご参照ください")
    
    if "0件" in error_msg or "該当" in error_msg:
        alternatives.insert(0, "• 条件を緩めるか期間を広げてみてください")
    
    response = f"**エラー**: {error_msg}\n\n**代替案**:\n" + "\n".join(alternatives)
    
    return response

# 7. メインアプリケーション
def main():
    # データセットアップ
    conn, df = setup_duckdb()
    if conn is None or df is None:
        st.error("データの初期化に失敗しました。")
        return
    
    # サイドバー
    with st.sidebar:
        st.header("データベース情報")
        st.write(f"レコード数: {len(df):,} 件")
        st.write(f"期間: {df['date'].min().strftime('%Y-%m-%d')} ~ {df['date'].max().strftime('%Y-%m-%d')}")
        
        st.subheader("スキーマ")
        st.write("**sales テーブル**")
        schema_info = [
            "• date: 取引日",
            "• month: 月初タイムスタンプ", 
            "• quarter: 四半期開始日",
            "• category: 商品カテゴリ",
            "• units: 販売数量",
            "• unit_price: 単価",
            "• region: 地域",
            "• sales_channel: 販売チャネル",
            "• customer_segment: 顧客セグメント", 
            "• revenue: 売上金額"
        ]
        for info in schema_info:
            st.write(info)
            
        st.subheader("サンプル値")
        for col in ['category', 'region', 'sales_channel', 'customer_segment']:
            unique_vals = df[col].unique()[:3]
            st.write(f"**{col}**: {', '.join(unique_vals)}")
        
        if st.button("データサンプル表示"):
            st.dataframe(df.head())
        
        st.subheader("質問例")
        examples = [
            "カテゴリ別に月毎の売上合計を折れ線で",
            "チャネルごとの売上合計", 
            "地域ごとの売上合計",
            "South地域の四半期別売上推移",
            "2025年1月の上位3カテゴリ",
            "平均単価の推移（チャネル別）"
        ]
        for example in examples:
            if st.button(f"📝 {example}", key=f"example_{example}"):
                st.session_state.example_query = example
    
    # チャット履歴の初期化
    if "messages" not in st.session_state:
        st.session_state.messages = []
    
    # 履歴表示  
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            
            if message["role"] == "assistant" and "data" in message:
                data = message["data"]
                
                # SQL表示
                if "sql" in data:
                    with st.expander("生成されたSQL"):
                        st.code(data["sql"], language="sql")
                
                # 統計情報
                if "stats" in data:
                    stats = data["stats"] 
                    st.write(f"📊 実行時間: {stats['elapsed_ms']}ms | 行数: {stats['rows']} | 列数: {stats['columns']}")
                
                # 結果表示
                if "result" in data and data["result"] is not None:
                    st.subheader("実行結果（表）")
                    st.dataframe(data["result"], use_container_width=True)
                    
                    # チャート表示
                    if "chart" in data and data["chart"] is not None:
                        st.subheader("グラフ")
                        st.plotly_chart(data["chart"], use_container_width=True)
    
    # ユーザー入力処理
    user_input = st.chat_input("例: カテゴリ別に月毎の売上合計を折れ線で")
    
    # 例文クリック時の処理
    if "example_query" in st.session_state:
        user_input = st.session_state.example_query
        del st.session_state.example_query
    
    if user_input:
        # ユーザーメッセージ追加
        st.session_state.messages.append({"role": "user", "content": user_input})
        with st.chat_message("user"):
            st.markdown(user_input)
        
        # アシスタント応答
        with st.chat_message("assistant"):
            with st.spinner("SQLを生成中..."):
                sql = generate_sql(user_input)
            
            if not sql:
                error_msg = generate_error_response("SQL生成に失敗しました", user_input)
                st.error(error_msg)
                st.session_state.messages.append({"role": "assistant", "content": error_msg})
                return
            
            # 安全チェック
            is_safe, safety_error = is_safe_sql(sql)
            if not is_safe:
                error_msg = generate_error_response(safety_error, user_input)
                st.error(error_msg)
                st.session_state.messages.append({"role": "assistant", "content": error_msg})
                return
            
            # LIMIT強制
            sql = enforce_limit(sql)
            
            # 事前チェック
            precheck_ok, precheck_error = precheck_sql(conn, sql)
            if not precheck_ok:
                error_msg = generate_error_response(precheck_error, user_input)
                st.error(error_msg)
                st.session_state.messages.append({"role": "assistant", "content": error_msg})
                return
            
            # SQL表示
            with st.expander("生成されたSQL"):
                st.code(sql, language="sql")
            
            # SQL実行
            with st.spinner("データを処理中..."):
                result_df, stats, exec_error = run_sql(conn, sql)
            
            if exec_error:
                error_msg = generate_error_response(exec_error, user_input)
                st.error(error_msg)
                st.session_state.messages.append({"role": "assistant", "content": error_msg})
                return
            
            if result_df is None or result_df.empty:
                error_msg = generate_error_response("該当するデータが0件でした", user_input)
                st.warning(error_msg)
                st.session_state.messages.append({
                    "role": "assistant", 
                    "content": error_msg,
                    "data": {"sql": sql, "stats": stats}
                })
                return
            
            # 統計表示
            st.write(f"📊 実行時間: {stats['elapsed_ms']}ms | 行数: {stats['rows']} | 列数: {stats['columns']}")
            
            # 結果表示（表は必須）
            st.subheader("実行結果（表）")
            st.dataframe(result_df, use_container_width=True)
            
            # チャート生成
            chart_spec = infer_chart(result_df)
            chart = None
            if chart_spec["type"] != "table":
                chart = create_chart(result_df, chart_spec)
                if chart:
                    st.subheader("グラフ")
                    st.plotly_chart(chart, use_container_width=True)
                else:
                    st.info("グラフ生成はできませんが、上記の表で結果をご確認いただけます。")
            else:
                st.info("このクエリの結果は表形式での表示が最適です。")
            
            # 結果の要約
            summary = f"クエリが正常に実行されました。{stats['rows']}行のデータを取得しました。"
            
            # 履歴に保存
            st.session_state.messages.append({
                "role": "assistant",
                "content": summary,
                "data": {
                    "sql": sql,
                    "result": result_df,
                    "chart": chart,
                    "stats": stats
                }
            })

if __name__ == "__main__":
    main()