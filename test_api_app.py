import requests
import streamlit as st
import pandas as pd
import plotly.express as px

def get_weather_data(city_name="Tokyo"):
    # 1) ジオコーディング
    geo = requests.get(
        "https://geocoding-api.open-meteo.com/v1/search",
        params={"name": city_name, "count": 1, "language": "ja", "format": "json"}
    ).json()
    lat = geo["results"][0]["latitude"]
    lon = geo["results"][0]["longitude"]

    # 2) 予報取得
    forecast = requests.get(
        "https://api.open-meteo.com/v1/forecast",
        params={"latitude": lat, "longitude": lon, "hourly": "temperature_2m", "timezone": "Asia/Tokyo"}
    ).json()

    times = forecast["hourly"]["time"]
    temps = forecast["hourly"]["temperature_2m"]
    
    return times, temps

def main():
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

if __name__ == "__main__":
    main()
