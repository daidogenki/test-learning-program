import requests

def get_weather_data(city_name="Tokyo"):
    # 1) Geocoding API
    geo_response = requests.get(
        "https://geocoding-api.open-meteo.com/v1/search",
        params={"name": city_name, "count": 1, "language": "ja", "format": "json"}
    )
    geo_data = geo_response.json()
    lat = geo_data["results"][0]["latitude"]
    lon = geo_data["results"][0]["longitude"]
    
    print(f"City: {city_name}")
    print(f"Latitude: {lat}, Longitude: {lon}")
    
    # 2) Weather forecast API
    forecast_response = requests.get(
        "https://api.open-meteo.com/v1/forecast",
        params={"latitude": lat, "longitude": lon, "hourly": "temperature_2m", "timezone": "Asia/Tokyo"}
    )
    forecast_data = forecast_response.json()
    
    times = forecast_data["hourly"]["time"]
    temps = forecast_data["hourly"]["temperature_2m"]
    
    print(f"Data count: {len(times)}")
    print(f"First 5 time and temperature data:")
    for i in range(5):
        print(f"  {times[i]}: {temps[i]}°C")
    
    return times, temps

if __name__ == "__main__":
    try:
        times, temps = get_weather_data("Tokyo")
        print("\nData fetch completed")
    except Exception as e:
        print(f"Error: {e}")