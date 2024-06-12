import requests
import pandas as pd
from datetime import datetime, timedelta

def fetch_daily_data(date, api_key):
    base_url = "https://api.polygon.io/v2/aggs/ticker"
    ticker = "I:SPX"  # Assuming 'I:SPX' is the correct ticker for the S&P 500 Index
    start_date = date.strftime('%Y-%m-%d')
    end_date = (date + timedelta(days=1)).strftime('%Y-%m-%d')
    url = f"{base_url}/{ticker}/range/1/day/{start_date}/{end_date}?apiKey={api_key}"

    try:
        response = requests.get(url)
        response.raise_for_status()  # Raises an HTTPError if the status is 4xx, 5xx
        data = response.json()
        if 'results' in data:
            return data['results']
        else:
            print(f"No data found for {date.strftime('%Y-%m-%d')}")
            return []
    except requests.RequestException as e:
        print(f"Request failed: {e}")
        return []

def fetch_period_data(start_date, end_date, api_key):
    all_data = pd.DataFrame()
    current_date = start_date
    while current_date <= end_date:
        daily_data = fetch_daily_data(current_date, api_key)
        if daily_data:
            daily_df = pd.DataFrame(daily_data)
            all_data = pd.concat([all_data, daily_df], ignore_index=True)
        current_date += timedelta(days=1)

    if not all_data.empty:
        all_data.to_csv('spx_prices.csv', index=False)
        print("Data saved to spx_prices.csv")
    else:
        print("No data fetched.")

api_key = "Tg0lvrLsHAfa3S0WutQcug5FpNGLNQRd"
start_date = datetime(2024, 2, 1)
end_date = datetime(2024, 3, 1)

fetch_period_data(start_date, end_date, api_key)
