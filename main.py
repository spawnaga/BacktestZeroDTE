import requests

# Your API key
api_key = 'Tg0lvrLsHAfa3S0WutQcug5FpNGLNQRd'

# Base URL for Polygon API (you might need to adjust this based on the specific endpoint for options data)
url = 'https://api.polygon.io/v2/aggs/ticker/I:SPX/options/endpoint'

# Required headers, including the edge ID and IP address for Launchpad's Edge User Insights
headers = {
    'Authorization': f'Bearer {api_key}',
    'X-Polygon-Edge-ID': 'rotdabbel@gmail.com',
    'X-Polygon-Edge-IP-Address': '57.129.1.195',
    'X-Polygon-Edge-User-Agent': 'MarketDataAnalyzer/1.0 (Python 3.8; OS Windows 10)',
    'Accept': 'application/json',  # or 'text/csv' if you prefer CSV responses
}

# Query parameters, adjust these based on the specific requirements for the options endpoint
params = {
    'date.gte': '2023-03-01',  # Start date for the options data
    'date.lte': '2024-2-29',  # End date for the options data
    # Include other relevant parameters here
}

# Making the GET request to the Polygon API
response = requests.get(url, headers=headers, params=params)

# Checking if the request was successful
if response.status_code == 200:
    # Processing the response data
    data = response.json()  # or response.text if you requested CSV format
    print(data)
else:
    print(f'Failed to get data: {response.status_code}, {response.text}')
