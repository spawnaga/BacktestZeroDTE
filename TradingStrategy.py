import os
import pandas as pd
import requests
import databento as db
import numpy as np
from datetime import datetime, timedelta

# Replace these with your actual API keys
DATABENTO_API_KEY = ""
POLYGON_API_KEY = "Tg0lvrLsHAfa3S0WutQcug5FpNGLNQRd"

TICKER = ""
START_DATE = "2023-01-01"
END_DATE = "2024-05-27"
INTERVAL = "1m"

cache_dir = "./cache"
os.makedirs(cache_dir, exist_ok=True)

def construct_ticker_symbol(underlying, expiration_date, strike_price, option_type):
    expiration_date_str = expiration_date.strftime('%y%m%d')
    strike_price = round(strike_price / 10) * 10
    strike_price_str = f"{strike_price:05d}000"
    ticker_symbol = f"O:{underlying}{expiration_date_str}{option_type}{strike_price_str}"
    return ticker_symbol, strike_price

class TradingStrategy:
    def __init__(self, databento_api_key, polygon_api_key):
        self.databento_client = db.Historical(databento_api_key)
        self.polygon_api_key = polygon_api_key
        self.expiration_date = None
        self.idx = None

    def get_ohlcv_data(self, filename):
        ohlcv_data = pd.read_csv(filename, index_col='timestamp', parse_dates=True)
        ohlcv_data = ohlcv_data[~ohlcv_data.index.duplicated(keep='first')]  # Ensure unique index
        return ohlcv_data

    def fetch_ohlcv_data_databento(self, ticker, start_date, end_date, interval):
        cache_file = os.path.join(cache_dir, f"{ticker}_ohlcv_{start_date}_{end_date}.csv")
        if os.path.exists(cache_file):
            print(f"Loading OHLCV data from {cache_file}")
            return pd.read_csv(cache_file, index_col='ts_event', parse_dates=True)

        else:

            print(f"Fetching OHLCV data for {ticker} from {start_date} to {end_date}")
            all_data = []
            current_start_date = datetime.strptime(start_date, "%Y-%m-%d")
            end_date_dt = datetime.strptime(end_date, "%Y-%m-%d")

            data = self.databento_client.timeseries.get_range(
                dataset="GLBX.MDP3",
                stype_in='parent',
                symbols=ticker,
                start=current_start_date.strftime("%Y-%m-%d"),
                end=end_date_dt.strftime("%Y-%m-%d"),
                schema="ohlcv-1m"
            ).to_df()

        return data

    def fetch_option_data_polygon(self, ticker, strike_price, expiration_date, start_date, end_date, option_type,
                                  interval='1/minute'):
        cache_file = os.path.join(cache_dir,
                                  f"{ticker}_{strike_price}_{expiration_date}_{option_type}_{start_date}_{end_date}.csv")

        if os.path.exists(cache_file):
            print(f"Loading data from {cache_file}")
            return pd.read_csv(cache_file, index_col='timestamp', parse_dates=True)

        print(f"Fetching data for {ticker} from {start_date} to {end_date}")
        url = f"https://api.polygon.io/v2/aggs/ticker/{ticker}/range/{interval}/{start_date}/{end_date}?apiKey={self.polygon_api_key}"
        response = requests.get(url)
        if response.status_code == 200:
            data = response.json()
            if data['queryCount'] > 0 and data['resultsCount'] > 0:
                results = data['results']
                df = pd.DataFrame(results)
                df['timestamp'] = pd.to_datetime(df['t'], unit='ms')
                df.set_index('timestamp', inplace=True)
                df.rename(columns={'o': 'open', 'h': 'high', 'l': 'low', 'c': 'close', 'v': 'volume'}, inplace=True)
                df = df[~df.index.duplicated(keep='first')]  # Ensure unique index

                # Check if high and low prices are both 0.05
                if (df['high'] == 0.05).all() and (df['low'] == 0.05).all():
                    print(f"Prices for {ticker} are 0.05, retrying with adjusted strike price.")
                    return pd.DataFrame()  # Return empty DataFrame to trigger retry

                df.to_csv(cache_file)
                return df
            else:
                print(f"No data found for ticker {ticker} in the given date range.")
                return pd.DataFrame()
        else:
            print(f"Error fetching data from Polygon.io: {response.status_code} - {response.text}")
            return pd.DataFrame()

    def fetch_option_data_with_retry(self, call_ticker, put_ticker, start_date, end_date):
        increment = 100000  # Define the increment for strike prices
        max_retries = 10

        for _ in range(max_retries):
            call_option = self.fetch_option_data_polygon(call_ticker, call_ticker.split(":")[-1].split("C")[0],
                                                         end_date, start_date, end_date, 'C')
            put_option = self.fetch_option_data_polygon(put_ticker, put_ticker.split(":")[-1].split("P")[0],
                                                        end_date, start_date, end_date, 'P')

            if not call_option.empty and not put_option.empty:
                return call_option, put_option

            # Adjust strike prices and try again
            call_strike_price = int(call_ticker.split("C")[-1]) // 1000000
            put_strike_price = int(put_ticker.split("P")[-1]) // 1000000

            call_strike_price += increment
            put_strike_price += increment

            call_ticker = construct_ticker_symbol("SPXW", pd.to_datetime(end_date), call_strike_price * 100, "C")[0]
            put_ticker = construct_ticker_symbol("SPXW", pd.to_datetime(end_date), put_strike_price * 100, "P")[0]

        return pd.DataFrame(), pd.DataFrame()

    def calculate_soonest_expiration_date(self, close_price, signal_time, trade_type, min_delta=0.3, max_delta=0.6):
        signal_time = pd.Timestamp(signal_time)
        days_until_monday = (0 - signal_time.weekday()) % 7
        days_until_wednesday = (2 - signal_time.weekday()) % 7
        days_until_friday = (4 - signal_time.weekday()) % 7
        expiration_date_monday = signal_time + pd.Timedelta(days=days_until_monday)
        expiration_date_wednesday = signal_time + pd.Timedelta(days_until_wednesday)
        expiration_date_friday = signal_time + pd.Timedelta(days_until_friday)
        soonest_expiration_date = min(expiration_date_monday, expiration_date_wednesday, expiration_date_friday)
        short_strike_price = round((close_price - 100) / 100) * 100
        long_strike_price = round((close_price + 100) / 100) * 100
        call_delta = 1.0
        put_delta = 1.0 - call_delta
        return soonest_expiration_date, short_strike_price, long_strike_price, put_delta

    def rmi(self, data, length):
        delta = data.diff()
        up = delta.where(delta > 0, 0).rolling(window=length).sum()
        down = -delta.where(delta < 0, 0).rolling(window=length).sum()
        rmi = np.where(down == 0, 100, np.where(up == 0, 0, 100 - (100 / (1 + up / down))))
        return pd.Series(rmi, index=data.index)

    def ci(self, data, length):
        high = data['high']
        low = data['low']
        close = data['close']

        # Calculate ATR
        atr = (high - low).rolling(window=length).sum()

        # Calculate the range of close prices over the rolling window
        close_range = close.rolling(window=length).max() - close.rolling(window=length).min()

        # Define a small epsilon value to prevent division by zero
        epsilon = 1e-10

        # Prevent log10(0) by ensuring the arguments to log10 are always positive
        close_range = close_range.replace(0, epsilon)
        atr = atr.replace(0, epsilon)

        # Ensure no NaN values in ATR and close_range
        if atr.isna().any() or close_range.isna().any():
            print("Warning: ATR or close_range contains NaN values.")

        choppiness = 100 * np.log10(atr / close_range) / np.log10(length)

        return choppiness

    def clean_ohlcv_data(self, data):
        # Remove rows where any OHLC value is an outlier using Z-score
        z_scores = np.abs((data - data.mean()) / data.std())
        filtered_entries = (z_scores < 2).all(axis=1)  # Keep rows where all values are within 2 standard deviations
        cleaned_data = data[filtered_entries]

        # Remove duplicate labels
        cleaned_data = cleaned_data[~cleaned_data.index.duplicated(keep='first')]

        # Ensure the index covers the full datetime range with no duplicates
        full_range = pd.date_range(start=cleaned_data.index.min(), end=cleaned_data.index.max(), freq='T')
        cleaned_data = cleaned_data.reindex(full_range).ffill()
        return cleaned_data

    def merge_option_data(self, ohlcv_df, option_data, option_type):
        # Ensure the index of option_data is a DatetimeIndex
        option_data.index = pd.to_datetime(option_data.index)
        option_data_resampled = option_data.resample('1min').last()

        # Merge the option prices with the OHLCV dataframe
        merged_df = pd.merge_asof(
            ohlcv_df.sort_index(),
            option_data_resampled[['close']].sort_index(),
            left_index=True,
            right_index=True,
            direction='nearest',
            tolerance=pd.Timedelta('1min')
        )

        if option_type == 'C':
            ohlcv_df['call_option_close'] = ohlcv_df['call_option_close'].combine_first(merged_df['close_y'])
        elif option_type == 'P':
            ohlcv_df['put_option_close'] = ohlcv_df['put_option_close'].combine_first(merged_df['close_y'])

        return ohlcv_df

    def generate_trade_signals(self, data, rmi_period=10, ci_period=14, slope_window=5, slope_threshold=1,
                               min_minutes_since_start=60, min_minutes_until_close=60,
                               min_option_price=2.50, max_trades_same_direction=3, max_spread_cost=0.20):
        global expiration_date_str
        data = self.clean_ohlcv_data(data[['open', 'high', 'low', 'close', 'volume']])
        retained_data = data.copy()
        retained_data['call_option_close'] = pd.NA
        retained_data['put_option_close'] = pd.NA
        retained_data.index = pd.to_datetime(retained_data.index)
        retained_data.index = retained_data.index.tz_localize(None) - pd.Timedelta(hours=5)
        retained_data.index = retained_data.index.tz_localize('UTC')
        rmi_period = int(round(rmi_period))
        ci_period = int(round(ci_period))
        rmi_values = self.rmi(retained_data['close'], rmi_period)
        ci_values = self.ci(retained_data, ci_period)
        slope = (rmi_values - rmi_values.shift(slope_window)) / slope_window
        market_open_time_utc = retained_data.index.normalize() + pd.Timedelta(hours=9, minutes=30)
        market_close_time_utc = retained_data.index.normalize() + pd.Timedelta(hours=16)

        short_signal = ((rmi_values.shift(1) > 90) &
                        (rmi_values < rmi_values.shift(1)) &
                        (slope.abs() > slope_threshold) &
                        (ci_values < 32) &
                        (retained_data.index >= market_open_time_utc + pd.Timedelta(minutes=min_minutes_since_start)) &
                        (retained_data.index <= market_close_time_utc - pd.Timedelta(minutes=min_minutes_until_close)) &
                        (retained_data['close'] > min_option_price))

        long_signal = ((rmi_values.shift(1) < 10) &
                       (rmi_values > rmi_values.shift(1)) &
                       (slope.abs() > slope_threshold) &
                       (ci_values > 68) &
                       (retained_data.index >= market_open_time_utc + pd.Timedelta(minutes=min_minutes_since_start)) &
                       (retained_data.index <= market_close_time_utc - pd.Timedelta(minutes=min_minutes_until_close)) &
                       (retained_data['close'] > min_option_price))

        retained_data['RMI'] = rmi_values
        retained_data['Slope'] = slope
        retained_data['CI'] = ci_values
        retained_data['Short Signal'] = short_signal
        retained_data['Long Signal'] = long_signal

        options_strategies = []
        current_trade = None

        for idx, row in retained_data.iterrows():
            if current_trade is not None:
                if self.should_close_trade(row):
                    options_strategies.append(("should exit", "should exit"))
                    current_trade = None
                else:
                    options_strategies.append(current_trade)
            else:
                if row['Short Signal'] or row['Long Signal']:
                    direction = 'short' if row['Short Signal'] else 'long'
                    expiration_date, long_strike, short_strike, put_delta = self.calculate_soonest_expiration_date(
                        row['close'], idx, direction)
                    call_ticker, call_strike = construct_ticker_symbol("SPXW", expiration_date, short_strike, "C")
                    put_ticker, put_strike = construct_ticker_symbol("SPXW", expiration_date, long_strike, "P")
                    expiration_date_str = expiration_date.strftime('%Y-%m-%d')

                    if row['Short Signal']:
                        call_entry = f"SELL STRIKE {call_strike} EXP {expiration_date_str}"
                        put_entry = f"BUY STRIKE {put_strike} EXP {expiration_date_str}"
                    else:
                        call_entry = f"BUY STRIKE {call_strike} EXP {expiration_date_str}"
                        put_entry = f"SELL STRIKE {put_strike} EXP {expiration_date_str}"

                    current_trade = (call_entry, put_entry)
                    options_strategies.append(current_trade)

                    if self.expiration_date != expiration_date_str or self.idx != idx.date():
                        call_option, put_option = self.fetch_option_data_with_retry(call_ticker, put_ticker,
                                                                                    idx.strftime('%Y-%m-%d'),
                                                                                    expiration_date_str)

                        print(f"Data for Call Option: {call_ticker}")
                        print(call_option.head())
                        print(f"Data for Put Option: {put_ticker}")
                        print(put_option.head())

                        if 'close' in call_option.columns and not call_option.empty:
                            call_option.index = pd.to_datetime(call_option.index, utc=True)
                            call_option['datetime'] = call_option.index
                        else:
                            print(f"Missing 'close' column for Call Option: {call_ticker}")

                        if 'close' in put_option.columns and not put_option.empty:
                            put_option.index = pd.to_datetime(put_option.index, utc=True)
                        else:
                            print(f"Missing 'close' column for Put Option: {put_ticker}")

                        self.expiration_date = expiration_date_str
                        self.idx = idx.date()

                    # Merge the option data
                    if not call_option.empty:
                        retained_data = self.merge_option_data(retained_data, call_option, 'C')
                    if not put_option.empty:
                        retained_data = self.merge_option_data(retained_data, put_option, 'P')
                else:
                    options_strategies.append((None, None))

        # Ensure options_strategies length matches retained_data length
        print(f"Length of options_strategies: {len(options_strategies)}, Length of retained_data: {len(retained_data)}")
        if len(options_strategies) < len(retained_data):
            options_strategies.extend([(None, None)] * (len(retained_data) - len(options_strategies)))
        elif len(options_strategies) > len(retained_data):
            options_strategies = options_strategies[:len(retained_data)]

        retained_data['Call Option'] = [strategy[0] if strategy is not None else None for strategy in
                                        options_strategies]
        retained_data['Put Option'] = [strategy[1] if strategy is not None else None for strategy in options_strategies]

        # Sync the call and put option close columns
        retained_data['call_option_close'] = np.where(retained_data['Call Option'].notnull(),
                                                      retained_data['call_option_close'], pd.NA)
        retained_data['put_option_close'] = np.where(retained_data['Put Option'].notnull(),
                                                     retained_data['put_option_close'], pd.NA)

        # Calculate PnL
        retained_data['PnL'] = retained_data.apply(lambda row: self.calculate_pnl(row), axis=1)

        # Save the final DataFrame to a CSV file in the output folder
        output_dir = "./output"
        os.makedirs(output_dir, exist_ok=True)
        retained_data.to_csv(os.path.join(output_dir, 'retained_data.csv'))

        return retained_data

    def should_close_trade(self, row):
        rmi_value = row['RMI']
        # Define your criteria for closing a trade
        if rmi_value < 20:
            return True
        elif rmi_value > 90 and rmi_value < 77:
            return True
        return False

    def fetch_option_data_with_retry(self, call_ticker, put_ticker, start_date, end_date):
        increment = 1
        # Adjust strike price and try again
        call_strike_price = int(call_ticker.split("C")[-1]) // 100000
        put_strike_price = int(put_ticker.split("P")[-1]) // 100000
        while increment <= 10:  # You can set the maximum number of retries
            call_option = self.fetch_option_data_polygon(call_ticker, call_ticker.split(":")[-1].split("C")[0],
                                                         end_date, start_date, end_date, 'C')
            put_option = self.fetch_option_data_polygon(put_ticker, put_ticker.split(":")[-1].split("P")[0],
                                                        end_date, start_date, end_date, 'P')

            if not call_option.empty and not put_option.empty:
                return call_option, put_option

            call_strike_price -= 1
            put_strike_price += 1

            call_ticker = construct_ticker_symbol("SPXW", pd.to_datetime(end_date), call_strike_price * 100, "C")[0]
            put_ticker = construct_ticker_symbol("SPXW", pd.to_datetime(end_date), put_strike_price * 100, "P")[0]

            increment += 1

        return pd.DataFrame(), pd.DataFrame()

    def calculate_pnl(self, row):
        if row['call_option_close'] is not None and row['put_option_close'] is not None:
            pnl = row['call_option_close'] - row['put_option_close']
        else:
            pnl = None
        return pnl

# Usage example
trading_strategy = TradingStrategy(DATABENTO_API_KEY, POLYGON_API_KEY)

# Fetch OHLCV data from Databento
ohlcv_data = trading_strategy.fetch_ohlcv_data_databento(TICKER, START_DATE, END_DATE, INTERVAL)

# Generate trade signals
signals = trading_strategy.generate_trade_signals(ohlcv_data)

# Display signals to verify columns are updated
print(signals[['RMI', 'Slope', 'CI', 'Short Signal', 'Long Signal', 'Call Option', 'Put Option', 'call_option_close',
               'put_option_close', 'PnL']])