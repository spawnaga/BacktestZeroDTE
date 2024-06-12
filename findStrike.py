import pandas as pd
import numpy as np
import math  # Import the math module for ceil() and floor()


class ShortTermOptionsAnalyzer:
    def __init__(self, ohlcv=pd.DataFrame(), tick_size=1, risk_free_rate=0.01):
        self.ohlcv = ohlcv
        self.tick_size = tick_size
        self.risk_free_rate = risk_free_rate

    def estimate_volatility(self, start_date, end_date):
        """Estimate historical volatility of the asset."""
        relevant_data = self.ohlcv.loc[start_date:end_date]
        daily_returns = relevant_data['close'].pct_change().dropna()
        volatility = daily_returns.std() * np.sqrt(252)  # Annualizing the daily volatility
        return volatility

    def adjust_to_tick_size(self, K):
        """Adjust the given strike price K to align with the tick size."""
        if K % self.tick_size != 0:
            adjusted_K = round(K / self.tick_size) * self.tick_size
        else:
            adjusted_K = K
        return adjusted_K

    def find_near_money_strikes(self, S):
        """Find near-money strike prices adjusted to the tick size and round accordingly."""
        # Adjust the current price to the nearest tick size
        adjusted_S = self.adjust_to_tick_size(S)

        # Round call strike up and put strike down to the nearest whole number
        near_money_call_strike = math.ceil(adjusted_S)+1
        near_money_put_strike = math.floor(adjusted_S)-1

        return near_money_call_strike, near_money_put_strike

    def analyze_options(self, purchase_date, expiration_date):
        """Analyze options for given purchase and expiration dates."""
        purchase_date = pd.to_datetime(purchase_date)
        expiration_date = pd.to_datetime(expiration_date)

        # Get the closing price on the purchase date as the current price
        S = self.ohlcv.loc[purchase_date.strftime('%Y-%m-%d'), 'close'].iloc[0]

        # Find near-money strikes for call and put options
        call_strike, put_strike = self.find_near_money_strikes(S)

        # Assuming simplified deltas for near-money options
        call_delta = 0.55  # Simplified assumption
        put_delta = -0.55  # Simplified assumption

        return {'Call Strike': call_strike, 'Call Delta': call_delta,
                'Put Strike': put_strike, 'Put Delta': put_delta}

