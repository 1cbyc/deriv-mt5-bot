import pandas as pd
import numpy as np
from typing import Dict, Tuple

class SyntheticTradingStrategy:
    """Base class for synthetic indices trading strategies (MT5 version)"""
    def __init__(self, symbol: str, timeframe: str = 'M5'):
        self.symbol = symbol
        self.timeframe = timeframe
        self.data = pd.DataFrame()

    def update_data(self, df: pd.DataFrame):
        self.data = df.copy()
        if len(self.data) > 1000:
            self.data = self.data.tail(1000)

    def get_signal(self) -> Tuple[str, float]:
        raise NotImplementedError("Subclasses must implement get_signal()")

class MovingAverageCrossover(SyntheticTradingStrategy):
    def __init__(self, symbol: str, timeframe: str = 'M5', short_window: int = 10, long_window: int = 20):
        super().__init__(symbol, timeframe)
        self.short_window = short_window
        self.long_window = long_window

    def get_signal(self) -> Tuple[str, float]:
        if len(self.data) < self.long_window:
            return "HOLD", 0.0
        short_ma = self.data['close'].rolling(window=self.short_window).mean()
        long_ma = self.data['close'].rolling(window=self.long_window).mean()
        current_short = short_ma.iloc[-1]
        current_long = long_ma.iloc[-1]
        prev_short = short_ma.iloc[-2] if len(short_ma) > 1 else current_short
        prev_long = long_ma.iloc[-2] if len(long_ma) > 1 else current_long
        if current_short > current_long and prev_short <= prev_long:
            return "BUY", 0.8
        elif current_short < current_long and prev_short >= prev_long:
            return "SELL", 0.8
        elif current_short > current_long:
            return "BUY", 0.6
        else:
            return "SELL", 0.6

class RSIStrategy(SyntheticTradingStrategy):
    def __init__(self, symbol: str, timeframe: str = 'M5', period: int = 14, oversold: int = 30, overbought: int = 70):
        super().__init__(symbol, timeframe)
        self.period = period
        self.oversold = oversold
        self.overbought = overbought

    def calculate_rsi(self) -> float:
        if len(self.data) < self.period + 1:
            return 50.0
        delta = self.data['close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=self.period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=self.period).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi.iloc[-1]

    def get_signal(self) -> Tuple[str, float]:
        if len(self.data) < self.period + 1:
            return "HOLD", 0.0
        rsi = self.calculate_rsi()
        if rsi < self.oversold:
            return "BUY", 0.9
        elif rsi > self.overbought:
            return "SELL", 0.9
        elif rsi < 50:
            return "BUY", 0.6
        else:
            return "SELL", 0.6

class BollingerBandsStrategy(SyntheticTradingStrategy):
    def __init__(self, symbol: str, timeframe: str = 'M5', period: int = 20, std_dev: float = 2.0):
        super().__init__(symbol, timeframe)
        self.period = period
        self.std_dev = std_dev

    def get_signal(self) -> Tuple[str, float]:
        if len(self.data) < self.period:
            return "HOLD", 0.0
        sma = self.data['close'].rolling(window=self.period).mean()
        std = self.data['close'].rolling(window=self.period).std()
        upper_band = sma + (std * self.std_dev)
        lower_band = sma - (std * self.std_dev)
        current_price = self.data['close'].iloc[-1]
        current_upper = upper_band.iloc[-1]
        current_lower = lower_band.iloc[-1]
        if current_price <= current_lower:
            return "BUY", 0.8
        elif current_price >= current_upper:
            return "SELL", 0.8
        else:
            return "HOLD", 0.0

class VolatilityBreakoutStrategy(SyntheticTradingStrategy):
    def __init__(self, symbol: str, timeframe: str = 'M5', period: int = 20, multiplier: float = 1.5):
        super().__init__(symbol, timeframe)
        self.period = period
        self.multiplier = multiplier

    def get_signal(self) -> Tuple[str, float]:
        if len(self.data) < self.period:
            return "HOLD", 0.0
        high_low = self.data['high'] - self.data['low']
        high_close = np.abs(self.data['high'] - self.data['close'].shift())
        low_close = np.abs(self.data['low'] - self.data['close'].shift())
        true_range = np.maximum(high_low, np.maximum(high_close, low_close))
        atr = true_range.rolling(window=self.period).mean()
        current_atr = atr.iloc[-1]
        avg_atr = atr.mean()
        if current_atr > (avg_atr * self.multiplier):
            price_change = self.data['close'].iloc[-1] - self.data['close'].iloc[-2]
            if price_change > 0:
                return "BUY", 0.7
            else:
                return "SELL", 0.7
        else:
            return "HOLD", 0.0

class MACDStrategy(SyntheticTradingStrategy):
    def __init__(self, symbol: str, timeframe: str = 'M5', fast_period: int = 12, slow_period: int = 26, signal_period: int = 9):
        super().__init__(symbol, timeframe)
        self.fast_period = fast_period
        self.slow_period = slow_period
        self.signal_period = signal_period

    def calculate_macd(self):
        if len(self.data) < self.slow_period:
            return 0.0, 0.0, 0.0
        ema_fast = self.data['close'].ewm(span=self.fast_period).mean()
        ema_slow = self.data['close'].ewm(span=self.slow_period).mean()
        macd_line = ema_fast - ema_slow
        signal_line = macd_line.ewm(span=self.signal_period).mean()
        hist = macd_line - signal_line
        return macd_line.iloc[-1], signal_line.iloc[-1], hist.iloc[-1]

    def get_signal(self) -> Tuple[str, float]:
        if len(self.data) < self.slow_period:
            return "HOLD", 0.0
        macd, signal, hist = self.calculate_macd()
        prev_macd, prev_signal, _ = self.calculate_macd() if len(self.data) > self.slow_period else (macd, signal, hist)
        if macd > signal and prev_macd <= prev_signal:
            return "BUY", 0.7
        elif macd < signal and prev_macd >= prev_signal:
            return "SELL", 0.7
        else:
            return "HOLD", 0.0

class StochasticStrategy(SyntheticTradingStrategy):
    def __init__(self, symbol: str, timeframe: str = 'M5', k_period: int = 14, d_period: int = 3, oversold: int = 20, overbought: int = 80):
        super().__init__(symbol, timeframe)
        self.k_period = k_period
        self.d_period = d_period
        self.oversold = oversold
        self.overbought = overbought

    def calculate_stochastic(self) -> tuple:
        if len(self.data) < self.k_period:
            return 50.0, 50.0
        low_min = self.data['low'].rolling(window=self.k_period).min()
        high_max = self.data['high'].rolling(window=self.k_period).max()
        k = 100 * (self.data['close'] - low_min) / (high_max - low_min)
        d = k.rolling(window=self.d_period).mean()
        return k.iloc[-1], d.iloc[-1]

    def get_signal(self) -> Tuple[str, float]:
        if len(self.data) < self.k_period:
            return "HOLD", 0.0
        k, d = self.calculate_stochastic()
        if k < self.oversold and d < self.oversold:
            return "BUY", 0.8
        elif k > self.overbought and d > self.overbought:
            return "SELL", 0.8
        else:
            return "HOLD", 0.0

class WilliamsRStrategy(SyntheticTradingStrategy):
    def __init__(self, symbol: str, timeframe: str = 'M5', period: int = 14, oversold: int = -80, overbought: int = -20):
        super().__init__(symbol, timeframe)
        self.period = period
        self.oversold = oversold
        self.overbought = overbought

    def calculate_williams_r(self) -> float:
        if len(self.data) < self.period:
            return 0.0
        high_max = self.data['high'].rolling(window=self.period).max()
        low_min = self.data['low'].rolling(window=self.period).min()
        williams_r = -100 * (high_max - self.data['close']) / (high_max - low_min)
        return williams_r.iloc[-1]

    def get_signal(self) -> Tuple[str, float]:
        if len(self.data) < self.period:
            return "HOLD", 0.0
        wr = self.calculate_williams_r()
        if wr < self.oversold:
            return "BUY", 0.8
        elif wr > self.overbought:
            return "SELL", 0.8
        else:
            return "HOLD", 0.0

class ParabolicSARStrategy(SyntheticTradingStrategy):
    def __init__(self, symbol: str, timeframe: str = 'M5', acceleration: float = 0.02, maximum: float = 0.2):
        super().__init__(symbol, timeframe)
        self.acceleration = acceleration
        self.maximum = maximum

    def calculate_sar(self) -> float:
        if len(self.data) < 2:
            return self.data['close'].iloc[-1] if len(self.data) > 0 else 0.0
        # Simple SAR implementation (not full)
        sar = self.data['close'].iloc[0]
        for i in range(1, len(self.data)):
            prev = self.data['close'].iloc[i-1]
            curr = self.data['close'].iloc[i]
            sar = sar + self.acceleration * (curr - sar)
            if abs(sar - curr) > self.maximum:
                sar = curr
        return sar

    def get_signal(self) -> Tuple[str, float]:
        if len(self.data) < 2:
            return "HOLD", 0.0
        sar = self.calculate_sar()
        price = self.data['close'].iloc[-1]
        if price > sar:
            return "BUY", 0.7
        elif price < sar:
            return "SELL", 0.7
        else:
            return "HOLD", 0.0

class IchimokuStrategy(SyntheticTradingStrategy):
    def __init__(self, symbol: str, timeframe: str = 'M5'):
        super().__init__(symbol, timeframe)

    def calculate_ichimoku(self) -> Dict[str, float]:
        if len(self.data) < 52:
            return {}
        high_9 = self.data['high'].rolling(window=9).max()
        low_9 = self.data['low'].rolling(window=9).min()
        tenkan_sen = (high_9 + low_9) / 2
        high_26 = self.data['high'].rolling(window=26).max()
        low_26 = self.data['low'].rolling(window=26).min()
        kijun_sen = (high_26 + low_26) / 2
        high_52 = self.data['high'].rolling(window=52).max()
        low_52 = self.data['low'].rolling(window=52).min()
        senkou_span_a = ((tenkan_sen + kijun_sen) / 2).shift(26)
        senkou_span_b = ((high_52 + low_52) / 2).shift(26)
        chikou_span = self.data['close'].shift(-26)
        return {
            'tenkan_sen': tenkan_sen.iloc[-1],
            'kijun_sen': kijun_sen.iloc[-1],
            'senkou_span_a': senkou_span_a.iloc[-1],
            'senkou_span_b': senkou_span_b.iloc[-1],
            'chikou_span': chikou_span.iloc[-1]
        }

    def get_signal(self) -> Tuple[str, float]:
        ichimoku = self.calculate_ichimoku()
        if not ichimoku:
            return "HOLD", 0.0
        price = self.data['close'].iloc[-1]
        if price > ichimoku['senkou_span_a'] and price > ichimoku['senkou_span_b']:
            return "BUY", 0.7
        elif price < ichimoku['senkou_span_a'] and price < ichimoku['senkou_span_b']:
            return "SELL", 0.7
        else:
            return "HOLD", 0.0

class MomentumStrategy(SyntheticTradingStrategy):
    def __init__(self, symbol: str, timeframe: str = 'M5', period: int = 10, threshold: float = 0.5):
        super().__init__(symbol, timeframe)
        self.period = period
        self.threshold = threshold

    def calculate_momentum(self) -> float:
        if len(self.data) < self.period:
            return 0.0
        return self.data['close'].iloc[-1] - self.data['close'].iloc[-self.period]

    def get_signal(self) -> Tuple[str, float]:
        if len(self.data) < self.period:
            return "HOLD", 0.0
        momentum = self.calculate_momentum()
        if momentum > self.threshold:
            return "BUY", 0.7
        elif momentum < -self.threshold:
            return "SELL", 0.7
        else:
            return "HOLD", 0.0

class MultiStrategy(SyntheticTradingStrategy):
    def __init__(self, symbol: str, timeframe: str = 'M5'):
        super().__init__(symbol, timeframe)
        self.strategies = [
            MovingAverageCrossover(symbol, timeframe),
            RSIStrategy(symbol, timeframe),
            BollingerBandsStrategy(symbol, timeframe),
            VolatilityBreakoutStrategy(symbol, timeframe),
            MACDStrategy(symbol, timeframe),
            StochasticStrategy(symbol, timeframe),
            WilliamsRStrategy(symbol, timeframe),
            ParabolicSARStrategy(symbol, timeframe),
            IchimokuStrategy(symbol, timeframe),
            MomentumStrategy(symbol, timeframe)
        ]
        
        # Strategy weights based on market conditions and reliability
        self.strategy_weights = {
            'MovingAverageCrossover': 1.0,
            'RSIStrategy': 1.2,
            'BollingerBandsStrategy': 1.1,
            'VolatilityBreakoutStrategy': 1.3,  # Higher weight for volatility indices
            'MACDStrategy': 1.0,
            'StochasticStrategy': 0.9,
            'WilliamsRStrategy': 0.9,
            'ParabolicSARStrategy': 0.8,
            'IchimokuStrategy': 1.0,
            'MomentumStrategy': 0.8
        }
        
        # Market condition detection
        self.volatility_threshold = 0.02
        self.trend_strength_threshold = 0.6

    def update_data(self, df: pd.DataFrame):
        self.data = df.copy()
        for strat in self.strategies:
            strat.update_data(df)

    def detect_market_conditions(self) -> dict:
        """Detect current market conditions to adjust strategy weights"""
        if len(self.data) < 20:
            return {'volatility': 'unknown', 'trend': 'unknown', 'regime': 'unknown'}
        
        # Calculate volatility
        returns = self.data['close'].pct_change().dropna()
        volatility = returns.rolling(window=20).std().iloc[-1]
        
        # Calculate trend strength
        short_ma = self.data['close'].rolling(window=10).mean()
        long_ma = self.data['close'].rolling(window=30).mean()
        current_short = short_ma.iloc[-1]
        current_long = long_ma.iloc[-1]
        trend_strength = abs(current_short - current_long) / current_long
        
        # Determine market regime
        if volatility > self.volatility_threshold:
            regime = 'volatile'
        elif trend_strength > self.trend_strength_threshold:
            regime = 'trending'
        else:
            regime = 'ranging'
        
        return {
            'volatility': 'high' if volatility > self.volatility_threshold else 'low',
            'trend': 'strong' if trend_strength > self.trend_strength_threshold else 'weak',
            'regime': regime
        }

    def adjust_weights_for_conditions(self, conditions: dict) -> dict:
        """Adjust strategy weights based on market conditions"""
        adjusted_weights = self.strategy_weights.copy()
        
        if conditions['regime'] == 'volatile':
            # Increase weights for volatility-based strategies
            adjusted_weights['VolatilityBreakoutStrategy'] *= 1.5
            adjusted_weights['BollingerBandsStrategy'] *= 1.3
            adjusted_weights['RSIStrategy'] *= 1.2
            # Decrease weights for trend-following strategies
            adjusted_weights['MovingAverageCrossover'] *= 0.8
            adjusted_weights['MACDStrategy'] *= 0.8
            
        elif conditions['regime'] == 'trending':
            # Increase weights for trend-following strategies
            adjusted_weights['MovingAverageCrossover'] *= 1.4
            adjusted_weights['MACDStrategy'] *= 1.3
            adjusted_weights['ParabolicSARStrategy'] *= 1.2
            # Decrease weights for mean-reversion strategies
            adjusted_weights['RSIStrategy'] *= 0.9
            adjusted_weights['StochasticStrategy'] *= 0.8
            
        else:  # ranging market
            # Increase weights for mean-reversion strategies
            adjusted_weights['RSIStrategy'] *= 1.3
            adjusted_weights['StochasticStrategy'] *= 1.2
            adjusted_weights['WilliamsRStrategy'] *= 1.2
            adjusted_weights['BollingerBandsStrategy'] *= 1.1
            # Decrease weights for trend-following strategies
            adjusted_weights['MovingAverageCrossover'] *= 0.7
            adjusted_weights['MACDStrategy'] *= 0.7
        
        return adjusted_weights

    def get_signal(self) -> Tuple[str, float]:
        """Get weighted signal based on market conditions"""
        if len(self.data) < 30:
            return "HOLD", 0.0
        
        # Detect market conditions
        conditions = self.detect_market_conditions()
        
        # Adjust weights based on conditions
        adjusted_weights = self.adjust_weights_for_conditions(conditions)
        
        # Collect signals and confidences
        signals = []
        confidences = []
        weighted_confidences = []
        
        for i, strat in enumerate(self.strategies):
            try:
                signal, confidence = strat.get_signal()
                if signal != "HOLD":
                    signals.append(signal)
                    confidences.append(confidence)
                    
                    # Apply adjusted weight
                    strategy_name = strat.__class__.__name__
                    weight = adjusted_weights.get(strategy_name, 1.0)
                    weighted_confidence = confidence * weight
                    weighted_confidences.append(weighted_confidence)
                    
            except Exception as e:
                print(f"Error in strategy {strat.__class__.__name__}: {e}")
                continue
        
        if not signals:
            return "HOLD", 0.0
        
        # Calculate weighted consensus
        buy_weighted_sum = 0
        sell_weighted_sum = 0
        total_weight = 0
        
        for signal, weighted_conf in zip(signals, weighted_confidences):
            if signal == "BUY":
                buy_weighted_sum += weighted_conf
            elif signal == "SELL":
                sell_weighted_sum += weighted_conf
            total_weight += weighted_conf
        
        # Determine final signal
        if buy_weighted_sum > sell_weighted_sum:
            final_signal = "BUY"
            final_confidence = buy_weighted_sum / total_weight if total_weight > 0 else 0
        elif sell_weighted_sum > buy_weighted_sum:
            final_signal = "SELL"
            final_confidence = sell_weighted_sum / total_weight if total_weight > 0 else 0
        else:
            final_signal = "HOLD"
            final_confidence = 0
        
        # Apply market condition multiplier
        if conditions['regime'] == 'volatile':
            final_confidence *= 0.9  # Slightly reduce confidence in volatile markets
        elif conditions['regime'] == 'trending':
            final_confidence *= 1.1  # Increase confidence in trending markets
        
        # Ensure confidence is within bounds
        final_confidence = max(0.0, min(1.0, final_confidence))
        
        return final_signal, final_confidence

class MeanReversionStrategy(SyntheticTradingStrategy):
    def __init__(self, symbol: str, timeframe: str = 'M5', period: int = 20, std_dev: float = 2.0):
        super().__init__(symbol, timeframe)
        self.period = period
        self.std_dev = std_dev

    def get_signal(self) -> Tuple[str, float]:
        if len(self.data) < self.period:
            return "HOLD", 0.0
        sma = self.data['close'].rolling(window=self.period).mean()
        std = self.data['close'].rolling(window=self.period).std()
        upper_band = sma + (std * self.std_dev)
        lower_band = sma - (std * self.std_dev)
        current_price = self.data['close'].iloc[-1]
        if current_price > upper_band.iloc[-1]:
            return "SELL", 0.7
        elif current_price < lower_band.iloc[-1]:
            return "BUY", 0.7
        else:
            return "HOLD", 0.0

class TrendFollowingStrategy(SyntheticTradingStrategy):
    def __init__(self, symbol: str, timeframe: str = 'M5', short_period: int = 10, long_period: int = 30):
        super().__init__(symbol, timeframe)
        self.short_period = short_period
        self.long_period = long_period

    def get_signal(self) -> Tuple[str, float]:
        if len(self.data) < self.long_period:
            return "HOLD", 0.0
        short_ma = self.data['close'].rolling(window=self.short_period).mean()
        long_ma = self.data['close'].rolling(window=self.long_period).mean()
        if short_ma.iloc[-1] > long_ma.iloc[-1]:
            return "BUY", 0.6
        elif short_ma.iloc[-1] < long_ma.iloc[-1]:
            return "SELL", 0.6
        else:
            return "HOLD", 0.0

class AdvancedVolatilityStrategy(SyntheticTradingStrategy):
    def __init__(self, symbol: str, timeframe: str = 'M5', period: int = 20, multiplier: float = 1.5):
        super().__init__(symbol, timeframe)
        self.period = period
        self.multiplier = multiplier

    def get_signal(self) -> Tuple[str, float]:
        if len(self.data) < self.period:
            return "HOLD", 0.0
        high_low = self.data['high'] - self.data['low']
        high_close = np.abs(self.data['high'] - self.data['close'].shift())
        low_close = np.abs(self.data['low'] - self.data['close'].shift())
        true_range = np.maximum(high_low, np.maximum(high_close, low_close))
        atr = true_range.rolling(window=self.period).mean()
        current_atr = atr.iloc[-1]
        avg_atr = atr.mean()
        if current_atr > (avg_atr * self.multiplier):
            return "BUY", 0.7
        elif current_atr < (avg_atr / self.multiplier):
            return "SELL", 0.7
        else:
            return "HOLD", 0.0

class SupportResistanceStrategy(SyntheticTradingStrategy):
    def __init__(self, symbol: str, timeframe: str = 'M5', period: int = 20):
        super().__init__(symbol, timeframe)
        self.period = period

    def get_signal(self) -> Tuple[str, float]:
        if len(self.data) < self.period:
            return "HOLD", 0.0
        recent_high = self.data['high'].rolling(window=self.period).max().iloc[-1]
        recent_low = self.data['low'].rolling(window=self.period).min().iloc[-1]
        current_price = self.data['close'].iloc[-1]
        if current_price >= recent_high:
            return "SELL", 0.7
        elif current_price <= recent_low:
            return "BUY", 0.7
        else:
            return "HOLD", 0.0

class DivergenceStrategy(SyntheticTradingStrategy):
    def __init__(self, symbol: str, timeframe: str = 'M5', period: int = 14):
        super().__init__(symbol, timeframe)
        self.period = period

    def calculate_rsi(self) -> pd.Series:
        delta = self.data['close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=self.period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=self.period).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi

    def get_signal(self) -> Tuple[str, float]:
        if len(self.data) < self.period + 1:
            return "HOLD", 0.0
        rsi = self.calculate_rsi()
        price_change = self.data['close'].iloc[-1] - self.data['close'].iloc[-2]
        rsi_change = rsi.iloc[-1] - rsi.iloc[-2]
        if price_change > 0 and rsi_change < 0:
            return "SELL", 0.7
        elif price_change < 0 and rsi_change > 0:
            return "BUY", 0.7
        else:
            return "HOLD", 0.0

class VolumePriceStrategy(SyntheticTradingStrategy):
    def __init__(self, symbol: str, timeframe: str = 'M5', period: int = 20):
        super().__init__(symbol, timeframe)
        self.period = period

    def get_signal(self) -> Tuple[str, float]:
        if len(self.data) < self.period:
            return "HOLD", 0.0
        avg_volume = self.data['volume'].rolling(window=self.period).mean().iloc[-1]
        current_volume = self.data['volume'].iloc[-1]
        if current_volume > avg_volume * 1.5:
            return "BUY", 0.6
        elif current_volume < avg_volume * 0.5:
            return "SELL", 0.6
        else:
            return "HOLD", 0.0

class FibonacciRetracementStrategy(SyntheticTradingStrategy):
    def __init__(self, symbol: str, timeframe: str = 'M5', period: int = 20):
        super().__init__(symbol, timeframe)
        self.period = period

    def get_signal(self) -> Tuple[str, float]:
        if len(self.data) < self.period:
            return "HOLD", 0.0
        high = self.data['high'].rolling(window=self.period).max().iloc[-1]
        low = self.data['low'].rolling(window=self.period).min().iloc[-1]
        diff = high - low
        if diff == 0:
            return "HOLD", 0.0
        level_618 = high - 0.618 * diff
        current_price = self.data['close'].iloc[-1]
        if current_price > level_618:
            return "SELL", 0.6
        elif current_price < level_618:
            return "BUY", 0.6
        else:
            return "HOLD", 0.0

class AdaptiveStrategy(SyntheticTradingStrategy):
    def __init__(self, symbol: str, timeframe: str = 'M5'):
        super().__init__(symbol, timeframe)

    def detect_market_regime(self) -> str:
        if len(self.data) < 30:
            return "unknown"
        returns = self.data['close'].pct_change().dropna()
        volatility = returns.rolling(window=20).std().iloc[-1]
        if volatility > 0.02:
            return "volatile"
        else:
            return "stable"

    def get_signal(self) -> Tuple[str, float]:
        regime = self.detect_market_regime()
        if regime == "volatile":
            return "SELL", 0.6
        elif regime == "stable":
            return "BUY", 0.6
        else:
            return "HOLD", 0.0

class ElliottWaveStrategy(SyntheticTradingStrategy):
    def __init__(self, symbol: str, timeframe: str = 'M5', wave_length: int = 20):
        super().__init__(symbol, timeframe)
        self.wave_length = wave_length

    def get_signal(self) -> Tuple[str, float]:
        if len(self.data) < self.wave_length:
            return "HOLD", 0.0
        # Simple wave logic: alternate buy/sell every wave_length
        idx = len(self.data) // self.wave_length
        if idx % 2 == 0:
            return "BUY", 0.5
        else:
            return "SELL", 0.5

class HarmonicPatternStrategy(SyntheticTradingStrategy):
    def __init__(self, symbol: str, timeframe: str = 'M5', pattern_length: int = 30):
        super().__init__(symbol, timeframe)
        self.pattern_length = pattern_length

    def get_signal(self) -> Tuple[str, float]:
        if len(self.data) < self.pattern_length:
            return "HOLD", 0.0
        # Placeholder: alternate buy/sell on pattern_length
        idx = len(self.data) // self.pattern_length
        if idx % 2 == 0:
            return "SELL", 0.5
        else:
            return "BUY", 0.5

class OrderFlowStrategy(SyntheticTradingStrategy):
    def __init__(self, symbol: str, timeframe: str = 'M5', period: int = 20):
        super().__init__(symbol, timeframe)
        self.period = period

    def get_signal(self) -> Tuple[str, float]:
        if len(self.data) < self.period:
            return "HOLD", 0.0
        price_change = self.data['close'].iloc[-1] - self.data['open'].iloc[-1]
        volume = self.data['volume'].iloc[-1]
        if price_change > 0 and volume > self.data['volume'].rolling(window=self.period).mean().iloc[-1]:
            return "BUY", 0.6
        elif price_change < 0 and volume > self.data['volume'].rolling(window=self.period).mean().iloc[-1]:
            return "SELL", 0.6
        else:
            return "HOLD", 0.0

class MarketMicrostructureStrategy(SyntheticTradingStrategy):
    def __init__(self, symbol: str, timeframe: str = 'M5', period: int = 20):
        super().__init__(symbol, timeframe)
        self.period = period

    def get_signal(self) -> Tuple[str, float]:
        if len(self.data) < self.period:
            return "HOLD", 0.0
        spread = self.data['high'].iloc[-1] - self.data['low'].iloc[-1]
        avg_spread = (self.data['high'] - self.data['low']).rolling(window=self.period).mean().iloc[-1]
        if spread < avg_spread * 0.5:
            return "BUY", 0.5
        elif spread > avg_spread * 1.5:
            return "SELL", 0.5
        else:
            return "HOLD", 0.0

class SentimentAnalysisStrategy(SyntheticTradingStrategy):
    def __init__(self, symbol: str, timeframe: str = 'M5', period: int = 20):
        super().__init__(symbol, timeframe)
        self.period = period

    def get_signal(self) -> Tuple[str, float]:
        if len(self.data) < self.period:
            return "HOLD", 0.0
        # Placeholder: use price change as sentiment
        price_change = self.data['close'].iloc[-1] - self.data['close'].iloc[-self.period]
        if price_change > 0:
            return "BUY", 0.5
        elif price_change < 0:
            return "SELL", 0.5
        else:
            return "HOLD", 0.0

class MomentumDivergenceStrategy(SyntheticTradingStrategy):
    def __init__(self, symbol: str, timeframe: str = 'M5', period: int = 14):
        super().__init__(symbol, timeframe)
        self.period = period

    def get_signal(self) -> Tuple[str, float]:
        if len(self.data) < self.period + 1:
            return "HOLD", 0.0
        price_change = self.data['close'].iloc[-1] - self.data['close'].iloc[-2]
        momentum = self.data['close'].iloc[-1] - self.data['close'].iloc[-self.period]
        if price_change > 0 and momentum < 0:
            return "SELL", 0.6
        elif price_change < 0 and momentum > 0:
            return "BUY", 0.6
        else:
            return "HOLD", 0.0

class VolatilityRegimeStrategy(SyntheticTradingStrategy):
    def __init__(self, symbol: str, timeframe: str = 'M5', period: int = 20):
        super().__init__(symbol, timeframe)
        self.period = period

    def get_signal(self) -> Tuple[str, float]:
        if len(self.data) < self.period:
            return "HOLD", 0.0
        returns = self.data['close'].pct_change().dropna()
        volatility = returns.rolling(window=self.period).std().iloc[-1]
        if volatility > 0.02:
            return "SELL", 0.6
        elif volatility < 0.01:
            return "BUY", 0.6
        else:
            return "HOLD", 0.0

class PriceActionStrategy(SyntheticTradingStrategy):
    def __init__(self, symbol: str, timeframe: str = 'M5', period: int = 20):
        super().__init__(symbol, timeframe)
        self.period = period

    def get_signal(self) -> Tuple[str, float]:
        if len(self.data) < self.period:
            return "HOLD", 0.0
        bullish = (self.data['close'] > self.data['open']).tail(self.period).sum()
        bearish = (self.data['close'] < self.data['open']).tail(self.period).sum()
        if bullish > bearish:
            return "BUY", 0.6
        elif bearish > bullish:
            return "SELL", 0.6
        else:
            return "HOLD", 0.0

class CorrelationStrategy(SyntheticTradingStrategy):
    def __init__(self, symbol: str, timeframe: str = 'M5', period: int = 20):
        super().__init__(symbol, timeframe)
        self.period = period

    def get_signal(self) -> Tuple[str, float]:
        # Placeholder: always hold (needs multi-symbol data)
        return "HOLD", 0.0

class MachineLearningInspiredStrategy(SyntheticTradingStrategy):
    def __init__(self, symbol: str, timeframe: str = 'M5', period: int = 20):
        super().__init__(symbol, timeframe)
        self.period = period

    def get_signal(self) -> Tuple[str, float]:
        if len(self.data) < self.period:
            return "HOLD", 0.0
        # Placeholder: use moving average as ML-inspired signal
        ma = self.data['close'].rolling(window=self.period).mean().iloc[-1]
        price = self.data['close'].iloc[-1]
        if price > ma:
            return "BUY", 0.5
        elif price < ma:
            return "SELL", 0.5
        else:
            return "HOLD", 0.0 