import pandas as pd
import numpy as np
from typing import Dict, List, Tuple
from .config import DerivConfig

class SyntheticTradingStrategy:
    """Base class for synthetic indices trading strategies"""
    
    def __init__(self, symbol: str, timeframe: str = '5m'):
        self.symbol = symbol
        self.timeframe = timeframe
        self.granularity = DerivConfig.TIMEFRAMES.get(timeframe, 300)
        self.data = pd.DataFrame()
        
    def add_candle(self, candle: Dict):
        """Add a new candle to the data"""
        new_row = pd.DataFrame([{
            'timestamp': candle.get('epoch', 0),
            'open': float(candle.get('open', 0)),
            'high': float(candle.get('high', 0)),
            'low': float(candle.get('low', 0)),
            'close': float(candle.get('close', 0)),
            'volume': float(candle.get('volume', 0))
        }])
        
        self.data = pd.concat([self.data, new_row], ignore_index=True)
        
        # Keep only last 1000 candles to prevent memory issues
        if len(self.data) > 1000:
            self.data = self.data.tail(1000)
    
    def get_signal(self) -> Tuple[str, float]:
        """Get trading signal - to be implemented by subclasses"""
        raise NotImplementedError("Subclasses must implement get_signal()")
    
    def reset(self):
        """Reset strategy data"""
        self.data = pd.DataFrame()

class MovingAverageCrossover(SyntheticTradingStrategy):
    """Moving Average Crossover Strategy for Synthetic Indices"""
    
    def __init__(self, symbol: str, timeframe: str = '5m', short_window: int = 10, long_window: int = 20):
        super().__init__(symbol, timeframe)
        self.short_window = short_window
        self.long_window = long_window
        
    def get_signal(self) -> Tuple[str, float]:
        """Get trading signal based on MA crossover"""
        if len(self.data) < self.long_window:
            return "HOLD", 0.0
        
        # Calculate moving averages
        short_ma = self.data['close'].rolling(window=self.short_window).mean()
        long_ma = self.data['close'].rolling(window=self.long_window).mean()
        
        current_short = short_ma.iloc[-1]
        current_long = long_ma.iloc[-1]
        prev_short = short_ma.iloc[-2] if len(short_ma) > 1 else current_short
        prev_long = long_ma.iloc[-2] if len(long_ma) > 1 else current_long
        
        # Check for crossover
        if current_short > current_long and prev_short <= prev_long:
            return "UP", 0.8  # Strong buy signal
        elif current_short < current_long and prev_short >= prev_long:
            return "DOWN", 0.8  # Strong sell signal
        elif current_short > current_long:
            return "UP", 0.6  # Weak buy signal
        else:
            return "DOWN", 0.6  # Weak sell signal

class RSIStrategy(SyntheticTradingStrategy):
    """RSI Strategy for Synthetic Indices"""
    
    def __init__(self, symbol: str, timeframe: str = '5m', period: int = 14, 
                 oversold: int = 30, overbought: int = 70):
        super().__init__(symbol, timeframe)
        self.period = period
        self.oversold = oversold
        self.overbought = overbought
        
    def calculate_rsi(self) -> float:
        """Calculate RSI value"""
        if len(self.data) < self.period + 1:
            return 50.0
        
        delta = self.data['close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=self.period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=self.period).mean()
        
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        
        return rsi.iloc[-1]
    
    def get_signal(self) -> Tuple[str, float]:
        """Get trading signal based on RSI"""
        if len(self.data) < self.period + 1:
            return "HOLD", 0.0
        
        rsi = self.calculate_rsi()
        
        if rsi < self.oversold:
            return "UP", 0.9  # Strong buy signal (oversold)
        elif rsi > self.overbought:
            return "DOWN", 0.9  # Strong sell signal (overbought)
        elif rsi < 50:
            return "UP", 0.6  # Weak buy signal
        else:
            return "DOWN", 0.6  # Weak sell signal

class BollingerBandsStrategy(SyntheticTradingStrategy):
    """Bollinger Bands Strategy for Synthetic Indices"""
    
    def __init__(self, symbol: str, timeframe: str = '5m', period: int = 20, std_dev: float = 2.0):
        super().__init__(symbol, timeframe)
        self.period = period
        self.std_dev = std_dev
        
    def get_signal(self) -> Tuple[str, float]:
        """Get trading signal based on Bollinger Bands"""
        if len(self.data) < self.period:
            return "HOLD", 0.0
        
        # Calculate Bollinger Bands
        sma = self.data['close'].rolling(window=self.period).mean()
        std = self.data['close'].rolling(window=self.period).std()
        
        upper_band = sma + (std * self.std_dev)
        lower_band = sma - (std * self.std_dev)
        
        current_price = self.data['close'].iloc[-1]
        current_upper = upper_band.iloc[-1]
        current_lower = lower_band.iloc[-1]
        
        # Check for signals
        if current_price <= current_lower:
            return "UP", 0.8  # Price at lower band - buy signal
        elif current_price >= current_upper:
            return "DOWN", 0.8  # Price at upper band - sell signal
        else:
            return "HOLD", 0.0

class VolatilityBreakoutStrategy(SyntheticTradingStrategy):
    """Volatility Breakout Strategy for Synthetic Indices"""
    
    def __init__(self, symbol: str, timeframe: str = '5m', period: int = 20, multiplier: float = 1.5):
        super().__init__(symbol, timeframe)
        self.period = period
        self.multiplier = multiplier
        
    def get_signal(self) -> Tuple[str, float]:
        """Get trading signal based on volatility breakout"""
        if len(self.data) < self.period:
            return "HOLD", 0.0
        
        # Calculate average true range (ATR)
        high_low = self.data['high'] - self.data['low']
        high_close = np.abs(self.data['high'] - self.data['close'].shift())
        low_close = np.abs(self.data['low'] - self.data['close'].shift())
        
        true_range = np.maximum(high_low, np.maximum(high_close, low_close))
        atr = true_range.rolling(window=self.period).mean()
        
        current_atr = atr.iloc[-1]
        avg_atr = atr.mean()
        
        # Check for volatility breakout
        if current_atr > (avg_atr * self.multiplier):
            # High volatility - look for momentum
            price_change = self.data['close'].iloc[-1] - self.data['close'].iloc[-2]
            if price_change > 0:
                return "UP", 0.7  # Upward momentum
            else:
                return "DOWN", 0.7  # Downward momentum
        else:
            return "HOLD", 0.0

class MACDStrategy(SyntheticTradingStrategy):
    """MACD (Moving Average Convergence Divergence) Strategy"""
    
    def __init__(self, symbol: str, timeframe: str = '5m', fast_period: int = 12, 
                 slow_period: int = 26, signal_period: int = 9):
        super().__init__(symbol, timeframe)
        self.fast_period = fast_period
        self.slow_period = slow_period
        self.signal_period = signal_period
        
    def calculate_macd(self) -> Tuple[float, float, float]:
        """Calculate MACD line, signal line, and histogram"""
        if len(self.data) < self.slow_period:
            return 0.0, 0.0, 0.0
        
        # Calculate EMAs
        ema_fast = self.data['close'].ewm(span=self.fast_period).mean()
        ema_slow = self.data['close'].ewm(span=self.slow_period).mean()
        
        # MACD line
        macd_line = ema_fast - ema_slow
        
        # Signal line
        signal_line = macd_line.ewm(span=self.signal_period).mean()
        
        # Histogram
        histogram = macd_line - signal_line
        
        return macd_line.iloc[-1], signal_line.iloc[-1], histogram.iloc[-1]
    
    def get_signal(self) -> Tuple[str, float]:
        """Get trading signal based on MACD"""
        if len(self.data) < self.slow_period:
            return "HOLD", 0.0
        
        macd_line, signal_line, histogram = self.calculate_macd()
        
        # Check for MACD crossover
        if macd_line > signal_line and histogram > 0:
            return "UP", 0.8  # Bullish signal
        elif macd_line < signal_line and histogram < 0:
            return "DOWN", 0.8  # Bearish signal
        else:
            return "HOLD", 0.0

class StochasticStrategy(SyntheticTradingStrategy):
    """Stochastic Oscillator Strategy"""
    
    def __init__(self, symbol: str, timeframe: str = '5m', k_period: int = 14, 
                 d_period: int = 3, oversold: int = 20, overbought: int = 80):
        super().__init__(symbol, timeframe)
        self.k_period = k_period
        self.d_period = d_period
        self.oversold = oversold
        self.overbought = overbought
        
    def calculate_stochastic(self) -> Tuple[float, float]:
        """Calculate %K and %D lines"""
        if len(self.data) < self.k_period:
            return 50.0, 50.0
        
        # Calculate %K
        lowest_low = self.data['low'].rolling(window=self.k_period).min()
        highest_high = self.data['high'].rolling(window=self.k_period).max()
        
        k_percent = 100 * ((self.data['close'] - lowest_low) / (highest_high - lowest_low))
        
        # Calculate %D (SMA of %K)
        d_percent = k_percent.rolling(window=self.d_period).mean()
        
        return k_percent.iloc[-1], d_percent.iloc[-1]
    
    def get_signal(self) -> Tuple[str, float]:
        """Get trading signal based on Stochastic"""
        if len(self.data) < self.k_period:
            return "HOLD", 0.0
        
        k_percent, d_percent = self.calculate_stochastic()
        
        # Check for oversold/overbought conditions
        if k_percent < self.oversold and d_percent < self.oversold:
            return "UP", 0.9  # Strong buy signal
        elif k_percent > self.overbought and d_percent > self.overbought:
            return "DOWN", 0.9  # Strong sell signal
        elif k_percent < 50 and d_percent < 50:
            return "UP", 0.6  # Weak buy signal
        elif k_percent > 50 and d_percent > 50:
            return "DOWN", 0.6  # Weak sell signal
        else:
            return "HOLD", 0.0

class WilliamsRStrategy(SyntheticTradingStrategy):
    """Williams %R Strategy"""
    
    def __init__(self, symbol: str, timeframe: str = '5m', period: int = 14, 
                 oversold: int = -80, overbought: int = -20):
        super().__init__(symbol, timeframe)
        self.period = period
        self.oversold = oversold
        self.overbought = overbought
        
    def calculate_williams_r(self) -> float:
        """Calculate Williams %R"""
        if len(self.data) < self.period:
            return -50.0
        
        highest_high = self.data['high'].rolling(window=self.period).max()
        lowest_low = self.data['low'].rolling(window=self.period).min()
        
        williams_r = -100 * ((highest_high - self.data['close']) / (highest_high - lowest_low))
        
        return williams_r.iloc[-1]
    
    def get_signal(self) -> Tuple[str, float]:
        """Get trading signal based on Williams %R"""
        if len(self.data) < self.period:
            return "HOLD", 0.0
        
        williams_r = self.calculate_williams_r()
        
        if williams_r < self.oversold:
            return "UP", 0.9  # Oversold - buy signal
        elif williams_r > self.overbought:
            return "DOWN", 0.9  # Overbought - sell signal
        else:
            return "HOLD", 0.0

class ParabolicSARStrategy(SyntheticTradingStrategy):
    """Parabolic SAR Strategy"""
    
    def __init__(self, symbol: str, timeframe: str = '5m', acceleration: float = 0.02, 
                 maximum: float = 0.2):
        super().__init__(symbol, timeframe)
        self.acceleration = acceleration
        self.maximum = maximum
        
    def calculate_sar(self) -> float:
        """Calculate Parabolic SAR"""
        if len(self.data) < 2:
            return self.data['close'].iloc[-1] if len(self.data) > 0 else 0.0
        
        # Simplified SAR calculation
        high = self.data['high'].iloc[-1]
        low = self.data['low'].iloc[-1]
        close = self.data['close'].iloc[-1]
        
        # Basic SAR logic
        if close > high * 0.99:  # Price near high
            return low * 0.99  # SAR below low
        elif close < low * 1.01:  # Price near low
            return high * 1.01  # SAR above high
        else:
            return close  # Neutral
        
    def get_signal(self) -> Tuple[str, float]:
        """Get trading signal based on Parabolic SAR"""
        if len(self.data) < 2:
            return "HOLD", 0.0
        
        sar = self.calculate_sar()
        current_price = self.data['close'].iloc[-1]
        
        if current_price > sar:
            return "UP", 0.7  # Bullish
        elif current_price < sar:
            return "DOWN", 0.7  # Bearish
        else:
            return "HOLD", 0.0

class IchimokuStrategy(SyntheticTradingStrategy):
    """Ichimoku Cloud Strategy"""
    
    def __init__(self, symbol: str, timeframe: str = '5m'):
        super().__init__(symbol, timeframe)
        
    def calculate_ichimoku(self) -> Dict[str, float]:
        """Calculate Ichimoku components"""
        if len(self.data) < 52:
            return {}
        
        # Tenkan-sen (Conversion Line): (9-period high + 9-period low)/2
        period9_high = self.data['high'].rolling(window=9).max()
        period9_low = self.data['low'].rolling(window=9).min()
        tenkan_sen = (period9_high + period9_low) / 2
        
        # Kijun-sen (Base Line): (26-period high + 26-period low)/2
        period26_high = self.data['high'].rolling(window=26).max()
        period26_low = self.data['low'].rolling(window=26).min()
        kijun_sen = (period26_high + period26_low) / 2
        
        # Senkou Span A (Leading Span A): (Conversion Line + Base Line)/2
        senkou_span_a = (tenkan_sen + kijun_sen) / 2
        
        # Senkou Span B (Leading Span B): (52-period high + 52-period low)/2
        period52_high = self.data['high'].rolling(window=52).max()
        period52_low = self.data['low'].rolling(window=52).min()
        senkou_span_b = (period52_high + period52_low) / 2
        
        return {
            'tenkan_sen': tenkan_sen.iloc[-1],
            'kijun_sen': kijun_sen.iloc[-1],
            'senkou_span_a': senkou_span_a.iloc[-1],
            'senkou_span_b': senkou_span_b.iloc[-1]
        }
    
    def get_signal(self) -> Tuple[str, float]:
        """Get trading signal based on Ichimoku Cloud"""
        if len(self.data) < 52:
            return "HOLD", 0.0
        
        ichimoku = self.calculate_ichimoku()
        current_price = self.data['close'].iloc[-1]
        
        # Check if price is above cloud (bullish)
        if current_price > max(ichimoku['senkou_span_a'], ichimoku['senkou_span_b']):
            return "UP", 0.8
        # Check if price is below cloud (bearish)
        elif current_price < min(ichimoku['senkou_span_a'], ichimoku['senkou_span_b']):
            return "DOWN", 0.8
        else:
            return "HOLD", 0.0

class MomentumStrategy(SyntheticTradingStrategy):
    """Momentum-based Strategy using Rate of Change"""
    
    def __init__(self, symbol: str, timeframe: str = '5m', period: int = 10, 
                 threshold: float = 0.5):
        super().__init__(symbol, timeframe)
        self.period = period
        self.threshold = threshold
        
    def calculate_momentum(self) -> float:
        """Calculate momentum using Rate of Change"""
        if len(self.data) < self.period:
            return 0.0
        
        current_price = self.data['close'].iloc[-1]
        past_price = self.data['close'].iloc[-self.period]
        
        momentum = ((current_price - past_price) / past_price) * 100
        return momentum
    
    def get_signal(self) -> Tuple[str, float]:
        """Get trading signal based on momentum"""
        if len(self.data) < self.period:
            return "HOLD", 0.0
        
        momentum = self.calculate_momentum()
        
        if momentum > self.threshold:
            return "UP", min(abs(momentum) / 10, 0.9)  # Strong positive momentum
        elif momentum < -self.threshold:
            return "DOWN", min(abs(momentum) / 10, 0.9)  # Strong negative momentum
        else:
            return "HOLD", 0.0

class MultiStrategy(SyntheticTradingStrategy):
    """Combines multiple strategies for better signals"""
    
    def __init__(self, symbol: str, timeframe: str = '5m'):
        super().__init__(symbol, timeframe)
        self.strategies = {
            # Basic strategies (10)
            'ma': MovingAverageCrossover(symbol, timeframe),
            'rsi': RSIStrategy(symbol, timeframe),
            'bb': BollingerBandsStrategy(symbol, timeframe),
            'vb': VolatilityBreakoutStrategy(symbol, timeframe),
            'macd': MACDStrategy(symbol, timeframe),
            'stoch': StochasticStrategy(symbol, timeframe),
            'williams': WilliamsRStrategy(symbol, timeframe),
            'sar': ParabolicSARStrategy(symbol, timeframe),
            'ichimoku': IchimokuStrategy(symbol, timeframe),
            'momentum': MomentumStrategy(symbol, timeframe),
            
            # Advanced strategies (8)
            'mean_reversion': MeanReversionStrategy(symbol, timeframe),
            'trend_following': TrendFollowingStrategy(symbol, timeframe),
            'advanced_volatility': AdvancedVolatilityStrategy(symbol, timeframe),
            'support_resistance': SupportResistanceStrategy(symbol, timeframe),
            'divergence': DivergenceStrategy(symbol, timeframe),
            'volume_price': VolumePriceStrategy(symbol, timeframe),
            'fibonacci': FibonacciRetracementStrategy(symbol, timeframe),
            'adaptive': AdaptiveStrategy(symbol, timeframe),
            
            # Ultra-advanced strategies (10) - NEW ADDITIONS
            'elliott_wave': ElliottWaveStrategy(symbol, timeframe),
            'harmonic_pattern': HarmonicPatternStrategy(symbol, timeframe),
            'order_flow': OrderFlowStrategy(symbol, timeframe),
            'market_microstructure': MarketMicrostructureStrategy(symbol, timeframe),
            'sentiment': SentimentAnalysisStrategy(symbol, timeframe),
            'momentum_divergence': MomentumDivergenceStrategy(symbol, timeframe),
            'volatility_regime': VolatilityRegimeStrategy(symbol, timeframe),
            'price_action': PriceActionStrategy(symbol, timeframe),
            'correlation': CorrelationStrategy(symbol, timeframe),
            'ml_inspired': MachineLearningInspiredStrategy(symbol, timeframe)
        }
        
    def add_candle(self, candle: Dict):
        """Add candle to all strategies"""
        super().add_candle(candle)
        for strategy in self.strategies.values():
            strategy.add_candle(candle)
    
    def get_signal(self) -> Tuple[str, float]:
        """Get combined signal from all strategies with weighted voting"""
        signals = {}
        for name, strategy in self.strategies.items():
            try:
                signal, confidence = strategy.get_signal()
                signals[name] = (signal, confidence)
            except Exception as e:
                print(f"Error in strategy {name}: {e}")
                signals[name] = ("HOLD", 0.0)
        
        # Count signals with confidence weighting
        up_votes = 0.0
        down_votes = 0.0
        hold_votes = 0.0
        
        for name, (signal, confidence) in signals.items():
            if signal == "UP":
                up_votes += confidence
            elif signal == "DOWN":
                down_votes += confidence
            else:
                hold_votes += confidence
        
        # Calculate total votes
        total_votes = up_votes + down_votes + hold_votes
        
        if total_votes == 0:
            return "HOLD", 0.0
        
        # Calculate percentages
        up_percentage = up_votes / total_votes
        down_percentage = down_votes / total_votes
        hold_percentage = hold_votes / total_votes
        
        # Enhanced signal generation with multiple confirmations
        # Require stronger consensus for more reliable signals
        if up_percentage > 0.35 and up_percentage > down_percentage:
            return "UP", min(up_percentage * 1.5, 0.95)
        elif down_percentage > 0.35 and down_percentage > up_percentage:
            return "DOWN", min(down_percentage * 1.5, 0.95)
        else:
            return "HOLD", 0.0

class MeanReversionStrategy(SyntheticTradingStrategy):
    """Mean Reversion Strategy using Bollinger Bands and Z-Score"""
    
    def __init__(self, symbol: str, timeframe: str = '5m', period: int = 20, std_dev: float = 2.0):
        super().__init__(symbol, timeframe)
        self.period = period
        self.std_dev = std_dev
        
    def get_signal(self) -> Tuple[str, float]:
        """Get trading signal based on mean reversion"""
        if len(self.data) < self.period:
            return "HOLD", 0.0
        
        # Calculate Bollinger Bands
        sma = self.data['close'].rolling(window=self.period).mean()
        std = self.data['close'].rolling(window=self.period).std()
        
        upper_band = sma + (std * self.std_dev)
        lower_band = sma - (std * self.std_dev)
        
        current_price = self.data['close'].iloc[-1]
        current_upper = upper_band.iloc[-1]
        current_lower = lower_band.iloc[-1]
        current_sma = sma.iloc[-1]
        
        # Calculate Z-Score
        z_score = (current_price - current_sma) / std.iloc[-1] if std.iloc[-1] > 0 else 0
        
        # Mean reversion signals
        if z_score > 2.0:  # Price too high
            return "DOWN", min(abs(z_score) / 3, 0.9)
        elif z_score < -2.0:  # Price too low
            return "UP", min(abs(z_score) / 3, 0.9)
        else:
            return "HOLD", 0.0

class TrendFollowingStrategy(SyntheticTradingStrategy):
    """Advanced Trend Following with Multiple Timeframes"""
    
    def __init__(self, symbol: str, timeframe: str = '5m', short_period: int = 10, long_period: int = 30):
        super().__init__(symbol, timeframe)
        self.short_period = short_period
        self.long_period = long_period
        
    def get_signal(self) -> Tuple[str, float]:
        """Get trading signal based on trend following"""
        if len(self.data) < self.long_period:
            return "HOLD", 0.0
        
        # Calculate EMAs
        ema_short = self.data['close'].ewm(span=self.short_period).mean()
        ema_long = self.data['close'].ewm(span=self.long_period).mean()
        
        # Calculate trend strength
        current_short = ema_short.iloc[-1]
        current_long = ema_long.iloc[-1]
        prev_short = ema_short.iloc[-2]
        prev_long = ema_long.iloc[-2]
        
        # Trend direction and strength
        trend_direction = 1 if current_short > current_long else -1
        trend_strength = abs(current_short - current_long) / current_long
        
        # Momentum confirmation
        momentum = (current_short - prev_short) / prev_short if prev_short > 0 else 0
        
        if trend_direction > 0 and momentum > 0:
            return "UP", min(trend_strength * 10, 0.9)
        elif trend_direction < 0 and momentum < 0:
            return "DOWN", min(trend_strength * 10, 0.9)
        else:
            return "HOLD", 0.0

class AdvancedVolatilityStrategy(SyntheticTradingStrategy):
    """Advanced Volatility Strategy with Breakout Detection"""
    
    def __init__(self, symbol: str, timeframe: str = '5m', period: int = 20, multiplier: float = 1.5):
        super().__init__(symbol, timeframe)
        self.period = period
        self.multiplier = multiplier
        
    def get_signal(self) -> Tuple[str, float]:
        """Get trading signal based on volatility breakout"""
        if len(self.data) < self.period:
            return "HOLD", 0.0
        
        # Calculate True Range
        high_low = self.data['high'] - self.data['low']
        high_close = np.abs(self.data['high'] - self.data['close'].shift())
        low_close = np.abs(self.data['low'] - self.data['close'].shift())
        
        true_range = np.maximum(high_low, np.maximum(high_close, low_close))
        atr = true_range.rolling(window=self.period).mean()
        
        # Calculate volatility ratio
        current_atr = atr.iloc[-1]
        avg_atr = atr.mean()
        volatility_ratio = current_atr / avg_atr if avg_atr > 0 else 1
        
        # Price momentum
        price_change = (self.data['close'].iloc[-1] - self.data['close'].iloc[-5]) / self.data['close'].iloc[-5]
        
        # Volatility breakout signals
        if volatility_ratio > self.multiplier:
            if price_change > 0.01:  # 1% positive momentum
                return "UP", min(volatility_ratio / 3, 0.9)
            elif price_change < -0.01:  # 1% negative momentum
                return "DOWN", min(volatility_ratio / 3, 0.9)
        
        return "HOLD", 0.0

class SupportResistanceStrategy(SyntheticTradingStrategy):
    """Support and Resistance Level Strategy"""
    
    def __init__(self, symbol: str, timeframe: str = '5m', period: int = 20):
        super().__init__(symbol, timeframe)
        self.period = period
        
    def get_signal(self) -> Tuple[str, float]:
        """Get trading signal based on support/resistance levels"""
        if len(self.data) < self.period * 2:
            return "HOLD", 0.0
        
        current_price = self.data['close'].iloc[-1]
        
        # Calculate support and resistance levels
        recent_highs = self.data['high'].rolling(window=self.period).max()
        recent_lows = self.data['low'].rolling(window=self.period).min()
        
        resistance = recent_highs.iloc[-1]
        support = recent_lows.iloc[-1]
        
        # Calculate distance to levels
        distance_to_resistance = (resistance - current_price) / current_price
        distance_to_support = (current_price - support) / current_price
        
        # Signal generation
        if distance_to_resistance < 0.005:  # Within 0.5% of resistance
            return "DOWN", 0.8
        elif distance_to_support < 0.005:  # Within 0.5% of support
            return "UP", 0.8
        else:
            return "HOLD", 0.0

class DivergenceStrategy(SyntheticTradingStrategy):
    """RSI Divergence Strategy"""
    
    def __init__(self, symbol: str, timeframe: str = '5m', period: int = 14):
        super().__init__(symbol, timeframe)
        self.period = period
        
    def calculate_rsi(self) -> pd.Series:
        """Calculate RSI values"""
        if len(self.data) < self.period + 1:
            return pd.Series([50.0] * len(self.data))
        
        delta = self.data['close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=self.period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=self.period).mean()
        
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        
        return rsi
    
    def get_signal(self) -> Tuple[str, float]:
        """Get trading signal based on RSI divergence"""
        if len(self.data) < self.period * 2:
            return "HOLD", 0.0
        
        rsi = self.calculate_rsi()
        
        # Look for divergence in last 10 periods
        lookback = min(10, len(self.data) // 2)
        
        # Price highs and lows
        price_highs = self.data['high'].rolling(window=3).max()
        price_lows = self.data['low'].rolling(window=3).min()
        
        # RSI highs and lows
        rsi_highs = rsi.rolling(window=3).max()
        rsi_lows = rsi.rolling(window=3).min()
        
        # Check for bearish divergence (price higher, RSI lower)
        if (price_highs.iloc[-1] > price_highs.iloc[-lookback] and 
            rsi_highs.iloc[-1] < rsi_highs.iloc[-lookback]):
            return "DOWN", 0.9
        
        # Check for bullish divergence (price lower, RSI higher)
        elif (price_lows.iloc[-1] < price_lows.iloc[-lookback] and 
              rsi_lows.iloc[-1] > rsi_lows.iloc[-lookback]):
            return "UP", 0.9
        
        return "HOLD", 0.0

class VolumePriceStrategy(SyntheticTradingStrategy):
    """Volume-Price Relationship Strategy"""
    
    def __init__(self, symbol: str, timeframe: str = '5m', period: int = 20):
        super().__init__(symbol, timeframe)
        self.period = period
        
    def get_signal(self) -> Tuple[str, float]:
        """Get trading signal based on volume-price relationship"""
        if len(self.data) < self.period:
            return "HOLD", 0.0
        
        # Calculate volume moving average
        volume_ma = self.data['volume'].rolling(window=self.period).mean()
        current_volume = self.data['volume'].iloc[-1]
        avg_volume = volume_ma.iloc[-1]
        
        # Price change
        price_change = (self.data['close'].iloc[-1] - self.data['close'].iloc[-2]) / self.data['close'].iloc[-2]
        
        # Volume ratio
        volume_ratio = current_volume / avg_volume if avg_volume > 0 else 1
        
        # High volume with price movement
        if volume_ratio > 1.5:  # 50% above average volume
            if price_change > 0.005:  # 0.5% price increase
                return "UP", min(volume_ratio / 3, 0.9)
            elif price_change < -0.005:  # 0.5% price decrease
                return "DOWN", min(volume_ratio / 3, 0.9)
        
        return "HOLD", 0.0

class FibonacciRetracementStrategy(SyntheticTradingStrategy):
    """Fibonacci Retracement Strategy"""
    
    def __init__(self, symbol: str, timeframe: str = '5m', period: int = 20):
        super().__init__(symbol, timeframe)
        self.period = period
        
    def get_signal(self) -> Tuple[str, float]:
        """Get trading signal based on Fibonacci retracement levels"""
        if len(self.data) < self.period:
            return "HOLD", 0.0
        
        # Find recent swing high and low
        recent_high = self.data['high'].rolling(window=self.period).max().iloc[-1]
        recent_low = self.data['low'].rolling(window=self.period).min().iloc[-1]
        
        current_price = self.data['close'].iloc[-1]
        price_range = recent_high - recent_low
        
        if price_range == 0:
            return "HOLD", 0.0
        
        # Calculate Fibonacci levels
        fib_236 = recent_high - (price_range * 0.236)
        fib_382 = recent_high - (price_range * 0.382)
        fib_500 = recent_high - (price_range * 0.500)
        fib_618 = recent_high - (price_range * 0.618)
        
        # Check if price is near Fibonacci levels
        tolerance = price_range * 0.01  # 1% tolerance
        
        if abs(current_price - fib_236) < tolerance:
            return "UP", 0.7
        elif abs(current_price - fib_382) < tolerance:
            return "UP", 0.8
        elif abs(current_price - fib_500) < tolerance:
            return "UP", 0.9
        elif abs(current_price - fib_618) < tolerance:
            return "UP", 0.8
        
        return "HOLD", 0.0

class AdaptiveStrategy(SyntheticTradingStrategy):
    """Adaptive Strategy that changes based on market conditions"""
    
    def __init__(self, symbol: str, timeframe: str = '5m'):
        super().__init__(symbol, timeframe)
        self.market_regime = "trending"  # trending, ranging, volatile
        
    def detect_market_regime(self) -> str:
        """Detect current market regime"""
        if len(self.data) < 50:
            return "trending"
        
        # Calculate volatility
        returns = self.data['close'].pct_change()
        volatility = returns.rolling(window=20).std().iloc[-1]
        
        # Calculate trend strength
        sma_short = self.data['close'].rolling(window=10).mean()
        sma_long = self.data['close'].rolling(window=30).mean()
        trend_strength = abs(sma_short.iloc[-1] - sma_long.iloc[-1]) / sma_long.iloc[-1]
        
        # Determine regime
        if volatility > 0.02:  # High volatility
            return "volatile"
        elif trend_strength > 0.01:  # Strong trend
            return "trending"
        else:
            return "ranging"
    
    def get_signal(self) -> Tuple[str, float]:
        """Get adaptive trading signal"""
        if len(self.data) < 50:
            return "HOLD", 0.0
        
        # Update market regime
        self.market_regime = self.detect_market_regime()
        
        # Different strategies for different regimes
        if self.market_regime == "trending":
            # Use trend following
            ema_short = self.data['close'].ewm(span=10).mean()
            ema_long = self.data['close'].ewm(span=30).mean()
            
            if ema_short.iloc[-1] > ema_long.iloc[-1]:
                return "UP", 0.8
            else:
                return "DOWN", 0.8
                
        elif self.market_regime == "ranging":
            # Use mean reversion
            sma = self.data['close'].rolling(window=20).mean()
            std = self.data['close'].rolling(window=20).std()
            
            current_price = self.data['close'].iloc[-1]
            current_sma = sma.iloc[-1]
            current_std = std.iloc[-1]
            
            z_score = (current_price - current_sma) / current_std if current_std > 0 else 0
            
            if z_score > 1.5:
                return "DOWN", 0.7
            elif z_score < -1.5:
                return "UP", 0.7
                
        elif self.market_regime == "volatile":
            # Use volatility breakout
            atr = self.data['high'].rolling(window=20).max() - self.data['low'].rolling(window=20).min()
            avg_atr = atr.mean()
            current_atr = atr.iloc[-1]
            
            if current_atr > avg_atr * 1.5:
                price_change = (self.data['close'].iloc[-1] - self.data['close'].iloc[-2]) / self.data['close'].iloc[-2]
                if price_change > 0:
                    return "UP", 0.8
                else:
                    return "DOWN", 0.8
        
        return "HOLD", 0.0 

class ElliottWaveStrategy(SyntheticTradingStrategy):
    """Elliott Wave Pattern Recognition Strategy"""
    
    def __init__(self, symbol: str, timeframe: str = '5m', wave_length: int = 20):
        super().__init__(symbol, timeframe)
        self.wave_length = wave_length
        
    def get_signal(self) -> Tuple[str, float]:
        """Get trading signal based on Elliott Wave patterns"""
        if len(self.data) < self.wave_length:
            return "HOLD", 0.0
        
        # Simplified Elliott Wave detection
        highs = self.data['high'].rolling(window=5).max()
        lows = self.data['low'].rolling(window=5).min()
        
        # Look for wave patterns in recent data
        recent_highs = highs.tail(10).values
        recent_lows = lows.tail(10).values
        
        # Detect impulse wave (5-wave pattern)
        if len(recent_highs) >= 5:
            # Check for higher highs in waves 1, 3, 5
            if (recent_highs[-1] > recent_highs[-3] > recent_highs[-5] and
                recent_lows[-2] > recent_lows[-4]):  # Wave 2 and 4 corrections
                return "UP", 0.85  # Impulse wave up
        
        # Detect correction wave (3-wave pattern)
        if len(recent_highs) >= 3:
            if (recent_highs[-1] < recent_highs[-2] and
                recent_lows[-1] > recent_lows[-2]):
                return "DOWN", 0.75  # Correction wave down
        
        return "HOLD", 0.0

class HarmonicPatternStrategy(SyntheticTradingStrategy):
    """Harmonic Pattern Recognition (Gartley, Butterfly, Bat, etc.)"""
    
    def __init__(self, symbol: str, timeframe: str = '5m', pattern_length: int = 30):
        super().__init__(symbol, timeframe)
        self.pattern_length = pattern_length
        
    def get_signal(self) -> Tuple[str, float]:
        """Get trading signal based on harmonic patterns"""
        if len(self.data) < self.pattern_length:
            return "HOLD", 0.0
        
        # Simplified Gartley pattern detection
        highs = self.data['high'].rolling(window=3).max()
        lows = self.data['low'].rolling(window=3).min()
        
        # Look for XABCD pattern
        if len(highs) >= 5:
            X, A, B, C, D = highs.iloc[-5:].values
            
            # Gartley pattern ratios (0.618, 0.382, 0.886)
            AB_ratio = abs(B - A) / abs(X - A) if abs(X - A) > 0 else 0
            BC_ratio = abs(C - B) / abs(A - B) if abs(A - B) > 0 else 0
            CD_ratio = abs(D - C) / abs(B - C) if abs(B - C) > 0 else 0
            
            # Check if ratios match Gartley pattern
            if (0.5 < AB_ratio < 0.7 and  # AB should be ~0.618
                0.3 < BC_ratio < 0.5 and  # BC should be ~0.382
                0.8 < CD_ratio < 1.0):    # CD should be ~0.886
                return "UP", 0.9  # Bullish Gartley
        
        return "HOLD", 0.0

class OrderFlowStrategy(SyntheticTradingStrategy):
    """Order Flow Analysis Strategy"""
    
    def __init__(self, symbol: str, timeframe: str = '5m', period: int = 20):
        super().__init__(symbol, timeframe)
        self.period = period
        
    def get_signal(self) -> Tuple[str, float]:
        """Get trading signal based on order flow analysis"""
        if len(self.data) < self.period:
            return "HOLD", 0.0
        
        # Analyze price-volume relationship for order flow
        volume = self.data['volume']
        price_change = self.data['close'].pct_change()
        
        # Calculate volume-weighted average price (VWAP)
        typical_price = (self.data['high'] + self.data['low'] + self.data['close']) / 3
        vwap = (typical_price * volume).rolling(window=self.period).sum() / volume.rolling(window=self.period).sum()
        
        current_price = self.data['close'].iloc[-1]
        current_vwap = vwap.iloc[-1]
        
        # Volume surge detection
        avg_volume = volume.rolling(window=self.period).mean().iloc[-1]
        current_volume = volume.iloc[-1]
        volume_surge = current_volume / avg_volume if avg_volume > 0 else 1
        
        # Order flow signals
        if current_price > current_vwap and volume_surge > 1.5:
            return "UP", min(volume_surge / 2, 0.9)  # Strong buying pressure
        elif current_price < current_vwap and volume_surge > 1.5:
            return "DOWN", min(volume_surge / 2, 0.9)  # Strong selling pressure
        
        return "HOLD", 0.0

class MarketMicrostructureStrategy(SyntheticTradingStrategy):
    """Market Microstructure Analysis Strategy"""
    
    def __init__(self, symbol: str, timeframe: str = '5m', period: int = 20):
        super().__init__(symbol, timeframe)
        self.period = period
        
    def get_signal(self) -> Tuple[str, float]:
        """Get trading signal based on market microstructure"""
        if len(self.data) < self.period:
            return "HOLD", 0.0
        
        # Calculate bid-ask spread proxy (high-low spread)
        spread = (self.data['high'] - self.data['low']) / self.data['close']
        avg_spread = spread.rolling(window=self.period).mean()
        current_spread = spread.iloc[-1]
        
        # Calculate price efficiency (how quickly price moves)
        price_efficiency = abs(self.data['close'].pct_change()).rolling(window=self.period).mean()
        current_efficiency = price_efficiency.iloc[-1]
        
        # Market microstructure signals
        if current_spread < avg_spread.iloc[-1] * 0.8:  # Tight spread
            if current_efficiency > 0.01:  # High efficiency
                return "UP", 0.8  # Efficient upward movement
        elif current_spread > avg_spread.iloc[-1] * 1.2:  # Wide spread
            if current_efficiency < 0.005:  # Low efficiency
                return "DOWN", 0.7  # Inefficient downward movement
        
        return "HOLD", 0.0

class SentimentAnalysisStrategy(SyntheticTradingStrategy):
    """Market Sentiment Analysis Strategy"""
    
    def __init__(self, symbol: str, timeframe: str = '5m', period: int = 20):
        super().__init__(symbol, timeframe)
        self.period = period
        
    def get_signal(self) -> Tuple[str, float]:
        """Get trading signal based on market sentiment"""
        if len(self.data) < self.period:
            return "HOLD", 0.0
        
        # Calculate sentiment indicators
        price_change = self.data['close'].pct_change()
        
        # Bullish sentiment: consecutive positive closes
        bullish_streak = 0
        for i in range(min(5, len(price_change))):
            if price_change.iloc[-(i+1)] > 0:
                bullish_streak += 1
            else:
                break
        
        # Bearish sentiment: consecutive negative closes
        bearish_streak = 0
        for i in range(min(5, len(price_change))):
            if price_change.iloc[-(i+1)] < 0:
                bearish_streak += 1
            else:
                break
        
        # Volume confirmation
        short_vol_avg = self.data['volume'].rolling(window=5).mean().iloc[-1]
        long_vol_avg = self.data['volume'].rolling(window=self.period).mean().iloc[-1]
        
        # Avoid division by zero
        if long_vol_avg > 0:
            volume_trend = short_vol_avg / long_vol_avg
        else:
            volume_trend = 1.0  # Default to neutral if no volume data
        
        # Sentiment signals
        if bullish_streak >= 3 and volume_trend > 1.2:
            return "UP", min(bullish_streak * 0.2, 0.9)  # Strong bullish sentiment
        elif bearish_streak >= 3 and volume_trend > 1.2:
            return "DOWN", min(bearish_streak * 0.2, 0.9)  # Strong bearish sentiment
        
        return "HOLD", 0.0

class MomentumDivergenceStrategy(SyntheticTradingStrategy):
    """Momentum Divergence Strategy"""
    
    def __init__(self, symbol: str, timeframe: str = '5m', period: int = 14):
        super().__init__(symbol, timeframe)
        self.period = period
        
    def get_signal(self) -> Tuple[str, float]:
        """Get trading signal based on momentum divergence"""
        if len(self.data) < self.period * 2:
            return "HOLD", 0.0
        
        # Calculate momentum indicators
        roc = self.data['close'].pct_change(self.period)  # Rate of Change
        momentum = self.data['close'] - self.data['close'].shift(self.period)
        
        # Look for divergence between price and momentum
        lookback = min(10, len(self.data) // 2)
        
        # Price highs and lows
        price_highs = self.data['high'].rolling(window=3).max()
        price_lows = self.data['low'].rolling(window=3).min()
        
        # Momentum highs and lows
        momentum_highs = momentum.rolling(window=3).max()
        momentum_lows = momentum.rolling(window=3).min()
        
        # Bullish divergence: price lower, momentum higher
        if (price_lows.iloc[-1] < price_lows.iloc[-lookback] and
            momentum_lows.iloc[-1] > momentum_lows.iloc[-lookback]):
            return "UP", 0.9
        
        # Bearish divergence: price higher, momentum lower
        elif (price_highs.iloc[-1] > price_highs.iloc[-lookback] and
              momentum_highs.iloc[-1] < momentum_highs.iloc[-lookback]):
            return "DOWN", 0.9
        
        return "HOLD", 0.0

class VolatilityRegimeStrategy(SyntheticTradingStrategy):
    """Volatility Regime Detection Strategy"""
    
    def __init__(self, symbol: str, timeframe: str = '5m', period: int = 20):
        super().__init__(symbol, timeframe)
        self.period = period
        
    def get_signal(self) -> Tuple[str, float]:
        """Get trading signal based on volatility regime"""
        if len(self.data) < self.period:
            return "HOLD", 0.0
        
        # Calculate volatility measures
        returns = self.data['close'].pct_change()
        volatility = returns.rolling(window=self.period).std()
        
        # Calculate volatility of volatility (vol of vol)
        vol_of_vol = volatility.rolling(window=self.period).std()
        
        current_vol = volatility.iloc[-1]
        current_vol_of_vol = vol_of_vol.iloc[-1]
        avg_vol = volatility.mean()
        
        # Volatility regime detection
        if current_vol > avg_vol * 1.5:  # High volatility regime
            if current_vol_of_vol > vol_of_vol.mean() * 1.2:  # Increasing volatility
                return "DOWN", 0.8  # High volatility often leads to downward pressure
        elif current_vol < avg_vol * 0.7:  # Low volatility regime
            if current_vol_of_vol < vol_of_vol.mean() * 0.8:  # Decreasing volatility
                return "UP", 0.7  # Low volatility often leads to upward movement
        
        return "HOLD", 0.0

class PriceActionStrategy(SyntheticTradingStrategy):
    """Advanced Price Action Strategy"""
    
    def __init__(self, symbol: str, timeframe: str = '5m', period: int = 20):
        super().__init__(symbol, timeframe)
        self.period = period
        
    def get_signal(self) -> Tuple[str, float]:
        """Get trading signal based on price action patterns"""
        if len(self.data) < self.period:
            return "HOLD", 0.0
        
        # Detect candlestick patterns
        open_prices = self.data['open']
        close_prices = self.data['close']
        high_prices = self.data['high']
        low_prices = self.data['low']
        
        # Bullish engulfing pattern
        if (close_prices.iloc[-1] > open_prices.iloc[-1] and  # Current candle is bullish
            open_prices.iloc[-1] < close_prices.iloc[-2] and  # Opens below previous close
            close_prices.iloc[-1] > open_prices.iloc[-2]):    # Closes above previous open
            return "UP", 0.8
        
        # Bearish engulfing pattern
        elif (close_prices.iloc[-1] < open_prices.iloc[-1] and  # Current candle is bearish
              open_prices.iloc[-1] > close_prices.iloc[-2] and  # Opens above previous close
              close_prices.iloc[-1] < open_prices.iloc[-2]):    # Closes below previous open
            return "DOWN", 0.8
        
        # Hammer pattern (bullish reversal)
        elif (close_prices.iloc[-1] > open_prices.iloc[-1] and
              (close_prices.iloc[-1] - low_prices.iloc[-1]) > 2 * (high_prices.iloc[-1] - close_prices.iloc[-1])):
            return "UP", 0.7
        
        # Shooting star pattern (bearish reversal)
        elif (open_prices.iloc[-1] > close_prices.iloc[-1] and
              (high_prices.iloc[-1] - open_prices.iloc[-1]) > 2 * (open_prices.iloc[-1] - low_prices.iloc[-1])):
            return "DOWN", 0.7
        
        return "HOLD", 0.0

class CorrelationStrategy(SyntheticTradingStrategy):
    """Cross-Asset Correlation Strategy"""
    
    def __init__(self, symbol: str, timeframe: str = '5m', period: int = 20):
        super().__init__(symbol, timeframe)
        self.period = period
        
    def get_signal(self) -> Tuple[str, float]:
        """Get trading signal based on correlation analysis"""
        if len(self.data) < self.period:
            return "HOLD", 0.0
        
        # Calculate autocorrelation (price correlation with itself)
        returns = self.data['close'].pct_change()
        
        # Calculate rolling correlation
        if len(returns) >= self.period:
            current_corr = returns.tail(self.period).corr(returns.tail(self.period).shift(1))
            
            # Strong positive autocorrelation suggests trend continuation
            if current_corr > 0.3:
                if returns.iloc[-1] > 0:
                    return "UP", min(abs(current_corr), 0.9)
                else:
                    return "DOWN", min(abs(current_corr), 0.9)
            
            # Strong negative autocorrelation suggests mean reversion
            elif current_corr < -0.3:
                if returns.iloc[-1] > 0:
                    return "DOWN", min(abs(current_corr), 0.9)
                else:
                    return "UP", min(abs(current_corr), 0.9)
        
        return "HOLD", 0.0

class MachineLearningInspiredStrategy(SyntheticTradingStrategy):
    """Machine Learning-Inspired Pattern Recognition"""
    
    def __init__(self, symbol: str, timeframe: str = '5m', period: int = 20):
        super().__init__(symbol, timeframe)
        self.period = period
        
    def get_signal(self) -> Tuple[str, float]:
        """Get trading signal based on ML-inspired pattern recognition"""
        if len(self.data) < self.period:
            return "HOLD", 0.0
        
        # Calculate multiple features (like ML features)
        features = {}
        
        # Price features
        features['price_momentum'] = self.data['close'].pct_change(5).iloc[-1]
        features['price_acceleration'] = self.data['close'].pct_change(5).diff().iloc[-1]
        features['price_volatility'] = self.data['close'].pct_change().rolling(10).std().iloc[-1]
        
        # Volume features
        features['volume_momentum'] = self.data['volume'].pct_change(5).iloc[-1]
        features['volume_volatility'] = self.data['volume'].pct_change().rolling(10).std().iloc[-1]
        
        # Technical features
        features['rsi'] = self._calculate_rsi()
        features['macd'] = self._calculate_macd()
        
        # Simple ML-inspired decision tree
        score = 0.0
        
        # Bullish conditions
        if features['price_momentum'] > 0.01: score += 0.2
        if features['price_acceleration'] > 0: score += 0.2
        if features['volume_momentum'] > 0.1: score += 0.2
        if features['rsi'] < 70: score += 0.2
        if features['macd'] > 0: score += 0.2
        
        # Bearish conditions
        if features['price_momentum'] < -0.01: score -= 0.2
        if features['price_acceleration'] < 0: score -= 0.2
        if features['volume_momentum'] < -0.1: score -= 0.2
        if features['rsi'] > 30: score -= 0.2
        if features['macd'] < 0: score -= 0.2
        
        # Generate signal based on score
        if score > 0.5:
            return "UP", min(score, 0.9)
        elif score < -0.5:
            return "DOWN", min(abs(score), 0.9)
        
        return "HOLD", 0.0
    
    def _calculate_rsi(self) -> float:
        """Calculate RSI"""
        if len(self.data) < 14:
            return 50.0
        
        delta = self.data['close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        
        return rsi.iloc[-1]
    
    def _calculate_macd(self) -> float:
        """Calculate MACD"""
        if len(self.data) < 26:
            return 0.0
        
        ema12 = self.data['close'].ewm(span=12).mean()
        ema26 = self.data['close'].ewm(span=26).mean()
        macd = ema12 - ema26
        
        return macd.iloc[-1] 