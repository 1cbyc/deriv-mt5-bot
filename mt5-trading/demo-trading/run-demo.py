#!/usr/bin/env python3
"""
Demo Trading Bot Launcher
Safe for testing - uses demo account only
"""

import sys
import os

# Add parent directory to path to import strategies
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from config import MT5Config
from strategies_mt5 import (
    SyntheticTradingStrategy, MovingAverageCrossover, RSIStrategy, 
    BollingerBandsStrategy, VolatilityBreakoutStrategy, MACDStrategy,
    StochasticStrategy, WilliamsRStrategy, ParabolicSARStrategy,
    IchimokuStrategy, MomentumStrategy, MeanReversionStrategy,
    TrendFollowingStrategy, AdvancedVolatilityStrategy, SupportResistanceStrategy,
    DivergenceStrategy, VolumePriceStrategy, FibonacciRetracementStrategy,
    AdaptiveStrategy, ElliottWaveStrategy, HarmonicPatternStrategy,
    OrderFlowStrategy, MarketMicrostructureStrategy, SentimentAnalysisStrategy,
    MomentumDivergenceStrategy, VolatilityRegimeStrategy, PriceActionStrategy,
    CorrelationStrategy, MachineLearningInspiredStrategy, MultiStrategy
)

class MT5DemoTradingBot:
    """MT5 Demo Trading Bot - SAFE FOR TESTING"""
    
    def __init__(self, symbols=None, strategy_type='multi'):
        # Import the main bot class from the moved file
        from main import MT5TradingBot
        self.bot = MT5TradingBot(symbols, strategy_type)
    
    def start(self):
        """Start the demo bot"""
        print("🚀 Starting MT5 Demo Trading Bot")
        print("✅ SAFE FOR TESTING - Demo account only")
        print("=" * 50)
        self.bot.start()

def main():
    """Main entry point for demo trading"""
    print("🎯 MT5 Demo Trading Bot Launcher")
    print("📁 Demo folder: Safe for testing")
    print("=" * 50)
    
    # Create and start the demo bot
    bot = MT5DemoTradingBot()
    bot.start()

if __name__ == "__main__":
    main() 