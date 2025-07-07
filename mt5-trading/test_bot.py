#!/usr/bin/env python3
"""
Test script for MT5 Trading Bot
This script tests the configuration and connection without placing any trades.
"""

import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import MetaTrader5 as mt5
import pandas as pd
from datetime import datetime
from dotenv import load_dotenv

# Import configurations
from demo_trading.config import MT5Config
from real_trading.real_config import MT5RealConfig

def test_demo_config():
    """Test demo configuration"""
    print("🧪 Testing Demo Configuration...")
    print("=" * 50)
    
    try:
        # Test config loading
        print(f"✅ MT5_LOGIN: {MT5Config.MT5_LOGIN}")
        print(f"✅ MT5_SERVER: {MT5Config.MT5_SERVER}")
        print(f"✅ SYMBOLS: {MT5Config.DEFAULT_SYMBOLS}")
        print(f"✅ STRATEGY: {MT5Config.DEFAULT_STRATEGY}")
        print(f"✅ VOLUME: {MT5Config.VOLUME}")
        print(f"✅ TIMEFRAME: {MT5Config.TIMEFRAME}")
        print(f"✅ CONFIDENCE_THRESHOLD: {MT5Config.CONFIDENCE_THRESHOLD}")
        print(f"✅ MAX_TOTAL_POSITIONS: {MT5Config.MAX_TOTAL_POSITIONS}")
        print(f"✅ MAX_DAILY_LOSS: {MT5Config.MAX_DAILY_LOSS}%")
        print(f"✅ MAX_DAILY_PROFIT: {MT5Config.MAX_DAILY_PROFIT}%")
        
        # Test strategy weights
        print(f"✅ Strategy Weights: {len(MT5Config.STRATEGY_WEIGHTS)} strategies configured")
        
        # Test volume mapping
        test_symbol = "Volatility 100 Index"
        min_volume = MT5Config.get_min_volume(test_symbol)
        print(f"✅ Min volume for {test_symbol}: {min_volume}")
        
        print("✅ Demo configuration test passed!")
        return True
        
    except Exception as e:
        print(f"❌ Demo configuration test failed: {e}")
        return False

def test_real_config():
    """Test real configuration"""
    print("\n🧪 Testing Real Configuration...")
    print("=" * 50)
    
    try:
        # Test config loading
        print(f"✅ MT5_REAL_LOGIN: {MT5RealConfig.MT5_LOGIN}")
        print(f"✅ MT5_REAL_SERVER: {MT5RealConfig.MT5_SERVER}")
        print(f"✅ SYMBOLS: {MT5RealConfig.DEFAULT_SYMBOLS}")
        print(f"✅ STRATEGY: {MT5RealConfig.DEFAULT_STRATEGY}")
        print(f"✅ VOLUME: {MT5RealConfig.VOLUME}")
        print(f"✅ TIMEFRAME: {MT5RealConfig.TIMEFRAME}")
        print(f"✅ CONFIDENCE_THRESHOLD: {MT5RealConfig.CONFIDENCE_THRESHOLD}")
        print(f"✅ MAX_TOTAL_POSITIONS: {MT5RealConfig.MAX_TOTAL_POSITIONS}")
        print(f"✅ MAX_DAILY_LOSS: {MT5RealConfig.MAX_DAILY_LOSS}%")
        print(f"✅ MAX_DAILY_PROFIT: {MT5RealConfig.MAX_DAILY_PROFIT}%")
        
        # Test strategy weights
        print(f"✅ Strategy Weights: {len(MT5RealConfig.STRATEGY_WEIGHTS)} strategies configured")
        
        # Test volume mapping
        test_symbol = "Volatility 100 Index"
        min_volume = MT5RealConfig.get_min_volume(test_symbol)
        print(f"✅ Min volume for {test_symbol}: {min_volume}")
        
        print("✅ Real configuration test passed!")
        return True
        
    except Exception as e:
        print(f"❌ Real configuration test failed: {e}")
        return False

def test_mt5_connection(account_type="demo"):
    """Test MT5 connection"""
    print(f"\n🔌 Testing MT5 {account_type.upper()} Connection...")
    print("=" * 50)
    
    try:
        # Initialize MT5
        if not mt5.initialize():
            print(f"❌ MT5 initialization failed: {mt5.last_error()}")
            return False
        
        # Get config based on account type
        if account_type == "demo":
            config = MT5Config
        else:
            config = MT5RealConfig
        
        # Login to MT5
        if not mt5.login(login=config.MT5_LOGIN, password=config.MT5_PASSWORD, server=config.MT5_SERVER):
            print(f"❌ MT5 login failed: {mt5.last_error()}")
            return False
        
        print(f"✅ Logged in to {account_type.upper()} account: {config.MT5_LOGIN}")
        
        # Get account info
        account_info = mt5.account_info()
        if account_info:
            print(f"✅ Balance: ${account_info.balance:.2f}")
            print(f"✅ Equity: ${account_info.equity:.2f}")
            print(f"✅ Account: {account_info.login}")
            print(f"✅ Broker: {account_info.company}")
        
        # Test symbol info
        test_symbol = "Volatility 100 Index"
        symbol_info = mt5.symbol_info(test_symbol)
        if symbol_info:
            print(f"✅ Symbol {test_symbol} is available")
            print(f"✅ Bid: {symbol_info.bid}")
            print(f"✅ Ask: {symbol_info.ask}")
            print(f"✅ Spread: {symbol_info.spread}")
        else:
            print(f"⚠️  Symbol {test_symbol} not found")
        
        # Test getting historical data
        rates = mt5.copy_rates_from_pos(test_symbol, mt5.TIMEFRAME_M5, 0, 10)
        if rates is not None and len(rates) > 0:
            print(f"✅ Historical data available: {len(rates)} candles")
            latest_rate = rates[-1]
            print(f"✅ Latest price: {latest_rate['close']:.5f}")
        else:
            print(f"⚠️  No historical data for {test_symbol}")
        
        # Disconnect
        mt5.shutdown()
        print(f"✅ {account_type.upper()} connection test passed!")
        return True
        
    except Exception as e:
        print(f"❌ {account_type.upper()} connection test failed: {e}")
        return False

def test_strategies():
    """Test strategy loading"""
    print("\n📊 Testing Strategy Loading...")
    print("=" * 50)
    
    try:
        # Import strategies
        from strategies_mt5 import (
            MovingAverageCrossover, RSIStrategy, BollingerBandsStrategy,
            VolatilityBreakoutStrategy, MACDStrategy, MultiStrategy
        )
        
        # Test strategy initialization
        test_symbol = "Volatility 100 Index"
        strategies = [
            MovingAverageCrossover(test_symbol, 'M5'),
            RSIStrategy(test_symbol, 'M5'),
            BollingerBandsStrategy(test_symbol, 'M5'),
            VolatilityBreakoutStrategy(test_symbol, 'M5'),
            MACDStrategy(test_symbol, 'M5'),
            MultiStrategy(test_symbol, 'M5')
        ]
        
        print(f"✅ Loaded {len(strategies)} strategies")
        
        # Test with sample data
        sample_data = pd.DataFrame({
            'open': [100.0, 101.0, 102.0, 103.0, 104.0],
            'high': [101.0, 102.0, 103.0, 104.0, 105.0],
            'low': [99.0, 100.0, 101.0, 102.0, 103.0],
            'close': [101.0, 102.0, 103.0, 104.0, 105.0],
            'volume': [1000, 1100, 1200, 1300, 1400]
        })
        
        for i, strategy in enumerate(strategies):
            try:
                strategy.update_data(sample_data)
                signal, confidence = strategy.get_signal()
                print(f"✅ Strategy {i+1} ({strategy.__class__.__name__}): {signal} (confidence: {confidence:.2f})")
            except Exception as e:
                print(f"⚠️  Strategy {i+1} ({strategy.__class__.__name__}): Error - {e}")
        
        print("✅ Strategy loading test passed!")
        return True
        
    except Exception as e:
        print(f"❌ Strategy loading test failed: {e}")
        return False

def main():
    """Main test function"""
    print("🚀 MT5 Trading Bot Test Suite")
    print("=" * 60)
    
    # Test configurations
    demo_config_ok = test_demo_config()
    real_config_ok = test_real_config()
    
    # Test strategies
    strategies_ok = test_strategies()
    
    # Test connections (only if configs are ok)
    demo_connection_ok = False
    real_connection_ok = False
    
    if demo_config_ok:
        demo_connection_ok = test_mt5_connection("demo")
    
    if real_config_ok:
        real_connection_ok = test_mt5_connection("real")
    
    # Summary
    print("\n📋 TEST SUMMARY")
    print("=" * 60)
    print(f"Demo Configuration: {'✅ PASS' if demo_config_ok else '❌ FAIL'}")
    print(f"Real Configuration: {'✅ PASS' if real_config_ok else '❌ FAIL'}")
    print(f"Strategy Loading: {'✅ PASS' if strategies_ok else '❌ FAIL'}")
    print(f"Demo Connection: {'✅ PASS' if demo_connection_ok else '❌ FAIL'}")
    print(f"Real Connection: {'✅ PASS' if real_connection_ok else '❌ FAIL'}")
    
    # Overall result
    all_tests_passed = demo_config_ok and real_config_ok and strategies_ok and (demo_connection_ok or real_connection_ok)
    
    if all_tests_passed:
        print("\n🎉 ALL TESTS PASSED! The bot is ready to run.")
        print("💡 You can now run the demo bot with: python demo-trading/main.py")
        print("💡 Or run the real bot with: python real-trading/main-real.py")
    else:
        print("\n⚠️  SOME TESTS FAILED! Please check the configuration and connection.")
        print("💡 Make sure MetaTrader 5 is running and credentials are correct.")

if __name__ == "__main__":
    main() 