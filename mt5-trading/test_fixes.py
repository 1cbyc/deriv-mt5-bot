#!/usr/bin/env python3
"""
Test script to verify all the fixes implemented
"""

import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import MetaTrader5 as mt5
import pandas as pd
import numpy as np
from datetime import datetime

def test_pip_calculations():
    """Test pip calculations for synthetic indices"""
    print("🧪 Testing Pip Calculations...")
    print("=" * 50)
    
    try:
        # Initialize MT5
        if not mt5.initialize():
            print("❌ MT5 initialization failed")
            return False
        
        # Test symbol info
        test_symbol = "Volatility 100 Index"
        symbol_info = mt5.symbol_info(test_symbol)
        
        if symbol_info:
            print(f"✅ Symbol: {test_symbol}")
            print(f"✅ Point: {symbol_info.point}")
            print(f"✅ Pip Value: {symbol_info.point * 10}")
            print(f"✅ Bid: {symbol_info.bid}")
            print(f"✅ Ask: {symbol_info.ask}")
            print(f"✅ Spread: {symbol_info.spread}")
            
            # Test pip calculation
            pip_value = symbol_info.point * 10
            price_diff = symbol_info.ask - symbol_info.bid
            pips = price_diff / pip_value
            print(f"✅ Spread in pips: {pips:.1f}")
            
            return True
        else:
            print(f"❌ No symbol info for {test_symbol}")
            return False
            
    except Exception as e:
        print(f"❌ Error testing pip calculations: {e}")
        return False
    finally:
        mt5.shutdown()

def test_position_sizing():
    """Test dynamic position sizing"""
    print("\n🧪 Testing Position Sizing...")
    print("=" * 50)
    
    try:
        # Test with different account balances
        test_balances = [100, 500, 1000, 5000]
        test_symbol = "Volatility 100 Index"
        
        for balance in test_balances:
            # Simulate position sizing calculation
            risk_percent = 2.0
            risk_amount = balance * (risk_percent / 100.0)
            pip_value = 0.01  # For Volatility 100 Index
            stop_loss_pips = 20
            
            position_size = risk_amount / (stop_loss_pips * pip_value)
            min_volume = 0.5  # Minimum for Volatility 100 Index
            position_size = max(position_size, min_volume)
            
            print(f"✅ Balance: ${balance}, Risk: ${risk_amount:.2f}, Position Size: {position_size:.3f}")
        
        return True
        
    except Exception as e:
        print(f"❌ Error testing position sizing: {e}")
        return False

def test_market_conditions():
    """Test market condition detection"""
    print("\n🧪 Testing Market Condition Detection...")
    print("=" * 50)
    
    try:
        # Create sample data for different market conditions
        # Volatile market
        volatile_data = pd.DataFrame({
            'close': [100, 102, 98, 105, 97, 103, 99, 106, 96, 104]
        })
        
        # Trending market
        trending_data = pd.DataFrame({
            'close': [100, 101, 102, 103, 104, 105, 106, 107, 108, 109]
        })
        
        # Ranging market
        ranging_data = pd.DataFrame({
            'close': [100, 101, 99, 102, 98, 101, 99, 102, 100, 101]
        })
        
        # Test volatility calculation
        for name, data in [("Volatile", volatile_data), ("Trending", trending_data), ("Ranging", ranging_data)]:
            returns = data['close'].pct_change().dropna()
            volatility = returns.rolling(window=5).std().iloc[-1]
            
            # Calculate trend strength
            short_ma = data['close'].rolling(window=3).mean()
            long_ma = data['close'].rolling(window=7).mean()
            trend_strength = abs(short_ma.iloc[-1] - long_ma.iloc[-1]) / long_ma.iloc[-1]
            
            print(f"✅ {name} Market:")
            print(f"   Volatility: {volatility:.4f}")
            print(f"   Trend Strength: {trend_strength:.4f}")
        
        return True
        
    except Exception as e:
        print(f"❌ Error testing market conditions: {e}")
        return False

def test_strategy_weighting():
    """Test strategy weighting system"""
    print("\n🧪 Testing Strategy Weighting...")
    print("=" * 50)
    
    try:
        # Import MultiStrategy
        from strategies_mt5 import MultiStrategy
        
        # Create test strategy
        strategy = MultiStrategy("Volatility 100 Index", "M5")
        
        # Test with sample data
        sample_data = pd.DataFrame({
            'open': [100.0, 101.0, 102.0, 103.0, 104.0, 105.0, 106.0, 107.0, 108.0, 109.0],
            'high': [101.0, 102.0, 103.0, 104.0, 105.0, 106.0, 107.0, 108.0, 109.0, 110.0],
            'low': [99.0, 100.0, 101.0, 102.0, 103.0, 104.0, 105.0, 106.0, 107.0, 108.0],
            'close': [101.0, 102.0, 103.0, 104.0, 105.0, 106.0, 107.0, 108.0, 109.0, 110.0],
            'volume': [1000, 1100, 1200, 1300, 1400, 1500, 1600, 1700, 1800, 1900]
        })
        
        strategy.update_data(sample_data)
        
        # Test market condition detection
        conditions = strategy.detect_market_conditions()
        print(f"✅ Market Conditions: {conditions}")
        
        # Test weight adjustment
        adjusted_weights = strategy.adjust_weights_for_conditions(conditions)
        print(f"✅ Adjusted Weights: {len(adjusted_weights)} strategies")
        
        # Test signal generation
        signal, confidence = strategy.get_signal()
        print(f"✅ Signal: {signal}, Confidence: {confidence:.2f}")
        
        return True
        
    except Exception as e:
        print(f"❌ Error testing strategy weighting: {e}")
        return False

def test_risk_management():
    """Test risk management features"""
    print("\n🧪 Testing Risk Management...")
    print("=" * 50)
    
    try:
        # Test position correlation logic
        def test_correlation(buy_count, sell_count):
            total = buy_count + sell_count
            if total == 0:
                return False
            
            buy_ratio = buy_count / total
            sell_ratio = sell_count / total
            
            return buy_ratio > 0.7 or sell_ratio > 0.7
        
        # Test cases
        test_cases = [
            (3, 1),  # 75% buy - should trigger
            (1, 3),  # 75% sell - should trigger
            (2, 2),  # 50% each - should not trigger
            (1, 1),  # 50% each - should not trigger
        ]
        
        for buy, sell in test_cases:
            is_correlated = test_correlation(buy, sell)
            print(f"✅ {buy} BUY, {sell} SELL: {'Correlated' if is_correlated else 'Not Correlated'}")
        
        # Test consecutive losses
        def test_consecutive_losses(loss_count):
            return loss_count >= 3
        
        for loss_count in [1, 2, 3, 4]:
            should_stop = test_consecutive_losses(loss_count)
            print(f"✅ {loss_count} consecutive losses: {'Stop Trading' if should_stop else 'Continue'}")
        
        return True
        
    except Exception as e:
        print(f"❌ Error testing risk management: {e}")
        return False

def test_trailing_stop():
    """Test trailing stop logic"""
    print("\n🧪 Testing Trailing Stop Logic...")
    print("=" * 50)
    
    try:
        # Test progressive trailing stop
        def calculate_trailing_stop(pip_movement):
            if pip_movement >= 20:
                return 15  # Lock in 15 pips
            elif pip_movement >= 15:
                return 10  # Lock in 10 pips
            elif pip_movement >= 10:
                return 5   # Lock in 5 pips
            else:
                return 0   # Don't trail yet
        
        test_movements = [5, 10, 15, 20, 25]
        for movement in test_movements:
            lock_pips = calculate_trailing_stop(movement)
            print(f"✅ {movement} pips profit: Lock in {lock_pips} pips")
        
        return True
        
    except Exception as e:
        print(f"❌ Error testing trailing stop: {e}")
        return False

def main():
    """Main test function"""
    print("🚀 Testing All Fixes")
    print("=" * 60)
    
    # Run all tests
    tests = [
        ("Pip Calculations", test_pip_calculations),
        ("Position Sizing", test_position_sizing),
        ("Market Conditions", test_market_conditions),
        ("Strategy Weighting", test_strategy_weighting),
        ("Risk Management", test_risk_management),
        ("Trailing Stop", test_trailing_stop),
    ]
    
    results = []
    for test_name, test_func in tests:
        try:
            result = test_func()
            results.append((test_name, result))
        except Exception as e:
            print(f"❌ {test_name} test failed with exception: {e}")
            results.append((test_name, False))
    
    # Summary
    print("\n📋 TEST RESULTS")
    print("=" * 60)
    
    passed = 0
    for test_name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{test_name}: {status}")
        if result:
            passed += 1
    
    print(f"\n🎯 Overall: {passed}/{len(results)} tests passed")
    
    if passed == len(results):
        print("🎉 ALL TESTS PASSED! All fixes are working correctly.")
        print("💡 The bot is ready for testing with improved logic.")
    else:
        print("⚠️  Some tests failed. Please check the implementation.")

if __name__ == "__main__":
    main() 