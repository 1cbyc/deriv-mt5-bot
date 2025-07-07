import MetaTrader5 as mt5
import pandas as pd
import numpy as np
import time
import signal
import sys
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
import os
from dotenv import load_dotenv
import threading

# Fix Unicode encoding for Windows console
if sys.platform == "win32":
    import codecs
    sys.stdout = codecs.getwriter("utf-8")(sys.stdout.detach())
    sys.stderr = codecs.getwriter("utf-8")(sys.stderr.detach())

# Import our custom modules
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

class MT5TradingBot:
    """MT5 Trading Bot for Deriv Synthetic Indices with Advanced Strategies"""
    
    def __init__(self, symbols: Optional[List[str]] = None, strategy_type: str = 'multi'):
        # Use configuration for symbols and strategy
        if symbols is None:
            self.symbols = MT5Config.get_symbols_from_env()
        else:
            self.symbols = symbols
            
        self.strategy_type = strategy_type
        self.connected = False
        self.running = False
        self.active_positions = {}
        self.symbol_performance = {}
        self.strategies = {}
        
        # Initialize strategies for each symbol
        self._initialize_strategies()
        
        # Initialize performance tracking
        for symbol in self.symbols:
            self.symbol_performance[symbol] = {
                'trades': 0,
                'wins': 0,
                'losses': 0,
                'total_pnl': 0.0,
                'last_signal': None,
                'last_signal_time': None,
                'last_confidence': 0.0
            }
        
        # Set up signal handlers for graceful shutdown
        if threading.current_thread() is threading.main_thread():
            signal.signal(signal.SIGINT, self._signal_handler)
            signal.signal(signal.SIGTERM, self._signal_handler)
    
    def _initialize_strategies(self):
        """Initialize strategies for each symbol"""
        strategy_class_name = MT5Config.get_strategy_class(self.strategy_type)
        
        for symbol in self.symbols:
            try:
                # Get the strategy class dynamically
                strategy_class = globals()[strategy_class_name]
                self.strategies[symbol] = strategy_class(symbol, MT5Config.TIMEFRAME)
                print(f"[OK] Initialized {strategy_class_name} for {symbol}")
            except Exception as e:
                print(f"[ERROR] Failed to initialize strategy for {symbol}: {e}")
                # Fallback to MultiStrategy
                self.strategies[symbol] = MultiStrategy(symbol, MT5Config.TIMEFRAME)
    
    def _signal_handler(self, signum, frame):
        """Handle shutdown signals"""
        print(f"\n[STOP] Received signal {signum}, shutting down gracefully...")
        self.stop()
        sys.exit(0)
    
    def connect(self):
        """Connect to Deriv MT5"""
        try:
            # Initialize MT5
            if not mt5.initialize():
                print(f"[ERROR] MT5 initialization failed: {mt5.last_error()}")
                return False
            
            # Login to Deriv MT5 demo account using config
            if not mt5.login(login=MT5Config.MT5_LOGIN, password=MT5Config.MT5_PASSWORD, server=MT5Config.MT5_SERVER):
                print(f"[ERROR] MT5 login failed: {mt5.last_error()}")
                return False
            
            print(f"[OK] Logged in to Deriv MT5 account: {MT5Config.MT5_LOGIN}")
            
            # Get account info
            account_info = mt5.account_info()
            if account_info:
                print(f"[BALANCE] Balance: ${account_info.balance:.2f}")
                print(f"[EQUITY] Equity: ${account_info.equity:.2f}")
                print(f"[ACCOUNT] Account: {account_info.login}")
                print(f"[BROKER] Broker: {account_info.company}")
            
            self.connected = True
            print("[OK] Connected to Deriv MT5 successfully!")
            return True
            
        except Exception as e:
            print(f"[ERROR] Connection failed: {e}")
            return False
    
    def disconnect(self):
        """Disconnect from MT5"""
        if self.connected:
            mt5.shutdown()
            self.connected = False
            print("[DISCONNECT] Disconnected from MT5")
    
    def get_historical_data(self, symbol: str, timeframe: str = 'M5', count: int = 100) -> pd.DataFrame:
        """Get historical data for a symbol"""
        try:
            # Convert timeframe string to MT5 timeframe
            tf_map = {
                'M1': mt5.TIMEFRAME_M1,
                'M5': mt5.TIMEFRAME_M5,
                'M15': mt5.TIMEFRAME_M15,
                'M30': mt5.TIMEFRAME_M30,
                'H1': mt5.TIMEFRAME_H1,
                'H4': mt5.TIMEFRAME_H4,
                'D1': mt5.TIMEFRAME_D1
            }
            
            mt5_timeframe = tf_map.get(timeframe, mt5.TIMEFRAME_M5)
            
            # Get rates
            rates = mt5.copy_rates_from_pos(symbol, mt5_timeframe, 0, count)
            
            if rates is None or len(rates) == 0:
                print(f"[WARNING] No data received for {symbol}")
                return pd.DataFrame()
            
            # Convert to DataFrame
            df = pd.DataFrame(rates)
            df['time'] = pd.to_datetime(df['time'], unit='s')
            df.set_index('time', inplace=True)
            
            return df
            
        except Exception as e:
            print(f"[ERROR] Error getting data for {symbol}: {e}")
            return pd.DataFrame()
    
    def calculate_signals(self, symbol: str, df: pd.DataFrame) -> Tuple[str, float]:
        """Calculate trading signals using the selected strategy"""
        if len(df) < 50:
            return "HOLD", 0.0
        
        try:
            # Update strategy data
            strategy = self.strategies.get(symbol)
            if strategy is None:
                print(f"[ERROR] No strategy found for {symbol}")
                return "HOLD", 0.0
            
            strategy.update_data(df)
            
            # Get signal from strategy
            signal, confidence = strategy.get_signal()
            
            # Apply strategy weight
            strategy_weight = MT5Config.get_strategy_weight(strategy.__class__.__name__)
            adjusted_confidence = confidence * strategy_weight
            
            # Debug output
            if signal != "HOLD":
                print(f"[SIGNAL] {symbol} - {strategy.__class__.__name__}: {signal} (confidence: {adjusted_confidence:.2f})")
            
            return signal, adjusted_confidence
            
        except Exception as e:
            print(f"[ERROR] Error calculating signals for {symbol}: {e}")
            return "HOLD", 0.0
    
    def place_order(self, symbol: str, order_type: str, volume: float, 
                   price: float = 0.0, sl: float = 0.0, tp: float = 0.0, max_retries: int = 3) -> bool:
        """Place an order on MT5 with retry mechanism for volume issues"""
        current_volume = volume
        
        for attempt in range(max_retries):
            try:
                # Prepare the request
                request = {
                    "action": mt5.TRADE_ACTION_DEAL,
                    "symbol": symbol,
                    "volume": current_volume,
                    "type": mt5.ORDER_TYPE_BUY if order_type == "BUY" else mt5.ORDER_TYPE_SELL,
                    "price": price,
                    "deviation": 20,
                    "magic": 234000,
                    # Use a simple, safe comment string for Deriv MT5
                    "comment": "pythonMT5bot",
                    "type_time": mt5.ORDER_TIME_GTC,
                    "type_filling": mt5.ORDER_FILLING_FOK,  # Use Fill or Kill instead of IOC
                }
                
                # Add stop loss and take profit if provided
                if sl > 0:
                    request["sl"] = sl
                if tp > 0:
                    request["tp"] = tp
                
                # Debug: Print the order request
                print(f"🔍 Order request for {symbol} (attempt {attempt + 1}): {request}")
                
                # Send the order
                result = mt5.order_send(request)
                
                # Debug: Print the result
                print(f"🔍 Order result for {symbol} (attempt {attempt + 1}): {result}")
                
                # Check if order was successful
                if result.retcode != mt5.TRADE_RETCODE_DONE:
                    print(f"[ERROR] Order failed for {symbol} (attempt {attempt + 1}): order_send returned None")
                    continue
                
                # Check for volume-related errors
                if "invalid volume" in result.comment.lower() or "volume" in result.comment.lower():
                    print(f"[ERROR] Order failed for {symbol} (attempt {attempt + 1}): {result.comment}")
                    # Try with smaller volume
                    current_volume = max(0.01, current_volume * 0.5)
                    print(f"[RETRY] Trying with volume: {current_volume}")
                    continue
                else:
                    print(f"[ERROR] Non-volume error, not retrying")
                    break
            
            except Exception as e:
                print(f"[ERROR] Error placing order for {symbol} (attempt {attempt + 1}): {e}")
                return False
        
        # If we get here, all attempts failed
        if attempt == max_retries:
            print(f"[ERROR] All attempts failed for {symbol}")
            return False
        
        # Success
        print(f"[SUCCESS] Order placed: {symbol} {order_type} {current_volume} lots at {result.price}")
        
        # Store position info
        self.active_positions[result.order] = {
            'symbol': symbol,
            'type': order_type,
            'volume': current_volume,
            'price': result.price,
            'time': datetime.now()
        }
        
        return True
    
    def close_position(self, ticket: int) -> bool:
        """Close a position by ticket"""
        try:
            # Get position info
            position = mt5.positions_get(ticket=ticket)
            if not position:
                print(f"[ERROR] Position {ticket} not found")
                return False
            
            position = position[0]
            
            # Prepare close request
            request = {
                "action": mt5.TRADE_ACTION_DEAL,
                "symbol": position.symbol,
                "volume": position.volume,
                "type": mt5.ORDER_TYPE_SELL if position.type == mt5.POSITION_TYPE_BUY else mt5.ORDER_TYPE_BUY,
                "position": ticket,
                "price": mt5.symbol_info_tick(position.symbol).ask if position.type == mt5.POSITION_TYPE_BUY else mt5.symbol_info_tick(position.symbol).bid,
                "deviation": 20,
                "magic": 234000,
                "comment": "python script close",
                "type_time": mt5.ORDER_TIME_GTC,
                "type_filling": mt5.ORDER_FILLING_IOC,
            }
            
            # Send close request
            result = mt5.order_send(request)
            if result.retcode != mt5.TRADE_RETCODE_DONE:
                print(f"[ERROR] Failed to close position {ticket}: {result.comment}")
                return False
            
            print(f"[SUCCESS] Position closed: {ticket} - Profit: ${result.profit:.2f}")
            return True
            
        except Exception as e:
            print(f"[ERROR] Error closing position {ticket}: {e}")
            return False
    
    def process_symbol(self, symbol: str):
        """Process a single symbol for trading signals"""
        try:
            # Get historical data
            df = self.get_historical_data(symbol)
            if df.empty:
                print(f"[ERROR] No price data for {symbol}")
                return
            
            # Calculate signals
            signal, confidence = self.calculate_signals(symbol, df)
            
            # Execute trade if signal is strong enough
            if signal != "HOLD" and confidence > 0.6:
                print(f"[SIGNAL] Executing trade: {symbol} {signal} (confidence: {confidence:.2f})")
                
                # Determine order type
                order_type = "BUY" if signal == "BUY" else "SELL"
                
                # Calculate volume based on account balance and risk
                volume = 0.01  # Default small volume
                
                # Place the order
                if self.place_order(symbol, order_type, volume):
                    print(f"[SUCCESS] Trade placed for {symbol}")
                else:
                    print(f"[ERROR] Failed to place trade for {symbol}")
            
        except Exception as e:
            print(f"[ERROR] Error processing {symbol}: {e}")
    
    def trading_loop(self):
        """Main trading loop"""
        print(f"[INFO] Starting trading loop...")
        
        while self.running:
            try:
                # Process each symbol
                for symbol in self.symbols:
                    if not self.running:
                        break
                    self.process_symbol(symbol)
                    time.sleep(1)  # Small delay between symbols
                
                # Check and manage existing positions
                self.check_and_manage_positions()
                
                # Wait before next cycle
                time.sleep(MT5Config.TRADING_INTERVAL)
                
            except Exception as e:
                print(f"[ERROR] Error in trading loop: {e}")
                time.sleep(5)  # Wait before retrying
    
    def start(self):
        """Start the trading bot"""
        print(f"[INFO] Strategy: {self.strategy_type}")
        print(f"[INFO] Symbols: {', '.join(self.symbols)}")
        
        # Connect to MT5
        if not self.connect():
            print("[ERROR] Failed to connect to MT5")
            return
        
        self.running = True
        print("[SUCCESS] MT5 Trading bot started successfully!")
        
        try:
            # Start the trading loop
            self.trading_loop()
        except KeyboardInterrupt:
            print("\n[STOP] Trading interrupted by user")
        except Exception as e:
            print(f"[ERROR] Trading error: {e}")
        finally:
            self.stop()
    
    def stop(self):
        """Stop the trading bot"""
        self.running = False
        self.disconnect()
        print("[STOP] MT5 Trading bot stopped")
        self._print_final_statistics()
    
    def _print_final_statistics(self):
        """Print final trading statistics"""
        print("\n📊 FINAL TRADING STATISTICS:")
        print("=" * 50)
        
        total_trades = 0
        total_wins = 0
        total_losses = 0
        total_pnl = 0.0
        
        for symbol, perf in self.symbol_performance.items():
            if perf['trades'] > 0:
                win_rate = (perf['wins'] / perf['trades'] * 100) if perf['trades'] > 0 else 0
                print(f"{symbol}:")
                print(f"  Trades: {perf['trades']}")
                print(f"  Win Rate: {win_rate:.1f}%")
                print(f"  P&L: ${perf['total_pnl']:.2f}")
                print()
                
                total_trades += perf['trades']
                total_wins += perf['wins']
                total_losses += perf['losses']
                total_pnl += perf['total_pnl']
        
        if total_trades > 0:
            overall_win_rate = (total_wins / total_trades * 100)
            print(f"OVERALL:")
            print(f"  Total Trades: {total_trades}")
            print(f"  Win Rate: {overall_win_rate:.1f}%")
            print(f"  Total P&L: ${total_pnl:.2f}")
    
    def check_and_manage_positions(self):
        """Check and manage existing positions"""
        try:
            positions = mt5.positions_get()
            if not positions:
                return
            
            for position in positions:
                # Calculate profit percentage
                profit_percent = (position.profit / position.price) * 100 if position.price > 0 else 0
                
                # Take partial profit if threshold reached
                if profit_percent >= 50:  # 50% profit
                    self.close_partial_position(position.ticket, position.volume * 0.5)
                    print(f"[PROFIT] Partial profit taken: {position.symbol} at {profit_percent:.2f}% profit")
                
                # Update trailing stop
                self.update_trailing_stop(position, profit_percent)
                
        except Exception as e:
            print(f"[ERROR] Error managing positions: {e}")
    
    def close_partial_position(self, ticket: int, volume: float) -> bool:
        """Close a partial position"""
        try:
            # Get position info
            position = mt5.positions_get(ticket=ticket)
            if not position:
                return False
            
            position = position[0]
            
            # Validate volume
            if volume >= position.volume:
                print(f"[ERROR] Invalid volume for partial close: {volume} (position volume: {position.volume})")
                return False
            
            # Prepare partial close request
            request = {
                "action": mt5.TRADE_ACTION_DEAL,
                "symbol": position.symbol,
                "volume": volume,
                "type": mt5.ORDER_TYPE_SELL if position.type == mt5.POSITION_TYPE_BUY else mt5.ORDER_TYPE_BUY,
                "position": ticket,
                "price": mt5.symbol_info_tick(position.symbol).ask if position.type == mt5.POSITION_TYPE_BUY else mt5.symbol_info_tick(position.symbol).bid,
                "deviation": 20,
                "magic": 234000,
                "comment": "python script partial close",
                "type_time": mt5.ORDER_TIME_GTC,
                "type_filling": mt5.ORDER_FILLING_IOC,
            }
            
            # Send partial close request
            result = mt5.order_send(request)
            if result.retcode != mt5.TRADE_RETCODE_DONE:
                print(f"[ERROR] Failed to close partial position {ticket}: {result.comment}")
                return False
            
            print(f"[SUCCESS] Partial position closed: {ticket} - Volume: {volume}")
            return True
            
        except Exception as e:
            print(f"[ERROR] Error closing partial position {ticket}: {e}")
            return False
    
    def update_trailing_stop(self, position, profit_percent: float):
        """Update trailing stop for a position"""
        try:
            # Simple trailing stop logic
            if profit_percent >= 20:  # Start trailing at 20% profit
                new_sl = position.price_open * 1.05  # 5% profit lock
                self.modify_position_sl(position.ticket, new_sl)
                
        except Exception as e:
            print(f"[ERROR] Error updating trailing stop for {position.symbol}: {e}")
    
    def modify_position_sl(self, ticket: int, new_sl: float) -> bool:
        """Modify stop loss for a position"""
        try:
            request = {
                "action": mt5.TRADE_ACTION_SLTP,
                "position": ticket,
                "symbol": "Volatility 100 Index",
                "sl": new_sl,
                "tp": 0.0
            }
            
            result = mt5.order_send(request)
            if result.retcode != mt5.TRADE_RETCODE_DONE:
                print(f"[ERROR] Failed to modify SL for position {ticket}: {result.comment}")
                return False
            
            print(f"[SUCCESS] Updated SL for position {ticket} to {new_sl:.5f}")
            return True
            
        except Exception as e:
            print(f"[ERROR] Error modifying SL for position {ticket}: {e}")
            return False

def main():
    """Main entry point"""
    # You can customize which symbols to trade
    # symbols = ['EURUSD', 'GBPUSD', 'USDJPY']  # Example: trade only specific ones
    
    bot = MT5TradingBot()  # Will trade all major forex pairs and indices
    bot.start()

if __name__ == "__main__":
    main() 