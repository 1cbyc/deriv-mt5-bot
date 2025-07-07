import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
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
        
        # Risk management tracking
        self.daily_pnl = 0.0
        self.daily_trades = 0
        self.session_start_time = datetime.now()
        
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
                'last_confidence': 0.0,
                'consecutive_losses': 0,
                'max_consecutive_losses': 0
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
    
    def calculate_position_size(self, symbol: str, account_balance: float, risk_percent: float = 2.0) -> float:
        """Calculate position size based on account balance and risk management"""
        try:
            # Get symbol info for proper calculations
            symbol_info = mt5.symbol_info(symbol)
            if not symbol_info:
                print(f"[ERROR] No symbol info for {symbol}")
                return MT5Config.get_min_volume(symbol)
            
            # Get current price
            tick = mt5.symbol_info_tick(symbol)
            if not tick:
                print(f"[ERROR] No tick data for {symbol}")
                return MT5Config.get_min_volume(symbol)
            
            current_price = tick.ask
            
            # Calculate proper pip value for this symbol
            pip_size = symbol_info.point
            pip_value = pip_size * 10  # 1 pip = 10 points for most synthetic indices
            
            # Calculate risk amount (2% of account balance)
            risk_amount = account_balance * (risk_percent / 100.0)
            
            # Calculate stop loss distance in pips
            stop_loss_pips = 20  # 20 pip stop loss
            
            # Calculate position size based on risk
            # Risk = Position Size * Stop Loss Distance * Pip Value
            # Position Size = Risk / (Stop Loss Distance * Pip Value)
            position_size = risk_amount / (stop_loss_pips * pip_value)
            
            # Ensure minimum volume
            min_volume = MT5Config.get_min_volume(symbol)
            position_size = max(position_size, min_volume)
            
            # Ensure maximum position size (5% of account)
            max_position_value = account_balance * (MT5Config.MAX_POSITION_SIZE_PERCENT / 100.0)
            max_position_size = max_position_value / current_price
            position_size = min(position_size, max_position_size)
            
            # Round to appropriate decimal places
            if symbol.startswith("Volatility"):
                position_size = round(position_size, 3)  # 0.001 precision
            else:
                position_size = round(position_size, 2)  # 0.01 precision
            
            print(f"[SIZE] {symbol}: Balance=${account_balance:.2f}, Risk=${risk_amount:.2f}, Size={position_size}")
            print(f"[SIZE] Pip Value: {pip_value}, Stop Loss: {stop_loss_pips} pips")
            
            return position_size
            
        except Exception as e:
            print(f"[ERROR] Error calculating position size for {symbol}: {e}")
            return MT5Config.get_min_volume(symbol)
    
    def check_risk_limits(self) -> bool:
        """Check if we can take new trades based on risk limits"""
        try:
            account_info = mt5.account_info()
            if not account_info:
                return False
            
            # Check daily loss limit
            if self.daily_pnl <= -MT5Config.MAX_DAILY_LOSS:
                print(f"[RISK] Daily loss limit reached: ${self.daily_pnl:.2f}")
                return False
            
            # Check daily profit limit
            if self.daily_pnl >= MT5Config.MAX_DAILY_PROFIT:
                print(f"[RISK] Daily profit limit reached: ${self.daily_pnl:.2f}")
                return False
            
            # Check maximum positions
            total_positions = mt5.positions_total()
            if total_positions >= MT5Config.MAX_TOTAL_POSITIONS:
                print(f"[RISK] Maximum positions reached: {total_positions}")
                return False
            
            # Check position correlation (avoid too many similar positions)
            if self._check_position_correlation():
                print(f"[RISK] Too many correlated positions")
                return False
            
            # Check consecutive losses
            if self._check_consecutive_losses():
                print(f"[RISK] Too many consecutive losses")
                return False
            
            return True
            
        except Exception as e:
            print(f"[ERROR] Error checking risk limits: {e}")
            return False
    
    def _check_position_correlation(self) -> bool:
        """Check if we have too many correlated positions"""
        try:
            positions = mt5.positions_get()
            if not positions:
                return False
            
            # Count positions by type (BUY/SELL)
            buy_positions = 0
            sell_positions = 0
            
            for position in positions:
                if position.type == mt5.POSITION_TYPE_BUY:
                    buy_positions += 1
                else:
                    sell_positions += 1
            
            # If we have more than 70% of positions in one direction, it's too correlated
            total_positions = len(positions)
            if total_positions > 0:
                buy_ratio = buy_positions / total_positions
                sell_ratio = sell_positions / total_positions
                
                if buy_ratio > 0.7 or sell_ratio > 0.7:
                    print(f"[RISK] Position correlation: {buy_ratio:.1%} BUY, {sell_ratio:.1%} SELL")
                    return True
            
            return False
            
        except Exception as e:
            print(f"[ERROR] Error checking position correlation: {e}")
            return False
    
    def _check_consecutive_losses(self) -> bool:
        """Check if we have too many consecutive losses"""
        try:
            # Check overall consecutive losses across all symbols
            max_consecutive_losses = 0
            for symbol, perf in self.symbol_performance.items():
                max_consecutive_losses = max(max_consecutive_losses, perf.get('consecutive_losses', 0))
            
            # If we have more than 3 consecutive losses, reduce trading
            if max_consecutive_losses >= 3:
                print(f"[RISK] Too many consecutive losses: {max_consecutive_losses}")
                return True
            
            return False
            
        except Exception as e:
            print(f"[ERROR] Error checking consecutive losses: {e}")
            return False
    
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
                    "comment": "pythonMT5bot",
                    "type_time": mt5.ORDER_TIME_GTC,
                    "type_filling": mt5.ORDER_FILLING_FOK,
                }
                
                # Add stop loss and take profit if provided
                if sl > 0:
                    request["sl"] = sl
                if tp > 0:
                    request["tp"] = tp
                
                # Send the order
                result = mt5.order_send(request)
                
                # Check if order was successful
                if result.retcode != mt5.TRADE_RETCODE_DONE:
                    print(f"[ERROR] Order failed for {symbol} (attempt {attempt + 1}): {result.comment}")
                    
                    # Try with smaller volume for volume-related errors
                    if "volume" in result.comment.lower():
                        current_volume = max(0.01, current_volume * 0.5)
                        print(f"[RETRY] Trying with volume: {current_volume}")
                        continue
                    else:
                        break
                else:
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
            
            except Exception as e:
                print(f"[ERROR] Error placing order for {symbol} (attempt {attempt + 1}): {e}")
                return False
        
        # If we get here, all attempts failed
        print(f"[ERROR] All attempts failed for {symbol}")
        return False
    
    def close_position(self, ticket: int) -> bool:
        """Close a position by ticket, retrying with different filling modes"""
        try:
            position = mt5.positions_get(ticket=ticket)
            if not position:
                print(f"[ERROR] Position {ticket} not found")
                return False
            position = position[0]
            
            # Try different filling modes
            filling_modes = [mt5.ORDER_FILLING_FOK, mt5.ORDER_FILLING_IOC, mt5.ORDER_FILLING_RETURN]
            
            for fill_mode in filling_modes:
                request = {
                    "action": mt5.TRADE_ACTION_DEAL,
                    "symbol": position.symbol,
                    "volume": position.volume,
                    "type": mt5.ORDER_TYPE_SELL if position.type == mt5.POSITION_TYPE_BUY else mt5.ORDER_TYPE_BUY,
                    "position": ticket,
                    "price": 0.0,  # Market price
                    "deviation": 20,
                    "magic": 234000,
                    "comment": "pythonMT5bot",
                    "type_time": mt5.ORDER_TIME_GTC,
                    "type_filling": fill_mode,
                }
                
                result = mt5.order_send(request)
                if result and result.retcode == mt5.TRADE_RETCODE_DONE:
                    print(f"[SUCCESS] Position closed: {ticket} - Profit: ${result.profit:.2f}")
                    return True
            
            print(f"[ERROR] Failed to close position {ticket}")
            return False
            
        except Exception as e:
            print(f"[ERROR] Error closing position {ticket}: {e}")
            return False
    
    def process_symbol(self, symbol: str):
        """Process a single symbol for trading signals"""
        try:
            # Check risk limits first
            if not self.check_risk_limits():
                print(f"[SKIP] Risk limits prevent trading {symbol}")
                return
            
            # Get historical data
            df = self.get_historical_data(symbol)
            if df.empty:
                print(f"[ERROR] No price data for {symbol}")
                return
            
            # Calculate signals
            signal, confidence = self.calculate_signals(symbol, df)
            
            # Execute trade if signal is strong enough
            if signal != "HOLD" and confidence > MT5Config.CONFIDENCE_THRESHOLD:
                print(f"[SIGNAL] Executing trade: {symbol} {signal} (confidence: {confidence:.2f})")
                
                # Determine order type
                order_type = "BUY" if signal == "BUY" else "SELL"
                
                # Calculate position size based on account balance
                account_info = mt5.account_info()
                if account_info:
                    volume = self.calculate_position_size(symbol, account_info.balance)
                else:
                    volume = MT5Config.get_min_volume(symbol)
                
                # Place the order
                if self.place_order(symbol, order_type, volume):
                    print(f"[SUCCESS] Trade placed for {symbol}")
                else:
                    print(f"[ERROR] Failed to place trade for {symbol}")
            
        except Exception as e:
            print(f"[ERROR] Error processing {symbol}: {e}")
    
    def check_and_manage_positions(self):
        """Enhanced position management with proper pip calculations and risk management"""
        try:
            positions = mt5.positions_get()
            if not positions:
                return
            
            account_info = mt5.account_info()
            account_balance = account_info.balance if account_info else 100.0
            
            for position in positions:
                symbol = position.symbol
                ticket = position.ticket
                profit = position.profit
                price_open = position.price_open
                price_current = position.price_current
                
                # CORRECTED: Proper pip calculation for synthetic indices
                # For synthetic indices like Volatility 100 Index, 1 pip = 0.01
                # But we need to check the actual symbol properties
                symbol_info = mt5.symbol_info(symbol)
                if symbol_info:
                    # Use the actual point value from symbol info
                    pip_size = symbol_info.point
                    # For most synthetic indices, 1 pip = 10 points
                    pip_value = pip_size * 10
                else:
                    # Fallback for synthetic indices
                    pip_value = 0.01
                
                # Calculate pip movement correctly
                if position.type == mt5.POSITION_TYPE_BUY:
                    pip_movement = (price_current - price_open) / pip_value
                else:
                    pip_movement = (price_open - price_current) / pip_value
                
                # Calculate profit percentage
                profit_percent = (profit / account_balance) * 100
                
                # Get current signal for potential reversal
                current_signal, current_confidence = self._get_current_signal(symbol)
                
                # Update daily P&L
                self.daily_pnl += profit
                
                # --- STOP LOSS LOGIC (20 pips or 1% account loss) ---
                if pip_movement <= -20 or profit_percent <= -1.0:
                    print(f"🛑 Closing {symbol} position {ticket} for stop loss ({pip_movement:.1f} pips, {profit_percent:.2f}%)")
                    if self.close_position(ticket):
                        self._update_performance(symbol, profit)
                    continue
                
                # --- TAKE PROFIT LOGIC (30 pips or 1% account profit) ---
                if pip_movement >= 30 or profit_percent >= 1.0:
                    print(f"💰 Closing {symbol} position {ticket} for take profit ({pip_movement:.1f} pips, {profit_percent:.2f}%)")
                    if self.close_position(ticket):
                        self._update_performance(symbol, profit)
                    continue
                
                # --- SIGNAL REVERSAL LOGIC (conservative - only very strong signals) ---
                if self._is_opposite_signal(position.type, current_signal) and current_confidence > 0.85:
                    print(f"🔄 Closing {symbol} position {ticket} for strong signal reversal (confidence: {current_confidence:.2f})")
                    if self.close_position(ticket):
                        self._update_performance(symbol, profit)
                    continue
                
                # --- ENHANCED TRAILING STOP LOGIC ---
                if pip_movement > 10:  # Start trailing after 10 pips profit
                    self._update_trailing_stop(position, pip_movement, pip_value)
                    
        except Exception as e:
            print(f"[ERROR] Error managing positions: {e}")
    
    def _update_performance(self, symbol: str, profit: float):
        """Update performance tracking for a symbol"""
        if symbol in self.symbol_performance:
            perf = self.symbol_performance[symbol]
            perf['trades'] += 1
            perf['total_pnl'] += profit
            
            if profit > 0:
                perf['wins'] += 1
                perf['consecutive_losses'] = 0
            else:
                perf['losses'] += 1
                perf['consecutive_losses'] += 1
                perf['max_consecutive_losses'] = max(perf['max_consecutive_losses'], perf['consecutive_losses'])
            
            # Update daily tracking
            self.daily_trades += 1
            self.daily_pnl += profit
            
            # Print performance update
            win_rate = (perf['wins'] / perf['trades'] * 100) if perf['trades'] > 0 else 0
            print(f"[PERF] {symbol}: Trade {perf['trades']}, Win Rate: {win_rate:.1f}%, P&L: ${perf['total_pnl']:.2f}")
            print(f"[PERF] Daily P&L: ${self.daily_pnl:.2f}, Daily Trades: {self.daily_trades}")
    
    def _get_current_signal(self, symbol: str) -> Tuple[str, float]:
        """Get current signal and confidence for a symbol"""
        try:
            df = self.get_historical_data(symbol, MT5Config.TIMEFRAME, 50)
            if df.empty:
                return "HOLD", 0.0
            
            strategy = self.strategies.get(symbol)
            if strategy is None:
                return "HOLD", 0.0
            
            strategy.update_data(df)
            signal, confidence = strategy.get_signal()
            
            # Apply strategy weight
            strategy_weight = MT5Config.get_strategy_weight(strategy.__class__.__name__)
            adjusted_confidence = confidence * strategy_weight
            
            return signal, adjusted_confidence
            
        except Exception as e:
            print(f"[ERROR] Error getting current signal for {symbol}: {e}")
            return "HOLD", 0.0
    
    def _is_opposite_signal(self, position_type: int, current_signal: str) -> bool:
        """Check if current signal is opposite to position type"""
        if position_type == mt5.POSITION_TYPE_BUY and current_signal == "SELL":
            return True
        elif position_type == mt5.POSITION_TYPE_SELL and current_signal == "BUY":
            return True
        return False
    
    def _update_trailing_stop(self, position, pip_movement: float, pip_value: float):
        """Enhanced trailing stop with proper pip calculations"""
        try:
            symbol = position.symbol
            ticket = position.ticket
            price_open = position.price_open
            price_current = position.price_current
            
            # Progressive trailing stop based on profit level
            if pip_movement >= 20:
                # After 20 pips profit, lock in 15 pips
                lock_pips = 15
            elif pip_movement >= 15:
                # After 15 pips profit, lock in 10 pips
                lock_pips = 10
            elif pip_movement >= 10:
                # After 10 pips profit, lock in 5 pips
                lock_pips = 5
            else:
                return  # Don't trail yet
            
            # Calculate new stop loss based on profit level
            if position.type == mt5.POSITION_TYPE_BUY:
                # For buy positions, trail below current price
                new_sl = price_open + (lock_pips * pip_value)
            else:
                # For sell positions, trail above current price
                new_sl = price_open - (lock_pips * pip_value)
            
            # Only update if new SL is better than current
            if (position.type == mt5.POSITION_TYPE_BUY and new_sl > position.sl) or \
               (position.type == mt5.POSITION_TYPE_SELL and (new_sl < position.sl or position.sl == 0)):
                print(f"📈 Updating trailing stop for {symbol} position {ticket} to lock in {lock_pips} pips")
                self.modify_position_sl(ticket, new_sl)
                
        except Exception as e:
            print(f"[ERROR] Error updating trailing stop for {position.symbol}: {e}")
    
    def modify_position_sl(self, ticket: int, new_sl: float) -> bool:
        """Modify stop loss for a position"""
        try:
            request = {
                "action": mt5.TRADE_ACTION_SLTP,
                "position": ticket,
                "sl": new_sl,
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
    
    def start(self):
        """Start the trading bot"""
        print(f"[INFO] Strategy: {self.strategy_type}")
        print(f"[INFO] Symbols: {', '.join(self.symbols)}")
        MT5Config.print_config()
        
        # Connect to MT5
        if not self.connect():
            print("[ERROR] Failed to connect to MT5. Please check that MetaTrader 5 is running and demo.env credentials are correct.")
            return
        
        self.running = True
        print("[SUCCESS] MT5 Trading bot started successfully!")
        
        cycles = 0
        try:
            while self.running:
                cycles += 1
                
                # Process each symbol
                for symbol in self.symbols:
                    if not self.running:
                        break
                    
                    print(f"[CYCLE {cycles}] Processing symbol: {symbol}")
                    
                    # Get market data
                    df = self.get_historical_data(symbol, MT5Config.TIMEFRAME, MT5Config.DATA_LOOKBACK)
                    if df.empty:
                        print(f"[WARNING] No data for {symbol}. Waiting for data...")
                        time.sleep(5)
                        continue
                    
                    # Process symbol for trading signals
                    self.process_symbol(symbol)
                    time.sleep(1)
                
                # Manage existing positions
                self.check_and_manage_positions()
                
                # Print status every 5 cycles
                if cycles % 5 == 0:
                    open_positions = mt5.positions_total() if self.connected else 0
                    total_trades = sum(perf['trades'] for perf in self.symbol_performance.values())
                    print(f"[STATUS] Cycles: {cycles}, Trades: {total_trades}, Open positions: {open_positions}, Daily P&L: ${self.daily_pnl:.2f}")
                
                # Check if no trades after 10 cycles
                if cycles == 10 and total_trades == 0:
                    print("[INFO] No trades made after 10 cycles. The bot may be waiting for valid signals.")
                
                time.sleep(MT5Config.TRADING_INTERVAL)
                
        except KeyboardInterrupt:
            print("\n[STOP] Trading interrupted by user")
        except Exception as e:
            print(f"[ERROR] Trading error: {e}")
        finally:
            self.stop()
    
    def stop(self):
        """Stop the trading bot"""
        self.running = False
        print("[STOP] Closing all open positions before shutdown...")
        self.close_all_positions()
        self.disconnect()
        print("[STOP] MT5 Trading bot stopped")
        self._print_final_statistics()
    
    def close_all_positions(self):
        """Close all open positions for safety"""
        try:
            positions = mt5.positions_get()
            if not positions:
                print("[SUCCESS] No open positions to close")
                return
            
            print(f"[STOP] Closing {len(positions)} open positions for safety...")
            
            for position in positions:
                symbol = position.symbol
                ticket = position.ticket
                profit = position.profit
                
                print(f"[CLOSING] {symbol} position {ticket} (profit: ${profit:.2f})")
                self.close_position(ticket)
                
            print("[SUCCESS] All positions closed")
            
        except Exception as e:
            print(f"[ERROR] Error closing all positions: {e}")
    
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
                print(f"  Max Consecutive Losses: {perf['max_consecutive_losses']}")
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
            print(f"  Daily P&L: ${self.daily_pnl:.2f}")

def main():
    """Main entry point"""
    bot = MT5TradingBot()
    bot.start()

if __name__ == "__main__":
    main() 