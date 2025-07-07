import time
import json
import signal
import sys
from datetime import datetime
from typing import Dict, List, Optional

from .api_client import DerivAPIClient
from .strategies import MultiStrategy, MovingAverageCrossover, RSIStrategy
from .risk_manager import RiskManager
from .config import DerivConfig

class DerivTradingBot:
    """Main trading bot for Deriv synthetic indices"""
    
    def __init__(self, symbols: Optional[List[str]] = None, strategy_type: str = 'multi'):
        # Default symbols - all synthetic indices and boom/crash indices
        if symbols is None:
            self.symbols = [
                # Volatility Indices
                'R_10', 'R_25', 'R_50', 'R_75', 'R_100'
                # Boom & Crash Indices - REMOVED (not available on this account)
                # 'BOOM1000', 'CRASH1000', 'BOOM500', 'CRASH500'
            ]
        else:
            self.symbols = symbols
            
        self.strategy_type = strategy_type
        self.api_client = DerivAPIClient()
        self.risk_manager = RiskManager()
        self.strategies = {}
        self.running = False
        self.active_contracts = {}
        self.symbol_performance = {}  # Track performance per symbol
        
        # Initialize strategies for each symbol
        for symbol in self.symbols:
            if strategy_type == 'multi':
                self.strategies[symbol] = MultiStrategy(symbol)
            elif strategy_type == 'ma':
                self.strategies[symbol] = MovingAverageCrossover(symbol)
            elif strategy_type == 'rsi':
                self.strategies[symbol] = RSIStrategy(symbol)
            else:
                self.strategies[symbol] = MultiStrategy(symbol)
            
            # Initialize performance tracking
            self.symbol_performance[symbol] = {
                'trades': 0,
                'wins': 0,
                'losses': 0,
                'total_pnl': 0.0,
                'last_signal': None,
                'last_signal_time': None
            }
        
        # Set up signal handlers for graceful shutdown
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)
    
    def _signal_handler(self, signum, frame):
        """Handle shutdown signals"""
        print(f"\n🛑 Received signal {signum}, shutting down gracefully...")
        self.stop()
        sys.exit(0)
    
    def start(self):
        """Start the trading bot"""
        print("🚀 Starting Deriv Trading Bot...")
        print(f"📈 Trading symbols: {', '.join(self.symbols)}")
        print(f"🎯 Strategy: {self.strategy_type}")
        print(f"📊 Total instruments: {len(self.symbols)}")
        
        # Validate configuration
        try:
            DerivConfig.validate_config()
        except ValueError as e:
            print(f"❌ Configuration error: {e}")
            return False
        
        # Connect to API
        if not self.api_client.connect():
            print("❌ Failed to connect to Deriv API")
            return False
        
        # Wait for authorization
        time.sleep(2)
        
        # Get initial balance
        self._get_balance()
        
        # Start trading
        self.running = True
        print("✅ Trading bot started successfully!")
        
        try:
            self._trading_loop()
        except KeyboardInterrupt:
            print("\n🛑 Trading interrupted by user")
        except Exception as e:
            print(f"❌ Trading error: {e}")
        finally:
            self.stop()
        
        return True
    
    def stop(self):
        """Stop the trading bot"""
        self.running = False
        self.api_client.disconnect()
        print("🛑 Trading bot stopped")
        self._print_final_statistics()
    
    def _get_balance(self):
        """Get and update account balance"""
        # Wait a bit for authorization to complete
        time.sleep(2)
        
        # Use balance from authorization response
        if hasattr(self.api_client, 'account_balance') and self.api_client.account_balance > 0:
            balance = self.api_client.account_balance
            self.risk_manager.update_balance(balance)
            print(f"💰 Using balance from authorization: ${balance:.2f}")
        else:
            # Fallback to get_balance request
            def balance_callback(data):
                if 'error' not in data:
                    # Try different possible balance fields
                    balance = 0
                    if 'get_settings' in data:
                        balance = float(data.get('get_settings', {}).get('balance', 0))
                    elif 'authorize' in data:
                        balance = float(data.get('authorize', {}).get('balance', 0))
                    
                    if balance > 0:
                        self.risk_manager.update_balance(balance)
                        print(f"💰 Current balance: ${balance:.2f}")
                    else:
                        print(f"⚠️  Could not parse balance from response: {data}")
            
            self.api_client.get_balance(balance_callback)
            time.sleep(1)
    
    def _trading_loop(self):
        """Main trading loop"""
        print("🔄 Starting trading loop...")
        
        while self.running:
            try:
                # Check account status and profit targets
                account_status = self.risk_manager.get_account_status()
                if account_status:
                    print(f"📊 Account Status: ${account_status['current_balance']:.2f} | P&L: {account_status['profit_percentage']:.2%} | Risk Budget: ${account_status['remaining_risk_budget']:.2f}")
                    
                    # Check if we should take profit
                    if account_status['should_take_profit']:
                        print(f"🎯 Profit target reached! Current profit: {account_status['profit_percentage']:.2%}")
                        # Close all active positions
                        self._close_all_positions()
                        print("✅ All positions closed for profit taking")
                        time.sleep(60)  # Wait before resuming
                        continue
                
                # Check if we should stop trading
                if self.risk_manager.should_stop_trading():
                    print("⏸️  Risk limits reached, pausing trading...")
                    time.sleep(60)  # Wait 1 minute before checking again
                    continue
                
                print(f"🔄 Processing {len(self.symbols)} symbols...")
                
                # Process each symbol with rotation
                for i, symbol in enumerate(self.symbols):
                    if not self.running:
                        break
                    
                    print(f"📊 Processing {symbol} ({i+1}/{len(self.symbols)})")
                    
                    # Add small delay between symbols to avoid overwhelming the API
                    if i > 0:
                        time.sleep(1)
                    
                    self._process_symbol(symbol)
                
                # Wait before next iteration
                print("⏳ Waiting 10 seconds before next iteration...")
                time.sleep(10)  # Increased delay for multi-instrument trading
                
            except Exception as e:
                print(f"❌ Error in trading loop: {e}")
                time.sleep(10)
    
    def _process_symbol(self, symbol: str):
        """Process a single symbol for trading opportunities"""
        try:
            # Get historical data for analysis
            def candles_callback(data):
                if 'error' not in data and 'candles' in data:
                    candles = data['candles']
                    strategy = self.strategies[symbol]
                    
                    # Add candles to strategy
                    for candle in candles:
                        strategy.add_candle(candle)
                    
                    # Get trading signal
                    signal, confidence = strategy.get_signal()
                    
                    # Update performance tracking
                    self.symbol_performance[symbol]['last_signal'] = signal
                    self.symbol_performance[symbol]['last_signal_time'] = datetime.now()
                    
                    # Debug output for all signals
                    if signal != "HOLD":
                        print(f"🔍 Signal detected: {symbol} {signal} (confidence: {confidence:.2f})")
                    
                    # Take trades at 0.3 confidence threshold
                    if signal != "HOLD" and confidence > 0.3:
                        print(f"🎯 Executing trade: {symbol} {signal} (confidence: {confidence:.2f})")
                        self._execute_trade(symbol, signal, confidence)
                    elif signal != "HOLD":
                        print(f"⚠️  Signal too weak: {symbol} {signal} (confidence: {confidence:.2f} < 0.3)")
                else:
                    if 'error' in data:
                        print(f"❌ Error getting candles for {symbol}: {data['error']}")
                    else:
                        print(f"⚠️  No candles data for {symbol}")
            
            # Get recent candles (last 100)
            self.api_client.get_candles(symbol, 60, 100, candles_callback)
            
        except Exception as e:
            print(f"❌ Error processing {symbol}: {e}")
    
    def _execute_trade(self, symbol: str, direction: str, confidence: float):
        """Execute a trade based on signal"""
        try:
            # Calculate position size with aggressive leverage
            base_position_size = self.risk_manager.calculate_position_size(confidence)
            
            # For strong signals, open multiple positions
            if confidence >= 0.6:
                num_positions = min(3, int(confidence * 5))  # Up to 3 positions for very strong signals
                position_size = base_position_size
            else:
                num_positions = 1
                position_size = base_position_size
            
            print(f"🎯 Signal: {symbol} {direction} (confidence: {confidence:.2f})")
            print(f"💰 Position size: ${position_size:.2f} x {num_positions} positions")
            
            # Check if we can trade
            if not self.risk_manager.can_trade(position_size * num_positions):
                print(f"❌ Risk manager blocked trade for {symbol}")
                return
            
            print(f"✅ Risk check passed, placing {num_positions} trade(s)...")
            
            # Place multiple trades for strong signals
            for i in range(num_positions):
                def trade_callback(data, pos_num=i+1):
                    if 'error' not in data and 'buy' in data:
                        contract_id = data['buy']['contract_id']
                        self.active_contracts[contract_id] = {
                            'symbol': symbol,
                            'direction': direction,
                            'amount': position_size,
                            'timestamp': datetime.now(),
                            'confidence': confidence,
                            'position_number': pos_num
                        }
                        print(f"✅ Trade {pos_num} placed: {symbol} {direction} - {contract_id}")
                        
                        # Update performance tracking
                        self.symbol_performance[symbol]['trades'] += 1
                        
                        # Monitor the contract
                        self._monitor_contract(contract_id)
                    else:
                        error_msg = data.get('error', {}).get('message', 'Unknown error')
                        print(f"❌ Trade {pos_num} failed for {symbol}: {error_msg}")
                
                # Buy contract (5 ticks duration)
                self.api_client.buy_contract(symbol, position_size, direction, 5, trade_callback)
                
                # Small delay between multiple trades
                if i < num_positions - 1:
                    time.sleep(0.5)
            
        except Exception as e:
            print(f"❌ Error executing trade for {symbol}: {e}")
    
    def _close_all_positions(self):
        """Close all active positions"""
        if not self.active_contracts:
            return
        
        print(f"🔄 Closing {len(self.active_contracts)} active positions...")
        
        for contract_id in list(self.active_contracts.keys()):
            try:
                # Sell the contract
                def sell_callback(data):
                    if 'error' not in data and 'sell' in data:
                        print(f"✅ Position closed: {contract_id}")
                    else:
                        error_msg = data.get('error', {}).get('message', 'Unknown error')
                        print(f"❌ Failed to close position {contract_id}: {error_msg}")
                
                self.api_client.send_request({
                    "sell": contract_id,
                    "price": 0  # Market price
                }, sell_callback)
                
                # Remove from active contracts
                if contract_id in self.active_contracts:
                    del self.active_contracts[contract_id]
                    
            except Exception as e:
                print(f"❌ Error closing position {contract_id}: {e}")
    
    def _monitor_contract(self, contract_id: str):
        """Monitor an active contract for completion with early profit-taking"""
        def contract_callback(data):
            if 'error' not in data and 'proposal_open_contract' in data:
                contract = data['proposal_open_contract']
                symbol = contract.get('underlying_symbol', 'Unknown')
                profit = float(contract.get('profit', 0))
                status = contract.get('status', 'Unknown')
                
                # Check if we should close early for profit or to prevent loss
                if status == 'open':
                    # Check account status for profit target
                    account_status = self.risk_manager.get_account_status()
                    if account_status and account_status['should_take_profit']:
                        print(f"🎯 Closing {symbol} early for profit target")
                        self._close_position(contract_id)
                        return
                    
                    # Check if position is going negative and we want to prevent loss
                    if profit < 0 and contract_id in self.active_contracts:
                        position_info = self.active_contracts[contract_id]
                        # Close if loss exceeds 50% of position amount
                        if abs(profit) > position_info['amount'] * 0.5:
                            print(f"🛑 Closing {symbol} to prevent further loss (current loss: ${profit:.2f})")
                            self._close_position(contract_id)
                            return
                
                if status == 'sold':
                    # Update performance tracking
                    if symbol in self.symbol_performance:
                        perf = self.symbol_performance[symbol]
                        perf['total_pnl'] += profit
                        if profit > 0:
                            perf['wins'] += 1
                        else:
                            perf['losses'] += 1
                    
                    # Record the trade
                    self.risk_manager.record_trade(
                        self.active_contracts[contract_id]['amount'],
                        profit,
                        symbol,
                        self.active_contracts[contract_id]['direction']
                    )
                    
                    print(f"📊 Contract closed: {symbol} - Profit: ${profit:.2f}")
                    
                    # Remove from active contracts
                    if contract_id in self.active_contracts:
                        del self.active_contracts[contract_id]
        
        # Check contract status every 10 seconds
        def check_contract():
            if contract_id in self.active_contracts:
                self.api_client.send_request({
                    "proposal_open_contract": 1,
                    "contract_id": contract_id,
                    "subscribe": 1
                }, contract_callback)
        
        # Initial check
        check_contract()
        
        # Schedule periodic checks
        import threading
        def periodic_check():
            for _ in range(30):  # Check for 5 minutes (30 * 10 seconds)
                if not self.running or contract_id not in self.active_contracts:
                    break
                time.sleep(10)
                check_contract()
        
        threading.Thread(target=periodic_check, daemon=True).start()
    
    def _close_position(self, contract_id: str):
        """Close a specific position"""
        try:
            def sell_callback(data):
                if 'error' not in data and 'sell' in data:
                    print(f"✅ Position closed early: {contract_id}")
                else:
                    error_msg = data.get('error', {}).get('message', 'Unknown error')
                    print(f"❌ Failed to close position {contract_id}: {error_msg}")
            
            self.api_client.send_request({
                "sell": contract_id,
                "price": 0  # Market price
            }, sell_callback)
            
        except Exception as e:
            print(f"❌ Error closing position {contract_id}: {e}")
    
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
        
        # Print risk manager stats
        risk_stats = self.risk_manager.get_daily_stats()
        print(f"\nRisk Summary:")
        print(f"  Daily P&L: ${risk_stats['net_pnl']:.2f}")
        print(f"  Consecutive Losses: {risk_stats['consecutive_losses']}")

def main():
    """Main entry point"""
    # You can customize which symbols to trade
    # symbols = ['R_100', 'BOOM1000', 'CRASH1000']  # Example: trade only specific ones
    
    bot = DerivTradingBot()  # Will trade all synthetic and boom/crash indices
    bot.start()

if __name__ == "__main__":
    main() 