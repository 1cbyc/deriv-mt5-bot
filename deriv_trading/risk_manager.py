import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional
from .config import DerivConfig

class RiskManager:
    """Risk management system for Deriv trading"""
    
    def __init__(self):
        self.daily_trades = []
        self.daily_loss = 0.0
        self.daily_start = datetime.now().date()
        self.max_consecutive_losses = 10
        self.consecutive_losses = 0
        self.max_drawdown = 0.20  # 20% max drawdown
        self.initial_balance = None
        self.current_balance = None
        
        # Dynamic risk management based on account balance
        self.max_account_loss_percentage = 0.25  # 25% max loss
        self.profit_target_percentage = 0.10     # 10% profit target
        self.max_position_percentage = 0.05      # 5% max per position
        
        # Track account performance
        self.peak_balance = None
        self.current_drawdown = 0.0
        self.total_profit_loss = 0.0
        
    def reset_daily_stats(self):
        """Reset daily statistics if it's a new day"""
        current_date = datetime.now().date()
        if current_date != self.daily_start:
            self.daily_trades = []
            self.daily_loss = 0.0
            self.daily_start = current_date
            print(f"📅 New trading day started: {current_date}")
    
    def can_trade(self, amount: float) -> bool:
        """Check if we can place a trade based on account balance risk management"""
        self.reset_daily_stats()
        
        if not self.current_balance or self.current_balance <= 0:
            print("❌ No balance available for trading")
            return False
        
        # Check if we've exceeded 25% account loss limit
        if self.initial_balance:
            current_loss = self.initial_balance - self.current_balance
            max_allowed_loss = self.initial_balance * self.max_account_loss_percentage
            
            if current_loss >= max_allowed_loss:
                print(f"❌ Maximum account loss reached: ${current_loss:.2f} (25% of ${self.initial_balance:.2f})")
                return False
            
            # Check remaining risk budget
            remaining_budget = max_allowed_loss - current_loss
            if amount > remaining_budget * 0.5:  # Don't use more than 50% of remaining budget
                print(f"❌ Trade amount too high for remaining risk budget: ${amount:.2f} > ${remaining_budget * 0.5:.2f}")
                return False
        
        # Check if trade amount exceeds 5% of current balance
        trade_percentage = amount / self.current_balance
        if trade_percentage > self.max_position_percentage:
            print(f"❌ Trade amount too high ({trade_percentage:.2%} of balance, max {self.max_position_percentage:.1%})")
            return False
        
        # Check daily trade limit
        if len(self.daily_trades) >= DerivConfig.MAX_DAILY_TRADES:
            print(f"❌ Daily trade limit reached ({DerivConfig.MAX_DAILY_TRADES})")
            return False
        
        return True
    
    def calculate_position_size(self, confidence: float, base_amount: Optional[float] = None) -> float:
        """Calculate position size based on account balance and confidence"""
        if base_amount is None:
            base_amount = DerivConfig.DEFAULT_AMOUNT
        
        # Ensure base_amount is float
        base_amount = float(base_amount)
        
        if not self.current_balance or self.current_balance <= 0:
            return base_amount
        
        # Calculate maximum safe position size based on account balance
        max_safe_position = self.current_balance * self.max_position_percentage
        
        # Calculate remaining risk budget
        current_loss = self.initial_balance - self.current_balance if self.initial_balance else 0
        max_allowed_loss = self.initial_balance * self.max_account_loss_percentage if self.initial_balance else 0
        remaining_risk_budget = max(0, max_allowed_loss - current_loss)
        
        # Position size based on confidence and remaining risk budget
        if confidence >= 0.8:
            leverage_multiplier = 3.0  # Very high confidence
        elif confidence >= 0.7:
            leverage_multiplier = 2.0  # High confidence
        elif confidence >= 0.6:
            leverage_multiplier = 1.5  # Medium-high confidence
        else:
            leverage_multiplier = 1.0  # Low confidence
        
        # Calculate position size
        position_size = base_amount * leverage_multiplier
        
        # Limit by remaining risk budget
        if remaining_risk_budget > 0:
            position_size = min(position_size, remaining_risk_budget * 0.5)  # Use 50% of remaining budget
        
        # Limit by max safe position
        position_size = min(position_size, max_safe_position)
        
        # Ensure minimum and maximum limits
        position_size = max(1.0, min(position_size, 200.0))
        
        return round(position_size, 2)
    
    def record_trade(self, amount: float, profit_loss: float, symbol: str, direction: str):
        """Record a completed trade"""
        self.reset_daily_stats()
        
        trade_record = {
            'timestamp': datetime.now(),
            'symbol': symbol,
            'direction': direction,
            'amount': amount,
            'profit_loss': profit_loss,
            'balance_after': self.current_balance + profit_loss if self.current_balance else None
        }
        
        self.daily_trades.append(trade_record)
        
        # Update daily loss
        if profit_loss < 0:
            self.daily_loss += abs(profit_loss)
        
        # Update consecutive losses
        if profit_loss < 0:
            self.consecutive_losses += 1
        else:
            self.consecutive_losses = 0
        
        # Update current balance
        if self.current_balance:
            self.current_balance += profit_loss
        
        print(f"📊 Trade recorded: {symbol} {direction} ${amount} → ${profit_loss:.2f}")
        print(f"💰 Daily P&L: ${self.daily_loss:.2f} | Consecutive losses: {self.consecutive_losses}")
    
    def update_balance(self, balance: float):
        """Update current account balance and track performance"""
        if self.initial_balance is None:
            self.initial_balance = balance
            self.peak_balance = balance
            print(f"💰 Initial balance set: ${balance:.2f}")
        
        self.current_balance = balance
        
        # Update peak balance
        if self.peak_balance is None or balance > self.peak_balance:
            self.peak_balance = balance
        
        # Calculate current drawdown
        if self.peak_balance:
            self.current_drawdown = (self.peak_balance - balance) / self.peak_balance
        
        # Calculate total P&L
        if self.initial_balance:
            self.total_profit_loss = balance - self.initial_balance
    
    def should_take_profit(self) -> bool:
        """Check if we should take profit based on 10% account target"""
        if not self.initial_balance or not self.current_balance:
            return False
        
        profit_percentage = (self.current_balance - self.initial_balance) / self.initial_balance
        return profit_percentage >= self.profit_target_percentage
    
    def get_account_status(self) -> Dict:
        """Get current account status and risk metrics"""
        if not self.initial_balance or not self.current_balance:
            return {}
        
        current_loss = self.initial_balance - self.current_balance
        max_allowed_loss = self.initial_balance * self.max_account_loss_percentage
        remaining_budget = max(0, max_allowed_loss - current_loss)
        profit_percentage = (self.current_balance - self.initial_balance) / self.initial_balance
        
        return {
            'initial_balance': self.initial_balance,
            'current_balance': self.current_balance,
            'total_pnl': self.total_profit_loss,
            'profit_percentage': profit_percentage,
            'current_loss': current_loss,
            'max_allowed_loss': max_allowed_loss,
            'remaining_risk_budget': remaining_budget,
            'current_drawdown': self.current_drawdown,
            'should_take_profit': self.should_take_profit()
        }
    
    def get_daily_stats(self) -> Dict:
        """Get daily trading statistics"""
        self.reset_daily_stats()
        
        total_trades = len(self.daily_trades)
        winning_trades = len([t for t in self.daily_trades if t['profit_loss'] > 0])
        losing_trades = len([t for t in self.daily_trades if t['profit_loss'] < 0])
        
        win_rate = (winning_trades / total_trades * 100) if total_trades > 0 else 0
        
        total_profit = sum(t['profit_loss'] for t in self.daily_trades if t['profit_loss'] > 0)
        total_loss = sum(abs(t['profit_loss']) for t in self.daily_trades if t['profit_loss'] < 0)
        
        return {
            'total_trades': total_trades,
            'winning_trades': winning_trades,
            'losing_trades': losing_trades,
            'win_rate': win_rate,
            'total_profit': total_profit,
            'total_loss': total_loss,
            'net_pnl': total_profit - total_loss,
            'daily_loss': self.daily_loss,
            'consecutive_losses': self.consecutive_losses
        }
    
    def should_stop_trading(self) -> bool:
        """Check if we should stop trading for the day"""
        stats = self.get_daily_stats()
        
        # Stop if win rate is too low
        if stats['total_trades'] >= 5 and stats['win_rate'] < 30:
            print(f"❌ Win rate too low ({stats['win_rate']:.1f}%) - stopping for the day")
            return True
        
        # Stop if daily loss limit reached
        if self.daily_loss >= DerivConfig.MAX_DAILY_LOSS:
            print(f"❌ Daily loss limit reached - stopping for the day")
            return True
        
        # Stop if too many consecutive losses
        if self.consecutive_losses >= self.max_consecutive_losses:
            print(f"❌ Too many consecutive losses - stopping for the day")
            return True
        
        return False
    
    def get_risk_summary(self) -> str:
        """Get a summary of current risk status"""
        stats = self.get_daily_stats()
        
        summary = f"""
📊 RISK SUMMARY:
├── Daily Trades: {stats['total_trades']}/{DerivConfig.MAX_DAILY_TRADES}
├── Win Rate: {stats['win_rate']:.1f}%
├── Daily P&L: ${stats['net_pnl']:.2f}
├── Daily Loss: ${self.daily_loss:.2f}/${DerivConfig.MAX_DAILY_LOSS}
├── Consecutive Losses: {self.consecutive_losses}/{self.max_consecutive_losses}
└── Balance: ${self.current_balance:.2f} (Initial: ${self.initial_balance:.2f})
"""
        return summary 