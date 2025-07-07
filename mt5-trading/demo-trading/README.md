# Demo Trading Bot - Safe Testing Environment

## 🎯 **Purpose**
This folder contains all demo trading files that are **safe for testing**. No real money is involved.

## 📁 **Files in this folder:**

- `main.py` - Main demo trading bot
- `config.py` - Demo configuration (uses demo account)
- `demo.env` - Demo environment variables
- `demo-config.py` - Alternative demo config
- `run-demo.py` - Simple launcher script
- `README.md` - This file

## 🚀 **How to Run Demo Bot:**

### Option 1: Direct run
```bash
cd mt5-trading/demo-trading
py main.py
```

### Option 2: Using launcher
```bash
cd mt5-trading/demo-trading
py run-demo.py
```

## ✅ **Safety Features:**

1. **Demo Account Only** - Uses demo credentials
2. **No Real Money** - All trades are simulated
3. **Safe for Testing** - Can test strategies without risk
4. **Git Safe** - This folder is committed to git (no sensitive data)

## 🔧 **Configuration:**

The demo bot uses:
- **Demo Account**: Deriv demo credentials
- **Symbols**: Volatility 100 Index (single instrument for testing)
- **Strategy**: MultiStrategy
- **Risk Management**: Conservative settings for testing

## 📊 **Expected Behavior:**

- Connects to demo account
- Analyzes market data
- Generates trading signals
- Places demo trades
- Manages positions with stop losses and take profits

## ⚠️ **Important Notes:**

- **This is for testing only** - No real money involved
- **Demo account has virtual money** - Can test with large amounts
- **All trades are simulated** - No actual market impact
- **Safe to experiment** - Try different settings and strategies

## 🔄 **Testing Workflow:**

1. **Start demo bot**: `py run-demo.py`
2. **Monitor signals**: Watch for trading signals
3. **Check trades**: Verify orders are placed correctly
4. **Test strategies**: Try different strategy combinations
5. **Adjust settings**: Modify risk parameters safely

## 📈 **Demo Account Benefits:**

- **Unlimited testing** - No financial risk
- **Real market data** - Same data as real trading
- **Full functionality** - All features work the same
- **Strategy validation** - Test before real trading

## 🎯 **Next Steps:**

After testing in demo:
1. **Verify strategies work** as expected
2. **Check risk management** is functioning
3. **Test position sizing** calculations
4. **Validate profit taking** logic
5. **Confirm trailing stops** work properly

Then move to real trading when ready! 