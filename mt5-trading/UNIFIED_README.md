# MT5 Trading Bot - Unified System

## 🌟 **Complete Trading Solution for Small Accounts ($10+)**

A comprehensive MetaTrader 5 trading bot system with web interface, optimized for small account scaling from $10 to $100+ systematically.

## 🚀 **Quick Start**

### **1. Start the Unified Web Interface**
```bash
cd mt5-trading
python start_unified_web.py
```

### **2. Access the Dashboard**
- **Local**: `http://localhost:5000`
- **Phone**: `http://[your-local-ip]:5000` (same WiFi network)

### **3. Connect and Trade**
1. Click "Connect MT5" in the dashboard
2. Choose your bot type (Demo/Real/Micro)
3. Start trading with real-time monitoring

## 📊 **Bot Types**

### **1. Demo Bot**
- **Purpose**: Safe testing and strategy validation
- **Account**: Demo account
- **Risk**: No real money
- **Best for**: Learning and testing strategies

### **2. Real Bot**
- **Purpose**: Standard real account trading
- **Account**: Real account with $700+ balance
- **Risk**: Conservative risk management
- **Best for**: Established accounts

### **3. Micro Bot** ⭐ **RECOMMENDED FOR $10+**
- **Purpose**: Small account scaling ($10 to $100+)
- **Account**: Real account with $10+ balance
- **Strategy**: Progressive volume scaling
- **Features**:
  - Starts with minimum volume (0.01)
  - Scales volume based on profit
  - Tight stop losses (0.5%)
  - Progressive profit taking
  - Balance-based risk management

## 🎯 **Scaling Strategy for Small Accounts**

### **Volume Scaling Rules**
```
Starting Balance: $10
Initial Volume: 0.01

Scaling Levels:
- 0.5% profit → Volume +20%
- 1.0% profit → Volume +50%
- 2.0% profit → Volume +100% (double)
```

### **Profit Taking Levels**
```
- 0.3% profit → Close 50% of position
- 0.6% profit → Close 70% of position
- 0.9% profit → Close 90% of position
```

### **Risk Management**
- **Stop Loss**: 0.5% (tight protection)
- **Take Profit**: 1.0% (realistic scaling)
- **Daily Loss Limit**: 2% (protect capital)
- **Daily Profit Limit**: 5% (realistic growth)

## 📈 **Scaling Targets**

| Level | Balance Target | Growth |
|-------|----------------|---------|
| 0 | $10 | Starting |
| 1 | $20 | 2x |
| 2 | $40 | 4x |
| 3 | $80 | 8x |
| Max | $100+ | 10x+ |

## 🖥️ **Web Interface Features**

### **Account Overview**
- Real-time balance and equity
- Profit/Loss tracking
- Win rate calculation
- Total trades counter
- Margin level monitoring

### **Bot Control**
- Start/Stop individual bots
- Real-time bot status
- Emergency stop all bots
- Bot performance monitoring

### **Manual Trading**
- Place manual trades
- Close individual positions
- Close all positions
- Real-time position monitoring

### **Real-time Monitoring**
- Live account updates
- Position tracking
- Profit/Loss updates
- Trading history

## 🔧 **Installation & Setup**

### **Requirements**
- Python 3.7+
- MetaTrader 5 terminal
- Deriv account (demo or real)
- Internet connection

### **Dependencies**
```bash
pip install flask python-dotenv MetaTrader5 pandas numpy
```

### **Configuration**
1. **Demo Trading**: No setup required
2. **Real Trading**: Update `real-trading/real.env` with your credentials
3. **Micro Trading**: Uses same credentials as real trading

## 📁 **File Structure**

```
mt5-trading/
├── start_unified_web.py          # Main start script
├── unified_web_interface.py      # Web interface
├── bot_manager.py                # Bot process manager
├── real-trading/
│   ├── main-real.py             # Real account bot
│   ├── main-micro-scaling.py    # Micro scaling bot ⭐
│   ├── real_config.py           # Real account config
│   ├── micro_scaling_config.py  # Micro scaling config ⭐
│   └── real.env                 # Account credentials
├── demo-trading/
│   └── main.py                  # Demo account bot
├── strategies_mt5.py            # Trading strategies
└── templates/
    └── unified_dashboard.html   # Web dashboard
```

## 🎮 **Usage Guide**

### **For Small Accounts ($10+)**

1. **Start the System**
   ```bash
   cd mt5-trading
   python start_unified_web.py
   ```

2. **Access Dashboard**
   - Open browser to `http://localhost:5000`

3. **Connect to MT5**
   - Click "Connect MT5" button
   - Ensure MT5 is running with AutoTrading enabled

4. **Start Micro Bot**
   - Click "Start" on the Micro Bot card
   - Bot will begin with minimum volume (0.01)

5. **Monitor Progress**
   - Watch balance grow from $10 to $20, $40, $80, $100+
   - Volume automatically scales with profits
   - Positions are managed automatically

### **For Larger Accounts ($700+)**

1. **Use Real Bot** instead of Micro Bot
2. **Higher volumes** and more aggressive strategies
3. **Multiple positions** allowed
4. **More complex strategies** available

## ⚙️ **Configuration Options**

### **Environment Variables** (`real.env`)
```env
MT5_REAL_LOGIN=your_login
MT5_REAL_PASSWORD=your_password
MT5_REAL_SERVER=DerivVU-Server-03
MT5_TIMEFRAME=M5
MT5_VOLUME=0.01
MT5_CONFIDENCE_THRESHOLD=0.7
MT5_TRADING_INTERVAL=20
MT5_SYMBOLS=Volatility 100 Index
MT5_STRATEGY=scaling_rsi
```

### **Scaling Configuration**
- Modify `micro_scaling_config.py` for custom scaling rules
- Adjust volume multipliers and profit thresholds
- Change risk management parameters

## 🔒 **Safety Features**

### **Risk Management**
- ✅ Tight stop losses (0.5%)
- ✅ Progressive profit taking
- ✅ Daily loss limits (2%)
- ✅ Balance-based position sizing
- ✅ Emergency stop functionality

### **Account Protection**
- ✅ Minimum volume requirements
- ✅ Margin level monitoring
- ✅ Consecutive loss tracking
- ✅ Session-based limits

## 📊 **Performance Tracking**

### **Real-time Metrics**
- Account balance and equity
- Profit/Loss per trade
- Win rate percentage
- Total trades count
- Daily profit/loss

### **Scaling Progress**
- Current scaling level
- Next target balance
- Volume scaling status
- Profit taking levels reached

## 🚨 **Important Notes**

### **Before Starting**
1. **Ensure MT5 is running** with AutoTrading enabled
2. **Verify account credentials** in `real.env`
3. **Check minimum balance** requirements
4. **Test with demo first** if unsure

### **Risk Warnings**
- ⚠️ **Real money trading** - only trade what you can afford to lose
- ⚠️ **Small accounts** are more volatile - expect ups and downs
- ⚠️ **Scaling takes time** - be patient with the process
- ⚠️ **Monitor regularly** - check dashboard frequently

### **Best Practices**
- ✅ Start with Micro Bot for small accounts
- ✅ Monitor the dashboard regularly
- ✅ Don't interfere with running bots
- ✅ Use emergency stop if needed
- ✅ Keep MT5 running and connected

## 🆘 **Troubleshooting**

### **Common Issues**

**MT5 Connection Failed**
- Ensure MT5 is installed and running
- Check credentials in `real.env`
- Verify server name is correct

**Bot Won't Start**
- Check if MT5 is connected
- Verify account has sufficient balance
- Check for error messages in console

**Web Interface Not Loading**
- Ensure port 5000 is not in use
- Check firewall settings
- Try accessing via localhost instead of IP

**Trading Issues**
- Verify AutoTrading is enabled in MT5
- Check minimum volume requirements
- Ensure sufficient margin

### **Support**
- Check console output for error messages
- Verify all dependencies are installed
- Ensure configuration files are correct
- Test with demo account first

## 🎯 **Success Tips**

### **For Small Account Scaling**
1. **Start with Micro Bot** - designed for your account size
2. **Be patient** - scaling takes time and consistency
3. **Don't interfere** - let the bot work automatically
4. **Monitor regularly** - check dashboard daily
5. **Stay consistent** - avoid manual interventions

### **For Best Results**
- Use Volatility 100 Index (most reliable)
- Keep MT5 running 24/7
- Don't panic during drawdowns
- Trust the scaling process
- Focus on long-term growth

## 📈 **Expected Results**

### **Small Account ($10+) Timeline**
- **Week 1-2**: $10 → $12-15 (establishing)
- **Week 3-4**: $15 → $20-25 (first scaling)
- **Month 2**: $25 → $40-50 (second scaling)
- **Month 3**: $50 → $80-100 (third scaling)

### **Success Factors**
- Consistent bot operation
- Proper risk management
- Market volatility (Volatility 100 Index)
- No manual interference
- Patience with the process

---

**🎉 Ready to start scaling your account? Run `python start_unified_web.py` and begin your journey from $10 to $100+!** 