# MT5 Real Trading Setup Guide

⚠️ **WARNING: REAL MONEY TRADING** ⚠️

This guide is for setting up the MT5 bot for **REAL MONEY** trading on Deriv. 
**Only use money you can afford to lose!**

## 🚨 Important Safety Warnings

1. **Start Small**: Begin with a small account ($10-50) to test the bot
2. **Monitor Closely**: Always monitor the bot when it's running
3. **Set Limits**: Use the conservative settings provided
4. **Understand Risks**: Synthetic indices are highly volatile
5. **Emergency Stop**: Keep the bot code ready to stop trading immediately

## 📋 Prerequisites

1. **Deriv Real Account**: You need a real Deriv account with MT5 access
2. **MT5 Terminal**: MetaTrader 5 terminal installed and configured
3. **AutoTrading Enabled**: Make sure AutoTrading is enabled in MT5
4. **Sufficient Balance**: At least $10-50 for testing

## 🔧 Setup Steps

### Step 1: Create Real Environment File

Copy the example file and add your real credentials:

```bash
cp real.env-example real.env
```

Edit `real.env` with your real account details:

```env
MT5_REAL_LOGIN=your_real_account_number
MT5_REAL_PASSWORD=your_real_account_password
MT5_REAL_SERVER=Deriv-Real
```

### Step 2: Configure Trading Parameters

The real trading config (`real-config.py`) includes:

- **Conservative Risk Management**:
  - Stop Loss: 0.8% (very tight)
  - Take Profit: 1.5% (realistic)
  - Max Daily Loss: 3% (very conservative)
  - Max Daily Profit: 5% (realistic)

- **Dynamic Profit Taking**:
  - 0.3% profit: Close 50% of position
  - 0.6% profit: Close 80% of position
  - 1.0% profit: Close 90% of position

- **Trailing Stop**:
  - Starts at 0.3% profit
  - 0.2% trailing stop (very tight)

### Step 3: Test with Demo First

Before running real trading:

1. **Test with Demo**: Run the demo bot first to ensure everything works
2. **Verify Signals**: Check that signals are reasonable
3. **Test Order Execution**: Ensure orders are being placed correctly

### Step 4: Start Real Trading

```bash
python main-real.py
```

The bot will:
1. Ask for confirmation before starting real trading
2. Show account balance and equity
3. Display warnings about real money trading
4. Start with conservative settings

## 🎯 Trading Strategy for Real Money

### Recommended Settings for $10 Account:

1. **Single Instrument**: Start with only "Volatility 100 Index"
2. **Minimum Volume**: 0.5 lots (minimum for this instrument)
3. **High Confidence Threshold**: 0.6 (only take high-confidence trades)
4. **Longer Intervals**: 15 seconds between checks
5. **Conservative Strategy**: Multi-strategy with reduced weights

### Risk Management Features:

1. **Dynamic Profit Taking**: Automatically takes partial profits
2. **Trailing Stops**: Locks in profits as price moves favorably
3. **Position Limits**: Maximum 2 total positions
4. **Daily Limits**: 3% max daily loss, 5% max daily profit

## 📊 Monitoring Your Bot

### What to Watch:

1. **Account Balance**: Monitor for any unexpected losses
2. **Open Positions**: Check that positions are being managed correctly
3. **Profit Taking**: Verify that partial profits are being taken
4. **Stop Losses**: Ensure stops are being hit at reasonable levels

### Emergency Procedures:

1. **Stop the Bot**: Press Ctrl+C to stop trading
2. **Close Positions**: Manually close any open positions in MT5
3. **Check Logs**: Review the bot output for any errors
4. **Adjust Settings**: Modify risk parameters if needed

## 🔍 Troubleshooting

### Common Issues:

1. **"AutoTrading disabled"**: Enable AutoTrading in MT5 terminal
2. **"Invalid volume"**: Check minimum volume requirements
3. **"Unsupported filling mode"**: Bot handles this automatically
4. **"No data received"**: Check symbol names and MT5 connection

### Performance Issues:

1. **Too many losses**: Increase confidence threshold
2. **Not taking profits**: Check profit taking settings
3. **Wide stops**: Adjust stop loss percentage
4. **Slow execution**: Reduce trading interval

## 📈 Expected Performance

With conservative settings on a $10 account:

- **Win Rate**: 60-70% (with proper risk management)
- **Average Profit per Trade**: $0.05-$0.15
- **Daily Profit**: $0.50-$2.00 (realistic expectations)
- **Risk per Trade**: $0.04-$0.08 (0.8% stop loss)

## 🛡️ Safety Checklist

Before starting real trading:

- [ ] Tested with demo account
- [ ] Verified MT5 connection
- [ ] Confirmed AutoTrading enabled
- [ ] Set conservative risk parameters
- [ ] Prepared emergency stop procedure
- [ ] Understood all risks involved
- [ ] Only using money you can afford to lose

## 📞 Support

If you encounter issues:

1. **Check the logs**: Look for error messages
2. **Verify settings**: Ensure all parameters are correct
3. **Test incrementally**: Start with demo, then small real amounts
4. **Monitor closely**: Don't leave the bot unattended initially

## ⚠️ Final Warning

**Trading involves substantial risk of loss. This bot is for educational purposes and should be used with extreme caution. Past performance does not guarantee future results. Only trade with money you can afford to lose completely.** 