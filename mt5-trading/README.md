# MT5 Trading Bot - Organized Structure

## 📁 **Folder Structure**

```
mt5-trading/
├── demo-trading/          # 🟢 SAFE FOR TESTING
│   ├── main.py           # Demo bot
│   ├── config.py         # Demo config
│   ├── demo.env          # Demo credentials
│   ├── run-demo.py       # Demo launcher
│   └── README.md         # Demo instructions
│
├── real-trading/          # 🔴 REAL MONEY (gitignored)
│   ├── main-real.py      # Real trading bot
│   ├── real_config.py    # Real config
│   ├── real.env          # Real credentials
│   ├── main-micro.py     # Micro account bot
│   ├── micro-config.py   # Micro config
│   ├── large-account-config.py # Large account config
│   └── REAL_TRADING_README.md  # Real trading guide
│
├── strategies_mt5.py      # Trading strategies
├── requirements.txt       # Dependencies
└── README.md             # This file
```

## 🎯 **Quick Start**

### **For Testing (Demo):**
```bash
cd mt5-trading/demo-trading
py run-demo.py
```

### **For Real Trading:**
```bash
cd mt5-trading/real-trading
py main-real.py
```

## 🟢 **Demo Trading (Safe)**

The `demo-trading/` folder contains:
- **Demo bot** with all features
- **Demo account** credentials
- **Safe for testing** - no real money
- **Git committed** - can be shared

**Use for:**
- Testing strategies
- Validating risk management
- Experimenting with settings
- Learning the system

## 🔴 **Real Trading (Protected)**

The `real-trading/` folder contains:
- **Real trading bots** with balance-based sizing
- **Real account** credentials (gitignored)
- **Multiple configurations** for different account sizes
- **Advanced risk management**

**Use for:**
- Actual trading with real money
- Different account sizes (micro, standard, large)
- Production trading

## 🛡️ **Security Features**

### **Git Protection:**
- `real-trading/` folder is completely gitignored
- All real credentials are protected
- Demo folder is safe to commit

### **Account Types:**
- **Micro**: $10-100 accounts
- **Standard**: $100-1000 accounts  
- **Large**: $1000+ accounts

## 📊 **Configuration Files**

### **Demo:**
- `config.py` - Standard demo settings
- `demo-config.py` - Alternative demo settings

### **Real:**
- `real_config.py` - Standard real trading
- `micro-config.py` - For small accounts
- `large-account-config.py` - For large accounts

## 🚀 **Getting Started**

1. **Start with Demo:**
   ```bash
   cd mt5-trading/demo-trading
   py run-demo.py
   ```

2. **Test thoroughly** in demo environment

3. **When ready for real trading:**
   ```bash
   cd mt5-trading/real-trading
   # Configure your real.env file first
   py main-real.py
   ```

## ⚠️ **Important Notes**

- **Always test in demo first**
- **Real trading involves real money**
- **Use appropriate config for your account size**
- **Monitor closely when starting real trading**

## 🔧 **Dependencies**

Install required packages:
```bash
pip install -r requirements.txt
```

## 📞 **Support**

- **Demo issues**: Check demo-trading/README.md
- **Real trading**: Check real-trading/REAL_TRADING_README.md
- **Strategy questions**: Check strategies_mt5.py

## 🎯 **Recommended Workflow**

1. **Setup**: Install dependencies
2. **Demo Test**: Run demo bot and verify everything works
3. **Strategy Test**: Test different strategies in demo
4. **Risk Test**: Verify risk management in demo
5. **Real Setup**: Configure real account when ready
6. **Real Trading**: Start with small amounts

**Remember: Demo first, real money later!** 