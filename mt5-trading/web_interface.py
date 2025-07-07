#!/usr/bin/env python3
"""
MT5 Trading Bot Web Interface
==============================

A modern Flask web application to monitor and control the MT5 trading bot
from any device with a beautiful, responsive interface.
"""

from flask import Flask, render_template, jsonify, request, redirect, url_for, flash
import threading
import time
import json
import os
import sys
from datetime import datetime, timedelta
from dotenv import load_dotenv
import logging

# Add the current directory to Python path for imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Import MT5 trading bot components
try:
    import sys, os
    # Get absolute paths to bot directories
    current_dir = os.path.dirname(os.path.abspath(__file__))
    demo_path = os.path.join(current_dir, 'demo-trading')
    real_path = os.path.join(current_dir, 'real-trading')
    
    # Add to Python path if not already there
    if demo_path not in sys.path:
        sys.path.insert(0, demo_path)
    if real_path not in sys.path:
        sys.path.insert(0, real_path)
    
    print(f"[DEBUG] Demo path: {demo_path}")
    print(f"[DEBUG] Real path: {real_path}")
    print(f"[DEBUG] Demo exists: {os.path.exists(demo_path)}")
    print(f"[DEBUG] Real exists: {os.path.exists(real_path)}")
    
    # Import bot classes with detailed error reporting
    try:
        from main import MT5TradingBot as DemoMT5Bot
        print("[DEBUG] DemoMT5Bot imported successfully")
    except Exception as e:
        print(f"[IMPORT ERROR] Could not import DemoMT5Bot: {e}")
        DemoMT5Bot = None
        
    try:
        from main_real import MT5RealTradingBot as RealMT5Bot
        print("[DEBUG] RealMT5Bot imported successfully")
    except Exception as e:
        print(f"[IMPORT ERROR] Could not import RealMT5Bot: {e}")
        RealMT5Bot = None
        
    try:
        from main_micro import MT5MicroTradingBot as MicroMT5Bot
        print("[DEBUG] MicroMT5Bot imported successfully")
    except Exception as e:
        print(f"[IMPORT ERROR] Could not import MicroMT5Bot: {e}")
        MicroMT5Bot = None
        
    BOT_IMPORTS_AVAILABLE = all([DemoMT5Bot, RealMT5Bot, MicroMT5Bot])
    print(f"[DEBUG] All imports available: {BOT_IMPORTS_AVAILABLE}")
    
except Exception as e:
    print(f"[IMPORT ERROR] Unexpected error: {e}")
    BOT_IMPORTS_AVAILABLE = False

# Load environment variables
load_dotenv()

app = Flask(__name__)
app.secret_key = os.getenv('SECRET_KEY', 'your-secret-key-change-this')

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class MT5BotManager:
    """Manages the MT5 trading bot for web interface"""
    
    def __init__(self):
        self.bot = None
        self.bot_thread = None
        self.status = "stopped"
        self.bot_type = None
        self.config = {}
        self.stats = {
            'balance': 0,
            'equity': 0,
            'profit': 0,
            'open_positions': 0,
            'total_trades': 0,
            'win_rate': 0,
            'last_trade': None
        }
        self.errors = []
        self.logs = []
        self.last_update = datetime.now()
        
    def start_bot(self, bot_type='demo', config=None):
        """Start the trading bot in a separate thread"""
        if self.status == "running":
            return False, "Bot is already running"
        
        if not BOT_IMPORTS_AVAILABLE:
            return False, "Bot modules not available. Check imports."
        
        try:
            self.bot_type = bot_type
            self.config = config or {}
            
            # Create bot instance based on type
            if bot_type == 'demo':
                self.bot = DemoMT5Bot()
            elif bot_type == 'real':
                self.bot = RealMT5Bot()
            elif bot_type == 'micro':
                self.bot = MicroMT5Bot()
            else:
                return False, f"Unknown bot type: {bot_type}"
            
            self.status = "starting"
            self.logs.append(f"Starting {bot_type} bot...")
            
            # Start bot in separate thread
            self.bot_thread = threading.Thread(target=self._run_bot)
            self.bot_thread.daemon = True
            self.bot_thread.start()
            
            return True, f"{bot_type.capitalize()} bot started successfully"
            
        except Exception as e:
            error_msg = f"Failed to start {bot_type} bot: {str(e)}"
            self.errors.append(error_msg)
            self.status = "error"
            return False, error_msg
    
    def stop_bot(self):
        """Stop the trading bot"""
        if self.bot:
            try:
                self.bot.stop()
                self.status = "stopping"
                self.logs.append("Stopping bot...")
                return True, "Bot stopping..."
            except Exception as e:
                error_msg = f"Error stopping bot: {str(e)}"
                self.errors.append(error_msg)
                return False, error_msg
        return False, "No bot running"
    
    def _run_bot(self):
        """Run the bot (called in separate thread)"""
        try:
            self.status = "running"
            self.logs.append("Bot is now running")
            self.bot.start()
        except Exception as e:
            error_msg = f"Bot error: {str(e)}"
            self.errors.append(error_msg)
            self.status = "error"
            self.logs.append(f"Error: {error_msg}")
        finally:
            self.status = "stopped"
            self.logs.append("Bot stopped")
    
    def get_status(self):
        """Get current bot status and statistics"""
        # Update stats if bot is available
        if self.bot and hasattr(self.bot, 'get_account_info'):
            try:
                account_info = self.bot.get_account_info()
                if account_info:
                    self.stats.update(account_info)
            except Exception as e:
                self.logs.append(f"Error getting account info: {str(e)}")
        
        return {
            'status': self.status,
            'bot_type': self.bot_type,
            'stats': self.stats,
            'config': self.config,
            'errors': self.errors[-5:],  # Last 5 errors
            'logs': self.logs[-10:],     # Last 10 logs
            'last_update': self.last_update.strftime('%Y-%m-%d %H:%M:%S')
        }
    
    def place_trade(self, symbol, order_type, volume, price=None, sl=None, tp=None):
        """Place a trade manually"""
        if not self.bot or self.status != "running":
            return False, "Bot is not running"
        
        try:
            result = self.bot.place_trade(symbol, order_type, volume, price, sl, tp)
            self.logs.append(f"Manual trade placed: {symbol} {order_type} {volume}")
            return True, f"Trade placed successfully: {result}"
        except Exception as e:
            error_msg = f"Failed to place trade: {str(e)}"
            self.errors.append(error_msg)
            return False, error_msg
    
    def close_position(self, ticket):
        """Close a specific position"""
        if not self.bot or self.status != "running":
            return False, "Bot is not running"
        
        try:
            result = self.bot.close_position(ticket)
            self.logs.append(f"Position closed: {ticket}")
            return True, f"Position closed successfully: {result}"
        except Exception as e:
            error_msg = f"Failed to close position: {str(e)}"
            self.errors.append(error_msg)
            return False, error_msg

# Create bot manager instance
bot_manager = MT5BotManager()

@app.route('/')
def index():
    """Main dashboard page"""
    return render_template('dashboard.html')

@app.route('/api/status')
def get_status():
    """API endpoint to get bot status"""
    return jsonify(bot_manager.get_status())

@app.route('/api/start', methods=['POST'])
def start_bot():
    """API endpoint to start the bot"""
    data = request.get_json() or {}
    bot_type = data.get('bot_type', 'demo')
    config = data.get('config', {})
    
    success, message = bot_manager.start_bot(bot_type, config)
    return jsonify({'success': success, 'message': message})

@app.route('/api/stop', methods=['POST'])
def stop_bot():
    """API endpoint to stop the bot"""
    success, message = bot_manager.stop_bot()
    return jsonify({'success': success, 'message': message})

@app.route('/api/trade', methods=['POST'])
def place_trade():
    """API endpoint to place a manual trade"""
    data = request.get_json() or {}
    symbol = data.get('symbol')
    order_type = data.get('order_type')
    volume = data.get('volume')
    price = data.get('price')
    sl = data.get('sl')
    tp = data.get('tp')
    
    if not all([symbol, order_type, volume]):
        return jsonify({'success': False, 'message': 'Missing required parameters'})
    
    success, message = bot_manager.place_trade(symbol, order_type, volume, price, sl, tp)
    return jsonify({'success': success, 'message': message})

@app.route('/api/close_position', methods=['POST'])
def close_position():
    """API endpoint to close a position"""
    data = request.get_json() or {}
    ticket = data.get('ticket')
    
    if not ticket:
        return jsonify({'success': False, 'message': 'Missing ticket parameter'})
    
    success, message = bot_manager.close_position(ticket)
    return jsonify({'success': success, 'message': message})

@app.route('/api/config')
def get_config():
    """Get available configuration options"""
    return jsonify({
        'bot_types': [
            {'id': 'demo', 'name': 'Demo Trading Bot', 'description': 'Safe demo account trading'},
            {'id': 'real', 'name': 'Real Trading Bot', 'description': 'Real account trading with risk management'},
            {'id': 'micro', 'name': 'Micro Trading Bot', 'description': 'Very small balance trading ($0.28-$1.00)'}
        ],
        'symbols': [
            'Volatility 100 Index', 'Volatility 75 Index', 'Volatility 50 Index',
            'Volatility 25 Index', 'Volatility 10 Index', 'Volatility 150 Index',
            'Volatility 200 Index', 'BOOM 1000 Index', 'CRASH 1000 Index',
            'BOOM 500 Index', 'CRASH 500 Index', 'STEP Index',
            'Bear Market Index', 'Bull Market Index'
        ],
        'order_types': [
            {'id': 'BUY', 'name': 'Buy'},
            {'id': 'SELL', 'name': 'Sell'}
        ]
    })

def create_templates():
    """Create the HTML templates for the web interface"""
    os.makedirs('templates', exist_ok=True)
    
    # Create dashboard template
    dashboard_html = '''<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>MT5 Trading Bot - Web Interface</title>
    <link href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.0.0/css/all.min.css" rel="stylesheet">
    <style>
        * {
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }
        
        body {
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            min-height: 100vh;
            padding: 20px;
        }
        
        .container {
            max-width: 1200px;
            margin: 0 auto;
            background: white;
            border-radius: 20px;
            box-shadow: 0 20px 40px rgba(0,0,0,0.1);
            overflow: hidden;
        }
        
        .header {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 30px;
            text-align: center;
        }
        
        .header h1 {
            font-size: 2.5rem;
            margin-bottom: 10px;
        }
        
        .header p {
            font-size: 1.1rem;
            opacity: 0.9;
        }
        
        .content {
            padding: 30px;
        }
        
        .status-card {
            background: #f8f9fa;
            border-radius: 15px;
            padding: 25px;
            margin-bottom: 30px;
            border-left: 5px solid #007bff;
        }
        
        .status-indicator {
            display: inline-block;
            width: 12px;
            height: 12px;
            border-radius: 50%;
            margin-right: 10px;
        }
        
        .status-running { background: #28a745; }
        .status-stopped { background: #dc3545; }
        .status-starting { background: #ffc107; }
        .status-stopping { background: #fd7e14; }
        .status-error { background: #dc3545; }
        
        .control-panel {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
            gap: 20px;
            margin-bottom: 30px;
        }
        
        .panel {
            background: white;
            border-radius: 15px;
            padding: 25px;
            box-shadow: 0 5px 15px rgba(0,0,0,0.08);
            border: 1px solid #e9ecef;
        }
        
        .panel h3 {
            color: #333;
            margin-bottom: 20px;
            font-size: 1.3rem;
        }
        
        .btn {
            padding: 12px 24px;
            border: none;
            border-radius: 8px;
            font-size: 1rem;
            font-weight: 600;
            cursor: pointer;
            transition: all 0.3s ease;
            margin: 5px;
        }
        
        .btn-primary {
            background: #007bff;
            color: white;
        }
        
        .btn-primary:hover {
            background: #0056b3;
            transform: translateY(-2px);
        }
        
        .btn-success {
            background: #28a745;
            color: white;
        }
        
        .btn-success:hover {
            background: #1e7e34;
            transform: translateY(-2px);
        }
        
        .btn-danger {
            background: #dc3545;
            color: white;
        }
        
        .btn-danger:hover {
            background: #c82333;
            transform: translateY(-2px);
        }
        
        .btn-warning {
            background: #ffc107;
            color: #212529;
        }
        
        .btn-warning:hover {
            background: #e0a800;
            transform: translateY(-2px);
        }
        
        .form-group {
            margin-bottom: 20px;
        }
        
        .form-group label {
            display: block;
            margin-bottom: 8px;
            font-weight: 600;
            color: #333;
        }
        
        .form-control {
            width: 100%;
            padding: 12px;
            border: 2px solid #e9ecef;
            border-radius: 8px;
            font-size: 1rem;
            transition: border-color 0.3s ease;
        }
        
        .form-control:focus {
            outline: none;
            border-color: #007bff;
        }
        
        .stats-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 20px;
            margin-bottom: 30px;
        }
        
        .stat-card {
            background: white;
            border-radius: 15px;
            padding: 20px;
            text-align: center;
            box-shadow: 0 5px 15px rgba(0,0,0,0.08);
            border: 1px solid #e9ecef;
        }
        
        .stat-value {
            font-size: 2rem;
            font-weight: bold;
            color: #007bff;
            margin-bottom: 5px;
        }
        
        .stat-label {
            color: #666;
            font-size: 0.9rem;
        }
        
        .logs-panel {
            background: #f8f9fa;
            border-radius: 15px;
            padding: 20px;
            max-height: 300px;
            overflow-y: auto;
        }
        
        .log-entry {
            padding: 8px 0;
            border-bottom: 1px solid #e9ecef;
            font-family: 'Courier New', monospace;
            font-size: 0.9rem;
        }
        
        .log-entry:last-child {
            border-bottom: none;
        }
        
        .error-log {
            color: #dc3545;
        }
        
        .success-log {
            color: #28a745;
        }
        
        .info-log {
            color: #007bff;
        }
        
        .alert {
            padding: 15px;
            border-radius: 8px;
            margin-bottom: 20px;
            border-left: 4px solid;
        }
        
        .alert-success {
            background: #d4edda;
            border-color: #28a745;
            color: #155724;
        }
        
        .alert-danger {
            background: #f8d7da;
            border-color: #dc3545;
            color: #721c24;
        }
        
        .alert-warning {
            background: #fff3cd;
            border-color: #ffc107;
            color: #856404;
        }
        
        .hidden {
            display: none;
        }
        
        @media (max-width: 768px) {
            .container {
                margin: 10px;
                border-radius: 15px;
            }
            
            .content {
                padding: 20px;
            }
            
            .control-panel {
                grid-template-columns: 1fr;
            }
            
            .stats-grid {
                grid-template-columns: repeat(2, 1fr);
            }
        }
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1><i class="fas fa-chart-line"></i> MT5 Trading Bot</h1>
            <p>Web Interface for MetaTrader 5 Trading Bot</p>
        </div>
        
        <div class="content">
            <!-- Status Card -->
            <div class="status-card">
                <h3>
                    <span class="status-indicator" id="statusIndicator"></span>
                    Bot Status: <span id="botStatus">Loading...</span>
                </h3>
                <p id="botType">Bot Type: -</p>
                <p id="lastUpdate">Last Update: -</p>
            </div>
            
            <!-- Alert Messages -->
            <div id="alertContainer"></div>
            
            <!-- Control Panel -->
            <div class="control-panel">
                <!-- Bot Control -->
                <div class="panel">
                    <h3><i class="fas fa-cogs"></i> Bot Control</h3>
                    <div class="form-group">
                        <label for="botTypeSelect">Bot Type:</label>
                        <select id="botTypeSelect" class="form-control">
                            <option value="demo">Demo Trading Bot</option>
                            <option value="real">Real Trading Bot</option>
                            <option value="micro">Micro Trading Bot</option>
                        </select>
                    </div>
                    <button id="startBtn" class="btn btn-success">
                        <i class="fas fa-play"></i> Start Bot
                    </button>
                    <button id="stopBtn" class="btn btn-danger">
                        <i class="fas fa-stop"></i> Stop Bot
                    </button>
                </div>
                
                <!-- Manual Trading -->
                <div class="panel">
                    <h3><i class="fas fa-hand-holding-usd"></i> Manual Trading</h3>
                    <div class="form-group">
                        <label for="symbolSelect">Symbol:</label>
                        <select id="symbolSelect" class="form-control">
                            <option value="Volatility 100 Index">Volatility 100 Index</option>
                            <option value="Volatility 75 Index">Volatility 75 Index</option>
                            <option value="Volatility 50 Index">Volatility 50 Index</option>
                        </select>
                    </div>
                    <div class="form-group">
                        <label for="orderTypeSelect">Order Type:</label>
                        <select id="orderTypeSelect" class="form-control">
                            <option value="BUY">Buy</option>
                            <option value="SELL">Sell</option>
                        </select>
                    </div>
                    <div class="form-group">
                        <label for="volumeInput">Volume:</label>
                        <input type="number" id="volumeInput" class="form-control" value="0.01" step="0.01" min="0.01">
                    </div>
                    <button id="placeTradeBtn" class="btn btn-primary">
                        <i class="fas fa-plus"></i> Place Trade
                    </button>
                </div>
            </div>
            
            <!-- Statistics -->
            <div class="stats-grid">
                <div class="stat-card">
                    <div class="stat-value" id="balanceValue">$0.00</div>
                    <div class="stat-label">Balance</div>
                </div>
                <div class="stat-card">
                    <div class="stat-value" id="equityValue">$0.00</div>
                    <div class="stat-label">Equity</div>
                </div>
                <div class="stat-card">
                    <div class="stat-value" id="profitValue">$0.00</div>
                    <div class="stat-label">Profit/Loss</div>
                </div>
                <div class="stat-card">
                    <div class="stat-value" id="positionsValue">0</div>
                    <div class="stat-label">Open Positions</div>
                </div>
                <div class="stat-card">
                    <div class="stat-value" id="tradesValue">0</div>
                    <div class="stat-label">Total Trades</div>
                </div>
                <div class="stat-card">
                    <div class="stat-value" id="winRateValue">0%</div>
                    <div class="stat-label">Win Rate</div>
                </div>
            </div>
            
            <!-- Logs -->
            <div class="panel">
                <h3><i class="fas fa-list"></i> Bot Logs</h3>
                <div class="logs-panel" id="logsContainer">
                    <div class="log-entry info-log">Web interface started...</div>
                </div>
            </div>
        </div>
    </div>
    
    <script>
        // Global variables
        let currentStatus = 'stopped';
        let updateInterval;
        
        // Initialize the interface
        document.addEventListener('DOMContentLoaded', function() {
            loadConfig();
            updateStatus();
            startAutoUpdate();
        });
        
        // Load configuration
        async function loadConfig() {
            try {
                const response = await fetch('/api/config');
                const config = await response.json();
                
                // Populate symbol select
                const symbolSelect = document.getElementById('symbolSelect');
                symbolSelect.innerHTML = '';
                config.symbols.forEach(symbol => {
                    const option = document.createElement('option');
                    option.value = symbol;
                    option.textContent = symbol;
                    symbolSelect.appendChild(option);
                });
                
            } catch (error) {
                console.error('Error loading config:', error);
            }
        }
        
        // Update bot status
        async function updateStatus() {
            try {
                const response = await fetch('/api/status');
                const data = await response.json();
                
                // Update status indicator
                const statusIndicator = document.getElementById('statusIndicator');
                const botStatus = document.getElementById('botStatus');
                const botType = document.getElementById('botType');
                const lastUpdate = document.getElementById('lastUpdate');
                
                statusIndicator.className = `status-indicator status-${data.status}`;
                botStatus.textContent = data.status.charAt(0).toUpperCase() + data.status.slice(1);
                botType.textContent = `Bot Type: ${data.bot_type || '-'}`;
                lastUpdate.textContent = `Last Update: ${data.last_update}`;
                
                currentStatus = data.status;
                
                // Update statistics
                if (data.stats) {
                    document.getElementById('balanceValue').textContent = `$${data.stats.balance.toFixed(2)}`;
                    document.getElementById('equityValue').textContent = `$${data.stats.equity.toFixed(2)}`;
                    document.getElementById('profitValue').textContent = `$${data.stats.profit.toFixed(2)}`;
                    document.getElementById('positionsValue').textContent = data.stats.open_positions;
                    document.getElementById('tradesValue').textContent = data.stats.total_trades;
                    document.getElementById('winRateValue').textContent = `${data.stats.win_rate.toFixed(1)}%`;
                }
                
                // Update logs
                updateLogs(data.logs || []);
                
            } catch (error) {
                console.error('Error updating status:', error);
            }
        }
        
        // Update logs display
        function updateLogs(logs) {
            const logsContainer = document.getElementById('logsContainer');
            logsContainer.innerHTML = '';
            
            logs.forEach(log => {
                const logEntry = document.createElement('div');
                logEntry.className = 'log-entry info-log';
                logEntry.textContent = log;
                logsContainer.appendChild(logEntry);
            });
            
            // Scroll to bottom
            logsContainer.scrollTop = logsContainer.scrollHeight;
        }
        
        // Start auto-update
        function startAutoUpdate() {
            updateInterval = setInterval(updateStatus, 2000); // Update every 2 seconds
        }
        
        // Show alert message
        function showAlert(message, type = 'info') {
            const alertContainer = document.getElementById('alertContainer');
            const alert = document.createElement('div');
            alert.className = `alert alert-${type}`;
            alert.textContent = message;
            
            alertContainer.appendChild(alert);
            
            // Remove alert after 5 seconds
            setTimeout(() => {
                alert.remove();
            }, 5000);
        }
        
        // Event listeners
        document.getElementById('startBtn').addEventListener('click', async function() {
            const botType = document.getElementById('botTypeSelect').value;
            
            try {
                const response = await fetch('/api/start', {
                    method: 'POST',
                    headers: {
                        'Content-Type': 'application/json'
                    },
                    body: JSON.stringify({
                        bot_type: botType,
                        config: {}
                    })
                });
                
                const result = await response.json();
                
                if (result.success) {
                    showAlert(result.message, 'success');
                } else {
                    showAlert(result.message, 'danger');
                }
                
            } catch (error) {
                showAlert('Error starting bot: ' + error.message, 'danger');
            }
        });
        
        document.getElementById('stopBtn').addEventListener('click', async function() {
            try {
                const response = await fetch('/api/stop', {
                    method: 'POST',
                    headers: {
                        'Content-Type': 'application/json'
                    }
                });
                
                const result = await response.json();
                
                if (result.success) {
                    showAlert(result.message, 'success');
                } else {
                    showAlert(result.message, 'danger');
                }
                
            } catch (error) {
                showAlert('Error stopping bot: ' + error.message, 'danger');
            }
        });
        
        document.getElementById('placeTradeBtn').addEventListener('click', async function() {
            const symbol = document.getElementById('symbolSelect').value;
            const orderType = document.getElementById('orderTypeSelect').value;
            const volume = parseFloat(document.getElementById('volumeInput').value);
            
            if (!symbol || !orderType || !volume) {
                showAlert('Please fill in all fields', 'warning');
                return;
            }
            
            try {
                const response = await fetch('/api/trade', {
                    method: 'POST',
                    headers: {
                        'Content-Type': 'application/json'
                    },
                    body: JSON.stringify({
                        symbol: symbol,
                        order_type: orderType,
                        volume: volume
                    })
                });
                
                const result = await response.json();
                
                if (result.success) {
                    showAlert(result.message, 'success');
                } else {
                    showAlert(result.message, 'danger');
                }
                
            } catch (error) {
                showAlert('Error placing trade: ' + error.message, 'danger');
            }
        });
    </script>
</body>
</html>'''
    
    with open('templates/dashboard.html', 'w', encoding='utf-8') as f:
        f.write(dashboard_html)

if __name__ == '__main__':
    # Create templates before running the app
    create_templates()
    
    print("🌐 Starting MT5 Trading Bot Web Interface...")
    print("📱 Access from your phone at: http://[YOUR_PC_IP]:5000")
    print("💻 Access from your PC at: http://localhost:5000")
    print("🔧 Make sure your phone and PC are on the same WiFi network!")
    
    # Run the Flask app
    app.run(host='0.0.0.0', port=5000, debug=False) 