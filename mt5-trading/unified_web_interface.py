#!/usr/bin/env python3
"""
Unified MT5 Trading Bot Web Interface
=====================================

A comprehensive Flask web application for managing MT5 trading bots.
Optimized for small accounts ($10+) with proper scaling strategies.
"""

from flask import Flask, render_template, jsonify, request, redirect, url_for
import os
import sys
import subprocess
import threading
import time
import json
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
import MetaTrader5 as mt5
import pandas as pd
import numpy as np

# Add current directory to path for imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from bot_manager import bot_manager

app = Flask(__name__)

# Global variables for real-time data
account_data = {
    'balance': 0.0,
    'equity': 0.0,
    'profit': 0.0,
    'margin': 0.0,
    'free_margin': 0.0,
    'margin_level': 0.0,
    'open_positions': [],
    'total_trades': 0,
    'win_rate': 0.0,
    'total_pnl': 0.0
}

trading_history = []
bot_status = {
    'demo': 'stopped',
    'real': 'stopped', 
    'micro': 'stopped'
}

class MT5Connection:
    """Handles MT5 connection and data retrieval"""
    
    def __init__(self):
        self.connected = False
        self.account_info = None
        
    def connect(self, account_type='real'):
        """Connect to MT5 account"""
        try:
            if not mt5.initialize():
                return False, "MT5 initialization failed"
            
            # Load credentials based on account type
            if account_type == 'real':
                from real_trading.real_config import MT5RealConfig
                login = MT5RealConfig.MT5_LOGIN
                password = MT5RealConfig.MT5_PASSWORD
                server = MT5RealConfig.MT5_SERVER
            elif account_type == 'micro':
                from real_trading.micro_config import MT5MicroConfig
                login = MT5MicroConfig.MT5_LOGIN
                password = MT5MicroConfig.MT5_PASSWORD
                server = MT5MicroConfig.MT5_SERVER
            else:
                return False, "Invalid account type"
            
            if not mt5.login(login=login, password=password, server=server):
                return False, f"MT5 login failed: {mt5.last_error()}"
            
            self.connected = True
            self.account_info = mt5.account_info()
            return True, "Connected successfully"
            
        except Exception as e:
            return False, f"Connection error: {str(e)}"
    
    def disconnect(self):
        """Disconnect from MT5"""
        if self.connected:
            mt5.shutdown()
            self.connected = False
    
    def get_account_data(self) -> Dict:
        """Get current account data"""
        if not self.connected:
            return account_data
        
        try:
            account_info = mt5.account_info()
            if not account_info:
                return account_data
            
            # Get open positions
            positions = mt5.positions_get()
            open_positions = []
            
            if positions:
                for pos in positions:
                    open_positions.append({
                        'ticket': pos.ticket,
                        'symbol': pos.symbol,
                        'type': 'BUY' if pos.type == 0 else 'SELL',
                        'volume': pos.volume,
                        'price_open': pos.price_open,
                        'price_current': pos.price_current,
                        'profit': pos.profit,
                        'swap': pos.swap,
                        'time': datetime.fromtimestamp(pos.time).strftime('%H:%M:%S')
                    })
            
            # Calculate win rate from history
            history = mt5.history_deals_get(datetime.now() - timedelta(days=30), datetime.now())
            total_trades = len(history) if history else 0
            winning_trades = len([d for d in history if d.profit > 0]) if history else 0
            win_rate = (winning_trades / total_trades * 100) if total_trades > 0 else 0
            
            # Calculate total PnL
            total_pnl = sum([d.profit for d in history]) if history else 0
            
            return {
                'balance': account_info.balance,
                'equity': account_info.equity,
                'profit': account_info.profit,
                'margin': account_info.margin,
                'free_margin': account_info.margin_free,
                'margin_level': account_info.margin_level,
                'open_positions': open_positions,
                'total_trades': total_trades,
                'win_rate': win_rate,
                'total_pnl': total_pnl
            }
            
        except Exception as e:
            print(f"Error getting account data: {e}")
            return account_data

# Global MT5 connection
mt5_conn = MT5Connection()

@app.route('/')
def index():
    """Main dashboard page"""
    return render_template('unified_dashboard.html')

@app.route('/api/status')
def get_status():
    """Get comprehensive status including bot status and account data"""
    # Update bot status
    bot_status.update(bot_manager.get_status()['bots'])
    
    # Get account data
    account_data.update(mt5_conn.get_account_data())
    
    return jsonify({
        'bots': bot_status,
        'account': account_data,
        'last_update': datetime.now().strftime('%H:%M:%S')
    })

@app.route('/api/start_bot', methods=['POST'])
def start_bot():
    """Start a specific bot"""
    data = request.get_json()
    bot_type = data.get('bot_type', 'demo')
    
    success, message = bot_manager.start_bot(bot_type)
    if success:
        bot_status[bot_type] = 'running'
    
    return jsonify({'success': success, 'message': message})

@app.route('/api/stop_bot', methods=['POST'])
def stop_bot():
    """Stop a specific bot"""
    data = request.get_json()
    bot_type = data.get('bot_type', 'demo')
    
    success, message = bot_manager.stop_bot(bot_type)
    if success:
        bot_status[bot_type] = 'stopped'
    
    return jsonify({'success': success, 'message': message})

@app.route('/api/stop_all', methods=['POST'])
def stop_all_bots():
    """Stop all bots"""
    success, message = bot_manager.stop_all_bots()
    for bot_type in bot_status:
        bot_status[bot_type] = 'stopped'
    
    return jsonify({'success': success, 'message': message})

@app.route('/api/connect_mt5', methods=['POST'])
def connect_mt5():
    """Connect to MT5 account"""
    data = request.get_json()
    account_type = data.get('account_type', 'real')
    
    success, message = mt5_conn.connect(account_type)
    return jsonify({'success': success, 'message': message})

@app.route('/api/disconnect_mt5', methods=['POST'])
def disconnect_mt5():
    """Disconnect from MT5"""
    mt5_conn.disconnect()
    return jsonify({'success': True, 'message': 'Disconnected from MT5'})

@app.route('/api/place_trade', methods=['POST'])
def place_trade():
    """Place a manual trade"""
    if not mt5_conn.connected:
        return jsonify({'success': False, 'message': 'Not connected to MT5'})
    
    data = request.get_json()
    symbol = data.get('symbol', 'Volatility 100 Index')
    order_type = data.get('type', 'BUY')
    volume = float(data.get('volume', 0.01))
    price = float(data.get('price', 0))
    
    try:
        # Convert order type
        mt5_order_type = mt5.ORDER_TYPE_BUY if order_type == 'BUY' else mt5.ORDER_TYPE_SELL
        
        # Place order
        request_dict = {
            "action": mt5.TRADE_ACTION_DEAL,
            "symbol": symbol,
            "volume": volume,
            "type": mt5_order_type,
            "price": price,
            "deviation": 20,
            "magic": 234000,
            "comment": "Manual trade from web interface",
            "type_time": mt5.ORDER_TIME_GTC,
            "type_filling": mt5.ORDER_FILLING_IOC,
        }
        
        result = mt5.order_send(request_dict)
        
        if result.retcode != mt5.TRADE_RETCODE_DONE:
            return jsonify({'success': False, 'message': f'Order failed: {result.comment}'})
        
        return jsonify({'success': True, 'message': f'{order_type} order placed successfully'})
        
    except Exception as e:
        return jsonify({'success': False, 'message': f'Error placing order: {str(e)}'})

@app.route('/api/close_position', methods=['POST'])
def close_position():
    """Close a specific position"""
    if not mt5_conn.connected:
        return jsonify({'success': False, 'message': 'Not connected to MT5'})
    
    data = request.get_json()
    ticket = int(data.get('ticket'))
    
    try:
        # Get position
        position = mt5.positions_get(ticket=ticket)
        if not position:
            return jsonify({'success': False, 'message': 'Position not found'})
        
        position = position[0]
        
        # Close position
        request_dict = {
            "action": mt5.TRADE_ACTION_DEAL,
            "symbol": position.symbol,
            "volume": position.volume,
            "type": mt5.ORDER_TYPE_SELL if position.type == 0 else mt5.ORDER_TYPE_BUY,
            "position": ticket,
            "price": position.price_current,
            "deviation": 20,
            "magic": 234000,
            "comment": "Position closed from web interface",
            "type_time": mt5.ORDER_TIME_GTC,
            "type_filling": mt5.ORDER_FILLING_IOC,
        }
        
        result = mt5.order_send(request_dict)
        
        if result.retcode != mt5.TRADE_RETCODE_DONE:
            return jsonify({'success': False, 'message': f'Close failed: {result.comment}'})
        
        return jsonify({'success': True, 'message': 'Position closed successfully'})
        
    except Exception as e:
        return jsonify({'success': False, 'message': f'Error closing position: {str(e)}'})

@app.route('/api/close_all_positions', methods=['POST'])
def close_all_positions():
    """Close all open positions"""
    if not mt5_conn.connected:
        return jsonify({'success': False, 'message': 'Not connected to MT5'})
    
    try:
        positions = mt5.positions_get()
        if not positions:
            return jsonify({'success': True, 'message': 'No positions to close'})
        
        closed_count = 0
        for position in positions:
            request_dict = {
                "action": mt5.TRADE_ACTION_DEAL,
                "symbol": position.symbol,
                "volume": position.volume,
                "type": mt5.ORDER_TYPE_SELL if position.type == 0 else mt5.ORDER_TYPE_BUY,
                "position": position.ticket,
                "price": position.price_current,
                "deviation": 20,
                "magic": 234000,
                "comment": "Position closed from web interface",
                "type_time": mt5.ORDER_TIME_GTC,
                "type_filling": mt5.ORDER_FILLING_IOC,
            }
            
            result = mt5.order_send(request_dict)
            if result.retcode == mt5.TRADE_RETCODE_DONE:
                closed_count += 1
        
        return jsonify({'success': True, 'message': f'Closed {closed_count} positions'})
        
    except Exception as e:
        return jsonify({'success': False, 'message': f'Error closing positions: {str(e)}'})

@app.route('/api/get_symbols')
def get_symbols():
    """Get available trading symbols"""
    if not mt5_conn.connected:
        return jsonify({'symbols': []})
    
    try:
        symbols = mt5.symbols_get()
        symbol_list = []
        
        for symbol in symbols:
            if 'Volatility' in symbol.name or 'Index' in symbol.name:
                symbol_list.append({
                    'name': symbol.name,
                    'description': symbol.description,
                    'min_volume': symbol.volume_min,
                    'max_volume': symbol.volume_max,
                    'volume_step': symbol.volume_step
                })
        
        return jsonify({'symbols': symbol_list})
        
    except Exception as e:
        return jsonify({'symbols': [], 'error': str(e)})

def create_unified_template():
    """Create the unified dashboard template"""
    os.makedirs('templates', exist_ok=True)
    
    html_content = '''<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>MT5 Trading Bot - Unified Dashboard</title>
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
            max-width: 1400px;
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
        
        .dashboard-grid {
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 30px;
            margin-bottom: 30px;
        }
        
        .account-card, .bot-control-card {
            background: white;
            border-radius: 15px;
            padding: 25px;
            box-shadow: 0 5px 15px rgba(0,0,0,0.08);
            border: 1px solid #e9ecef;
        }
        
        .card-title {
            font-size: 1.5rem;
            font-weight: 600;
            margin-bottom: 20px;
            color: #333;
        }
        
        .account-stats {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(150px, 1fr));
            gap: 15px;
            margin-bottom: 20px;
        }
        
        .stat-item {
            text-align: center;
            padding: 15px;
            background: #f8f9fa;
            border-radius: 10px;
        }
        
        .stat-value {
            font-size: 1.5rem;
            font-weight: 700;
            color: #333;
        }
        
        .stat-label {
            font-size: 0.9rem;
            color: #666;
            margin-top: 5px;
        }
        
        .profit { color: #28a745; }
        .loss { color: #dc3545; }
        .neutral { color: #6c757d; }
        
        .bot-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
            gap: 20px;
            margin-bottom: 20px;
        }
        
        .bot-card {
            background: white;
            border-radius: 15px;
            padding: 20px;
            box-shadow: 0 5px 15px rgba(0,0,0,0.08);
            border: 1px solid #e9ecef;
        }
        
        .bot-header {
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-bottom: 15px;
        }
        
        .bot-title {
            font-size: 1.2rem;
            font-weight: 600;
            color: #333;
        }
        
        .status-indicator {
            width: 12px;
            height: 12px;
            border-radius: 50%;
        }
        
        .status-running { background: #28a745; }
        .status-stopped { background: #dc3545; }
        .status-starting { background: #ffc107; }
        
        .btn {
            padding: 10px 20px;
            border: none;
            border-radius: 8px;
            font-size: 1rem;
            font-weight: 600;
            cursor: pointer;
            transition: all 0.3s ease;
            margin: 5px;
        }
        
        .btn-success { background: #28a745; color: white; }
        .btn-danger { background: #dc3545; color: white; }
        .btn-warning { background: #ffc107; color: #212529; }
        .btn-info { background: #17a2b8; color: white; }
        
        .btn:hover {
            transform: translateY(-2px);
            box-shadow: 0 5px 15px rgba(0,0,0,0.2);
        }
        
        .trading-section {
            margin-top: 30px;
        }
        
        .trading-form {
            background: #f8f9fa;
            padding: 20px;
            border-radius: 10px;
            margin-bottom: 20px;
        }
        
        .form-group {
            margin-bottom: 15px;
        }
        
        .form-label {
            display: block;
            margin-bottom: 5px;
            font-weight: 600;
            color: #333;
        }
        
        .form-input {
            width: 100%;
            padding: 10px;
            border: 1px solid #ddd;
            border-radius: 5px;
            font-size: 1rem;
        }
        
        .positions-table {
            width: 100%;
            border-collapse: collapse;
            margin-top: 20px;
        }
        
        .positions-table th,
        .positions-table td {
            padding: 12px;
            text-align: left;
            border-bottom: 1px solid #ddd;
        }
        
        .positions-table th {
            background: #f8f9fa;
            font-weight: 600;
        }
        
        .alert {
            padding: 15px;
            border-radius: 8px;
            margin-bottom: 20px;
        }
        
        .alert-success { background: #d4edda; color: #155724; border: 1px solid #c3e6cb; }
        .alert-danger { background: #f8d7da; color: #721c24; border: 1px solid #f5c6cb; }
        .alert-warning { background: #fff3cd; color: #856404; border: 1px solid #ffeaa7; }
        
        .loading {
            display: none;
            text-align: center;
            padding: 20px;
        }
        
        .spinner {
            border: 4px solid #f3f3f3;
            border-top: 4px solid #3498db;
            border-radius: 50%;
            width: 40px;
            height: 40px;
            animation: spin 1s linear infinite;
            margin: 0 auto;
        }
        
        @keyframes spin {
            0% { transform: rotate(0deg); }
            100% { transform: rotate(360deg); }
        }
        
        @media (max-width: 768px) {
            .dashboard-grid {
                grid-template-columns: 1fr;
            }
            
            .bot-grid {
                grid-template-columns: 1fr;
            }
        }
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1><i class="fas fa-chart-line"></i> MT5 Trading Bot</h1>
            <p>Unified Dashboard for Demo, Real & Micro Trading</p>
        </div>
        
        <div class="content">
            <!-- Account Overview -->
            <div class="dashboard-grid">
                <div class="account-card">
                    <div class="card-title">
                        <i class="fas fa-wallet"></i> Account Overview
                    </div>
                    <div class="account-stats">
                        <div class="stat-item">
                            <div class="stat-value" id="balance">$0.00</div>
                            <div class="stat-label">Balance</div>
                        </div>
                        <div class="stat-item">
                            <div class="stat-value" id="equity">$0.00</div>
                            <div class="stat-label">Equity</div>
                        </div>
                        <div class="stat-item">
                            <div class="stat-value" id="profit">$0.00</div>
                            <div class="stat-label">Profit</div>
                        </div>
                        <div class="stat-item">
                            <div class="stat-value" id="win-rate">0%</div>
                            <div class="stat-label">Win Rate</div>
                        </div>
                    </div>
                    <div class="account-stats">
                        <div class="stat-item">
                            <div class="stat-value" id="total-trades">0</div>
                            <div class="stat-label">Total Trades</div>
                        </div>
                        <div class="stat-item">
                            <div class="stat-value" id="open-positions">0</div>
                            <div class="stat-label">Open Positions</div>
                        </div>
                        <div class="stat-item">
                            <div class="stat-value" id="total-pnl">$0.00</div>
                            <div class="stat-label">Total PnL</div>
                        </div>
                        <div class="stat-item">
                            <div class="stat-value" id="margin-level">0%</div>
                            <div class="stat-label">Margin Level</div>
                        </div>
                    </div>
                    <div style="text-align: center; margin-top: 20px;">
                        <button class="btn btn-info" onclick="connectMT5()">
                            <i class="fas fa-plug"></i> Connect MT5
                        </button>
                        <button class="btn btn-warning" onclick="disconnectMT5()">
                            <i class="fas fa-unlink"></i> Disconnect
                        </button>
                    </div>
                </div>
                
                <div class="bot-control-card">
                    <div class="card-title">
                        <i class="fas fa-robot"></i> Bot Control
                    </div>
                    <div class="bot-grid">
                        <div class="bot-card">
                            <div class="bot-header">
                                <div class="bot-title">Demo Bot</div>
                                <div class="status-indicator status-stopped" id="demo-status"></div>
                            </div>
                            <p style="color: #666; margin-bottom: 15px;">Safe demo account trading</p>
                            <button class="btn btn-success" onclick="startBot('demo')">
                                <i class="fas fa-play"></i> Start
                            </button>
                            <button class="btn btn-danger" onclick="stopBot('demo')">
                                <i class="fas fa-stop"></i> Stop
                            </button>
                        </div>
                        
                        <div class="bot-card">
                            <div class="bot-header">
                                <div class="bot-title">Real Bot</div>
                                <div class="status-indicator status-stopped" id="real-status"></div>
                            </div>
                            <p style="color: #666; margin-bottom: 15px;">Real account trading</p>
                            <button class="btn btn-success" onclick="startBot('real')">
                                <i class="fas fa-play"></i> Start
                            </button>
                            <button class="btn btn-danger" onclick="stopBot('real')">
                                <i class="fas fa-stop"></i> Stop
                            </button>
                        </div>
                        
                        <div class="bot-card">
                            <div class="bot-header">
                                <div class="bot-title">Micro Bot</div>
                                <div class="status-indicator status-stopped" id="micro-status"></div>
                            </div>
                            <p style="color: #666; margin-bottom: 15px;">Small account optimized</p>
                            <button class="btn btn-success" onclick="startBot('micro')">
                                <i class="fas fa-play"></i> Start
                            </button>
                            <button class="btn btn-danger" onclick="stopBot('micro')">
                                <i class="fas fa-stop"></i> Stop
                            </button>
                        </div>
                    </div>
                    <div style="text-align: center; margin-top: 20px;">
                        <button class="btn btn-danger" onclick="stopAllBots()">
                            <i class="fas fa-power-off"></i> Stop All Bots
                        </button>
                    </div>
                </div>
            </div>
            
            <!-- Manual Trading -->
            <div class="trading-section">
                <div class="card-title">
                    <i class="fas fa-hand-paper"></i> Manual Trading
                </div>
                <div class="trading-form">
                    <div style="display: grid; grid-template-columns: 1fr 1fr 1fr 1fr; gap: 15px;">
                        <div class="form-group">
                            <label class="form-label">Symbol</label>
                            <select class="form-input" id="trade-symbol">
                                <option value="Volatility 100 Index">Volatility 100 Index</option>
                                <option value="Volatility 75 Index">Volatility 75 Index</option>
                                <option value="Volatility 50 Index">Volatility 50 Index</option>
                            </select>
                        </div>
                        <div class="form-group">
                            <label class="form-label">Type</label>
                            <select class="form-input" id="trade-type">
                                <option value="BUY">BUY</option>
                                <option value="SELL">SELL</option>
                            </select>
                        </div>
                        <div class="form-group">
                            <label class="form-label">Volume</label>
                            <input type="number" class="form-input" id="trade-volume" value="0.01" step="0.01" min="0.01">
                        </div>
                        <div class="form-group">
                            <label class="form-label">Price</label>
                            <input type="number" class="form-input" id="trade-price" value="0" step="0.01">
                        </div>
                    </div>
                    <div style="text-align: center; margin-top: 20px;">
                        <button class="btn btn-success" onclick="placeTrade()">
                            <i class="fas fa-plus"></i> Place Trade
                        </button>
                        <button class="btn btn-danger" onclick="closeAllPositions()">
                            <i class="fas fa-times"></i> Close All Positions
                        </button>
                    </div>
                </div>
                
                <!-- Open Positions -->
                <div id="positions-container">
                    <h3 style="margin-bottom: 15px;">Open Positions</h3>
                    <div id="positions-table-container">
                        <p style="text-align: center; color: #666;">No open positions</p>
                    </div>
                </div>
            </div>
            
            <!-- Alerts -->
            <div id="alerts-container"></div>
            
            <!-- Loading -->
            <div class="loading" id="loading">
                <div class="spinner"></div>
                <p>Processing...</p>
            </div>
        </div>
    </div>
    
    <script>
        // Global variables
        let updateInterval;
        
        // Initialize dashboard
        document.addEventListener('DOMContentLoaded', function() {
            updateStatus();
            updateInterval = setInterval(updateStatus, 5000); // Update every 5 seconds
        });
        
        // Update status
        async function updateStatus() {
            try {
                const response = await fetch('/api/status');
                const data = await response.json();
                
                // Update account data
                updateAccountData(data.account);
                
                // Update bot status
                updateBotStatus(data.bots);
                
                // Update positions
                updatePositions(data.account.open_positions);
                
            } catch (error) {
                console.error('Error updating status:', error);
            }
        }
        
        // Update account data
        function updateAccountData(account) {
            document.getElementById('balance').textContent = '$' + account.balance.toFixed(2);
            document.getElementById('equity').textContent = '$' + account.equity.toFixed(2);
            
            const profitElement = document.getElementById('profit');
            profitElement.textContent = '$' + account.profit.toFixed(2);
            profitElement.className = 'stat-value ' + (account.profit >= 0 ? 'profit' : 'loss');
            
            document.getElementById('win-rate').textContent = account.win_rate.toFixed(1) + '%';
            document.getElementById('total-trades').textContent = account.total_trades;
            document.getElementById('open-positions').textContent = account.open_positions.length;
            document.getElementById('total-pnl').textContent = '$' + account.total_pnl.toFixed(2);
            document.getElementById('margin-level').textContent = account.margin_level.toFixed(1) + '%';
        }
        
        // Update bot status
        function updateBotStatus(bots) {
            for (const [botType, status] of Object.entries(bots)) {
                const statusElement = document.getElementById(botType + '-status');
                if (statusElement) {
                    statusElement.className = 'status-indicator status-' + status;
                }
            }
        }
        
        // Update positions table
        function updatePositions(positions) {
            const container = document.getElementById('positions-table-container');
            
            if (positions.length === 0) {
                container.innerHTML = '<p style="text-align: center; color: #666;">No open positions</p>';
                return;
            }
            
            let table = '<table class="positions-table">';
            table += '<thead><tr><th>Ticket</th><th>Symbol</th><th>Type</th><th>Volume</th><th>Open Price</th><th>Current Price</th><th>Profit</th><th>Time</th><th>Action</th></tr></thead><tbody>';
            
            positions.forEach(pos => {
                const profitClass = pos.profit >= 0 ? 'profit' : 'loss';
                table += `<tr>
                    <td>${pos.ticket}</td>
                    <td>${pos.symbol}</td>
                    <td>${pos.type}</td>
                    <td>${pos.volume}</td>
                    <td>${pos.price_open.toFixed(5)}</td>
                    <td>${pos.price_current.toFixed(5)}</td>
                    <td class="${profitClass}">$${pos.profit.toFixed(2)}</td>
                    <td>${pos.time}</td>
                    <td><button class="btn btn-danger btn-sm" onclick="closePosition(${pos.ticket})">Close</button></td>
                </tr>`;
            });
            
            table += '</tbody></table>';
            container.innerHTML = table;
        }
        
        // Bot control functions
        async function startBot(botType) {
            showLoading();
            try {
                const response = await fetch('/api/start_bot', {
                    method: 'POST',
                    headers: {'Content-Type': 'application/json'},
                    body: JSON.stringify({bot_type: botType})
                });
                const data = await response.json();
                showAlert(data.success ? 'success' : 'danger', data.message);
                updateStatus();
            } catch (error) {
                showAlert('danger', 'Error starting bot: ' + error.message);
            }
            hideLoading();
        }
        
        async function stopBot(botType) {
            showLoading();
            try {
                const response = await fetch('/api/stop_bot', {
                    method: 'POST',
                    headers: {'Content-Type': 'application/json'},
                    body: JSON.stringify({bot_type: botType})
                });
                const data = await response.json();
                showAlert(data.success ? 'success' : 'danger', data.message);
                updateStatus();
            } catch (error) {
                showAlert('danger', 'Error stopping bot: ' + error.message);
            }
            hideLoading();
        }
        
        async function stopAllBots() {
            showLoading();
            try {
                const response = await fetch('/api/stop_all', {
                    method: 'POST',
                    headers: {'Content-Type': 'application/json'}
                });
                const data = await response.json();
                showAlert(data.success ? 'success' : 'danger', data.message);
                updateStatus();
            } catch (error) {
                showAlert('danger', 'Error stopping bots: ' + error.message);
            }
            hideLoading();
        }
        
        // MT5 connection functions
        async function connectMT5() {
            showLoading();
            try {
                const response = await fetch('/api/connect_mt5', {
                    method: 'POST',
                    headers: {'Content-Type': 'application/json'},
                    body: JSON.stringify({account_type: 'real'})
                });
                const data = await response.json();
                showAlert(data.success ? 'success' : 'danger', data.message);
                updateStatus();
            } catch (error) {
                showAlert('danger', 'Error connecting to MT5: ' + error.message);
            }
            hideLoading();
        }
        
        async function disconnectMT5() {
            showLoading();
            try {
                const response = await fetch('/api/disconnect_mt5', {
                    method: 'POST',
                    headers: {'Content-Type': 'application/json'}
                });
                const data = await response.json();
                showAlert(data.success ? 'success' : 'danger', data.message);
                updateStatus();
            } catch (error) {
                showAlert('danger', 'Error disconnecting from MT5: ' + error.message);
            }
            hideLoading();
        }
        
        // Trading functions
        async function placeTrade() {
            const symbol = document.getElementById('trade-symbol').value;
            const type = document.getElementById('trade-type').value;
            const volume = parseFloat(document.getElementById('trade-volume').value);
            const price = parseFloat(document.getElementById('trade-price').value);
            
            if (!symbol || !type || !volume) {
                showAlert('warning', 'Please fill in all required fields');
                return;
            }
            
            showLoading();
            try {
                const response = await fetch('/api/place_trade', {
                    method: 'POST',
                    headers: {'Content-Type': 'application/json'},
                    body: JSON.stringify({
                        symbol: symbol,
                        type: type,
                        volume: volume,
                        price: price
                    })
                });
                const data = await response.json();
                showAlert(data.success ? 'success' : 'danger', data.message);
                updateStatus();
            } catch (error) {
                showAlert('danger', 'Error placing trade: ' + error.message);
            }
            hideLoading();
        }
        
        async function closePosition(ticket) {
            showLoading();
            try {
                const response = await fetch('/api/close_position', {
                    method: 'POST',
                    headers: {'Content-Type': 'application/json'},
                    body: JSON.stringify({ticket: ticket})
                });
                const data = await response.json();
                showAlert(data.success ? 'success' : 'danger', data.message);
                updateStatus();
            } catch (error) {
                showAlert('danger', 'Error closing position: ' + error.message);
            }
            hideLoading();
        }
        
        async function closeAllPositions() {
            showLoading();
            try {
                const response = await fetch('/api/close_all_positions', {
                    method: 'POST',
                    headers: {'Content-Type': 'application/json'}
                });
                const data = await response.json();
                showAlert(data.success ? 'success' : 'danger', data.message);
                updateStatus();
            } catch (error) {
                showAlert('danger', 'Error closing positions: ' + error.message);
            }
            hideLoading();
        }
        
        // Utility functions
        function showAlert(type, message) {
            const container = document.getElementById('alerts-container');
            const alert = document.createElement('div');
            alert.className = `alert alert-${type}`;
            alert.innerHTML = message;
            container.appendChild(alert);
            
            setTimeout(() => {
                alert.remove();
            }, 5000);
        }
        
        function showLoading() {
            document.getElementById('loading').style.display = 'block';
        }
        
        function hideLoading() {
            document.getElementById('loading').style.display = 'none';
        }
    </script>
</body>
</html>'''
    
    with open('templates/unified_dashboard.html', 'w', encoding='utf-8') as f:
        f.write(html_content)

def main():
    """Main function to start the unified web interface"""
    print("🌐 MT5 Trading Bot - Unified Web Interface")
    print("=" * 50)
    
    # Create template
    create_unified_template()
    
    # Get local IP
    import socket
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("8.8.8.8", 80))
        local_ip = s.getsockname()[0]
        s.close()
    except:
        local_ip = "localhost"
    
    print(f"\n📱 Access from your phone:")
    print(f"   http://{local_ip}:5000")
    print(f"\n💻 Access from your PC:")
    print(f"   http://localhost:5000")
    print(f"\n🔧 Make sure your phone and PC are on the same WiFi network!")
    print(f"\n⚠️  IMPORTANT: Make sure MT5 is running and AutoTrading is enabled!")
    print(f"\n🚀 Starting unified web interface...")
    print("=" * 50)
    
    # Start the web interface
    try:
        app.run(host='0.0.0.0', port=5000, debug=False)
    except KeyboardInterrupt:
        print("\n🛑 Web interface stopped by user")
    except Exception as e:
        print(f"❌ Error starting web interface: {e}")

if __name__ == "__main__":
    main() 