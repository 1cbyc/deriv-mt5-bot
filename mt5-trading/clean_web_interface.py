#!/usr/bin/env python3
"""
Clean MT5 Trading Bot Web Interface
===================================

A clean Flask web application that manages MT5 bots as separate processes.
"""

from flask import Flask, render_template, jsonify, request
import os
from datetime import datetime
from bot_manager import bot_manager

app = Flask(__name__)

@app.route('/')
def index():
    """Main dashboard page"""
    return render_template('clean_dashboard.html')

@app.route('/api/status')
def get_status():
    """API endpoint to get bot status"""
    return jsonify(bot_manager.get_status())

@app.route('/api/start', methods=['POST'])
def start_bot():
    """API endpoint to start a bot"""
    data = request.get_json() or {}
    bot_type = data.get('bot_type', 'demo')
    
    success, message = bot_manager.start_bot(bot_type)
    return jsonify({'success': success, 'message': message})

@app.route('/api/stop', methods=['POST'])
def stop_bot():
    """API endpoint to stop a bot"""
    data = request.get_json() or {}
    bot_type = data.get('bot_type', 'demo')
    
    success, message = bot_manager.stop_bot(bot_type)
    return jsonify({'success': success, 'message': message})

@app.route('/api/stop_all', methods=['POST'])
def stop_all_bots():
    """API endpoint to stop all bots"""
    success, message = bot_manager.stop_all_bots()
    return jsonify({'success': success, 'message': message})

@app.route('/api/bot_info/<bot_type>')
def get_bot_info(bot_type):
    """API endpoint to get detailed bot info"""
    return jsonify(bot_manager.get_bot_info(bot_type))

@app.route('/api/config')
def get_config():
    """Get available configuration options"""
    return jsonify({
        'bot_types': [
            {'id': 'demo', 'name': 'Demo Trading Bot', 'description': 'Safe demo account trading'},
            {'id': 'real', 'name': 'Real Trading Bot', 'description': 'Real account trading with risk management'},
            {'id': 'micro', 'name': 'Micro Trading Bot', 'description': 'Very small balance trading ($0.28-$1.00)'}
        ]
    })

def create_clean_template():
    """Create a clean HTML template"""
    os.makedirs('templates', exist_ok=True)
    
    html_content = '''<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>MT5 Trading Bot Dashboard</title>
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
        
        .bot-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(350px, 1fr));
            gap: 20px;
            margin-bottom: 30px;
        }
        
        .bot-card {
            background: white;
            border-radius: 15px;
            padding: 25px;
            box-shadow: 0 5px 15px rgba(0,0,0,0.08);
            border: 1px solid #e9ecef;
        }
        
        .bot-card h3 {
            color: #333;
            margin-bottom: 15px;
            font-size: 1.3rem;
        }
        
        .bot-status {
            display: flex;
            align-items: center;
            margin-bottom: 15px;
        }
        
        .status-indicator {
            width: 12px;
            height: 12px;
            border-radius: 50%;
            margin-right: 10px;
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
        
        .logs-panel {
            background: #f8f9fa;
            border-radius: 15px;
            padding: 20px;
            max-height: 400px;
            overflow-y: auto;
        }
        
        .log-entry {
            padding: 5px 0;
            border-bottom: 1px solid #e9ecef;
            font-family: 'Courier New', monospace;
            font-size: 0.9rem;
        }
        
        .log-entry:last-child {
            border-bottom: none;
        }
        
        .control-panel {
            background: white;
            border-radius: 15px;
            padding: 25px;
            margin-bottom: 20px;
            box-shadow: 0 5px 15px rgba(0,0,0,0.08);
            border: 1px solid #e9ecef;
        }
        
        @media (max-width: 768px) {
            .container {
                margin: 10px;
                border-radius: 15px;
            }
            
            .content {
                padding: 20px;
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
            <p>Process-Based Bot Management Dashboard</p>
        </div>
        
        <div class="content">
            <!-- Alert Messages -->
            <div id="alertContainer"></div>
            
            <!-- Control Panel -->
            <div class="control-panel">
                <h3><i class="fas fa-cogs"></i> Global Controls</h3>
                <button class="btn btn-warning" onclick="stopAllBots()">
                    <i class="fas fa-stop-circle"></i> Stop All Bots
                </button>
            </div>
            
            <!-- Bot Cards -->
            <div class="bot-grid" id="botGrid">
                <!-- Bot cards will be populated by JavaScript -->
            </div>
            
            <!-- Logs -->
            <div class="logs-panel">
                <h3><i class="fas fa-list"></i> System Logs</h3>
                <div id="logsContainer">
                    <div class="log-entry">Dashboard loaded...</div>
                </div>
            </div>
        </div>
    </div>
    
    <script>
        let botConfig = [];
        
        // Initialize the dashboard
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
                botConfig = config.bot_types;
                populateBotGrid();
            } catch (error) {
                console.error('Error loading config:', error);
            }
        }
        
        // Populate bot grid
        function populateBotGrid() {
            const grid = document.getElementById('botGrid');
            grid.innerHTML = '';
            
            botConfig.forEach(bot => {
                const card = document.createElement('div');
                card.className = 'bot-card';
                card.innerHTML = `
                    <h3><i class="fas fa-robot"></i> ${bot.name}</h3>
                    <p>${bot.description}</p>
                    <div class="bot-status">
                        <span class="status-indicator status-stopped" id="status-${bot.id}"></span>
                        <span id="status-text-${bot.id}">Stopped</span>
                    </div>
                    <div>
                        <button class="btn btn-success" onclick="startBot('${bot.id}')" id="start-${bot.id}">
                            <i class="fas fa-play"></i> Start
                        </button>
                        <button class="btn btn-danger" onclick="stopBot('${bot.id}')" id="stop-${bot.id}" style="display: none;">
                            <i class="fas fa-stop"></i> Stop
                        </button>
                    </div>
                `;
                grid.appendChild(card);
            });
        }
        
        // Update status
        async function updateStatus() {
            try {
                const response = await fetch('/api/status');
                const data = await response.json();
                
                // Update bot statuses
                botConfig.forEach(bot => {
                    const status = data.bots[bot.id] || 'stopped';
                    const statusIndicator = document.getElementById(`status-${bot.id}`);
                    const statusText = document.getElementById(`status-text-${bot.id}`);
                    const startBtn = document.getElementById(`start-${bot.id}`);
                    const stopBtn = document.getElementById(`stop-${bot.id}`);
                    
                    statusIndicator.className = `status-indicator status-${status}`;
                    statusText.textContent = status.charAt(0).toUpperCase() + status.slice(1);
                    
                    if (status === 'running') {
                        startBtn.style.display = 'none';
                        stopBtn.style.display = 'inline-block';
                    } else {
                        startBtn.style.display = 'inline-block';
                        stopBtn.style.display = 'none';
                    }
                });
                
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
                logEntry.className = 'log-entry';
                logEntry.textContent = log;
                logsContainer.appendChild(logEntry);
            });
            
            // Scroll to bottom
            logsContainer.scrollTop = logsContainer.scrollHeight;
        }
        
        // Start auto-update
        function startAutoUpdate() {
            setInterval(updateStatus, 2000); // Update every 2 seconds
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
        
        // Bot control functions
        async function startBot(botType) {
            try {
                const response = await fetch('/api/start', {
                    method: 'POST',
                    headers: {
                        'Content-Type': 'application/json'
                    },
                    body: JSON.stringify({bot_type: botType})
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
        }
        
        async function stopBot(botType) {
            try {
                const response = await fetch('/api/stop', {
                    method: 'POST',
                    headers: {
                        'Content-Type': 'application/json'
                    },
                    body: JSON.stringify({bot_type: botType})
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
        }
        
        async function stopAllBots() {
            try {
                const response = await fetch('/api/stop_all', {
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
                showAlert('Error stopping all bots: ' + error.message, 'danger');
            }
        }
    </script>
</body>
</html>'''
    
    with open('templates/clean_dashboard.html', 'w', encoding='utf-8') as f:
        f.write(html_content)

if __name__ == '__main__':
    create_clean_template()
    
    print("🌐 Starting Clean MT5 Trading Bot Web Interface...")
    print("📱 Access from your phone at: http://[YOUR_PC_IP]:5000")
    print("💻 Access from your PC at: http://localhost:5000")
    print("🔧 This version runs bots as separate processes - no import issues!")
    
    app.run(host='0.0.0.0', port=5000, debug=False) 