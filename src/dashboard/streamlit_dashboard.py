"""
Streamlit Dashboard for TIAS

Interactive web dashboard for monitoring trading system performance.
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
import requests
from datetime import datetime, timedelta
import time
import logging

logger = logging.getLogger(__name__)

# Configure Streamlit page
st.set_page_config(
    page_title="TIAS Trading Dashboard",
    page_icon="📈",
    layout="wide"
)

class TradingDashboard:
    """Streamlit dashboard for the TIAS trading system"""
    
    def __init__(self, api_base_url: str = "http://localhost:8000"):
        self.api_base_url = api_base_url
        self.refresh_interval = 5
    
    def run(self):
        """Main dashboard application"""
        st.title("🚀 TIAS Trading System Dashboard")
        st.markdown("---")
        
        # Sidebar
        self._render_sidebar()
        
        # Main content area
        tab1, tab2, tab3 = st.tabs(["📊 Overview", "💼 Positions", "📈 Performance"])
        
        with tab1:
            self._render_overview_tab()
        
        with tab2:
            self._render_positions_tab()
        
        with tab3:
            self._render_performance_tab()
    
    def _render_sidebar(self):
        """Render sidebar with system controls"""
        st.sidebar.header("🎛️ System Controls")
        
        # System status
        try:
            status = self._get_system_status()
            if status:
                st.sidebar.success("✅ System Online")
                st.sidebar.metric("Total Equity", f"${status['total_equity']:,.2f}")
                st.sidebar.metric("Daily P&L", f"${status['daily_pnl']:,.2f}")
                st.sidebar.metric("Open Positions", status['open_positions'])
                
                # Trading controls
                st.sidebar.subheader("Trading Controls")
                
                col1, col2 = st.sidebar.columns(2)
                
                with col1:
                    if st.button("▶️ Start"):
                        self._start_trading()
                
                with col2:
                    if st.button("⏹️ Stop"):
                        self._stop_trading()
                
                # Trading status
                trading_status = "🟢 Active" if status['trading_enabled'] else "🔴 Inactive"
                st.sidebar.write(f"Status: {trading_status}")
            else:
                st.sidebar.error("❌ System Offline")
        
        except Exception as e:
            st.sidebar.error(f"❌ Connection Error")
        
        # Auto-refresh
        st.sidebar.subheader("Settings")
        auto_refresh = st.sidebar.checkbox("Auto Refresh", value=True)
        
        if auto_refresh:
            time.sleep(self.refresh_interval)
            st.rerun()
    
    def _render_overview_tab(self):
        """Render system overview tab"""
        st.header("System Overview")
        
        try:
            status = self._get_system_status()
            performance = self._get_performance_metrics()
            
            if status and performance:
                # Key metrics row
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    st.metric(
                        "Total Equity",
                        f"${status['total_equity']:,.2f}",
                        delta=f"${status['daily_pnl']:,.2f}"
                    )
                
                with col2:
                    st.metric(
                        "Total Return",
                        f"{performance['total_return_pct']:.2%}"
                    )
                
                with col3:
                    st.metric(
                        "Sharpe Ratio",
                        f"{performance['sharpe_ratio']:.2f}"
                    )
                
                with col4:
                    st.metric(
                        "Max Drawdown",
                        f"{performance['max_drawdown']:.2%}"
                    )
                
                # Trading activity
                st.subheader("Trading Activity")
                
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.metric("Signals Processed", status['signals_processed'])
                
                with col2:
                    st.metric("Orders Filled", status['orders_filled'])
                
                with col3:
                    st.metric("Win Rate", f"{performance['win_rate']:.2%}")
                
                # Equity curve chart
                st.subheader("Equity Curve")
                equity_chart = self._create_equity_curve_chart()
                if equity_chart:
                    st.plotly_chart(equity_chart, use_container_width=True)
            
            else:
                st.error("Unable to load system data")
        
        except Exception as e:
            st.error(f"Error loading overview: {str(e)}")
    
    def _render_positions_tab(self):
        """Render positions tab"""
        st.header("Current Positions")
        
        try:
            positions = self._get_positions()
            
            if positions:
                # Create positions DataFrame
                df = pd.DataFrame(positions)
                df['timestamp'] = pd.to_datetime(df['timestamp'])
                
                # Display positions table
                st.dataframe(df, use_container_width=True)
                
                # Position summary
                st.subheader("Position Summary")
                
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    total_value = (df['quantity'] * df['current_price']).sum()
                    st.metric("Total Position Value", f"${total_value:,.2f}")
                
                with col2:
                    total_pnl = df['unrealized_pnl'].sum()
                    st.metric("Total Unrealized P&L", f"${total_pnl:,.2f}")
                
                with col3:
                    st.metric("Number of Positions", len(df))
            
            else:
                st.info("No open positions")
        
        except Exception as e:
            st.error(f"Error loading positions: {str(e)}")
    
    def _render_performance_tab(self):
        """Render performance analysis tab"""
        st.header("Performance Analysis")
        
        try:
            performance = self._get_performance_metrics()
            
            if performance:
                # Performance metrics
                col1, col2 = st.columns(2)
                
                with col1:
                    st.subheader("Return Metrics")
                    st.metric("Total Return", f"{performance['total_return_pct']:.2%}")
                    st.metric("Total Return ($)", f"${performance['total_return']:,.2f}")
                    st.metric("Win Rate", f"{performance['win_rate']:.2%}")
                    st.metric("Total Trades", performance['total_trades'])
                
                with col2:
                    st.subheader("Risk Metrics")
                    st.metric("Sharpe Ratio", f"{performance['sharpe_ratio']:.2f}")
                    st.metric("Max Drawdown", f"{performance['max_drawdown']:.2%}")
            
            else:
                st.error("Unable to load performance data")
        
        except Exception as e:
            st.error(f"Error loading performance: {str(e)}")
    
    def _get_system_status(self):
        """Get system status from API"""
        try:
            response = requests.get(f"{self.api_base_url}/status", timeout=5)
            if response.status_code == 200:
                return response.json()
            return None
        except Exception:
            return None
    
    def _get_positions(self):
        """Get positions from API"""
        try:
            response = requests.get(f"{self.api_base_url}/positions", timeout=5)
            if response.status_code == 200:
                return response.json()
            return []
        except Exception:
            return []
    
    def _get_performance_metrics(self):
        """Get performance metrics from API"""
        try:
            response = requests.get(f"{self.api_base_url}/performance", timeout=5)
            if response.status_code == 200:
                return response.json()
            return None
        except Exception:
            return None
    
    def _start_trading(self):
        """Start trading via API"""
        try:
            response = requests.post(f"{self.api_base_url}/trading/start", timeout=5)
            if response.status_code == 200:
                st.sidebar.success("Trading started!")
            else:
                st.sidebar.error("Failed to start trading")
        except Exception:
            st.sidebar.error("Error starting trading")
    
    def _stop_trading(self):
        """Stop trading via API"""
        try:
            response = requests.post(f"{self.api_base_url}/trading/stop", timeout=5)
            if response.status_code == 200:
                st.sidebar.success("Trading stopped!")
            else:
                st.sidebar.error("Failed to stop trading")
        except Exception:
            st.sidebar.error("Error stopping trading")
    
    def _create_equity_curve_chart(self):
        """Create equity curve chart"""
        try:
            # Create sample data for demo
            dates = pd.date_range(start=datetime.now() - timedelta(days=30), 
                                end=datetime.now(), freq='H')
            equity = 100000 * (1 + np.cumsum(np.random.normal(0.0001, 0.01, len(dates))))
            
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=dates,
                y=equity,
                mode='lines',
                name='Equity',
                line=dict(color='blue', width=2)
            ))
            
            fig.update_layout(
                title="Equity Curve",
                xaxis_title="Date",
                yaxis_title="Equity ($)",
                hovermode='x unified'
            )
            
            return fig
        except Exception:
            return None

# Main application
if __name__ == "__main__":
    dashboard = TradingDashboard()
    dashboard.run() 