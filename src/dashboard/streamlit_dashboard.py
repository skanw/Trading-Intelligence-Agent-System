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
import json
import asyncio
import yfinance as yf
import os
from threading import Thread
import queue
try:
    from newsapi import NewsApiClient
    NEWSAPI_AVAILABLE = True
except ImportError:
    NEWSAPI_AVAILABLE = False

logger = logging.getLogger(__name__)

# Configure Streamlit page
st.set_page_config(
    page_title="TIAS Trading Dashboard",
    page_icon="📈",
    layout="wide"
)

# Enhanced CSS for better visual hierarchy and accessibility
st.markdown("""
<style>
    /* Main layout improvements */
    .main > div {
        padding-top: 1rem;
    }
    
    /* Enhanced metric cards with gradients and shadows */
    .stMetric {
        background: linear-gradient(135deg, #1e293b 0%, #334155 100%);
        border: 1px solid #475569;
        padding: 1.5rem;
        border-radius: 12px;
        margin: 0.75rem 0;
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.15);
        transition: all 0.3s ease;
    }
    
    .stMetric:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 20px rgba(0, 0, 0, 0.2);
    }
    
    .stMetric > div {
        color: #f8fafc !important;
    }
    
    .stMetric [data-testid="metric-container"] {
        background: transparent;
    }
    
    /* Improved metric values */
    .stMetric [data-testid="metric-container"] > div:first-child {
        font-size: 2rem !important;
        font-weight: 700 !important;
        letter-spacing: -0.025em !important;
        color: #ffffff !important;
    }
    
    /* Metric labels */
    .stMetric [data-testid="metric-container"] > div:last-child {
        font-size: 0.875rem !important;
        color: #94a3b8 !important;
        font-weight: 500 !important;
    }
    
    /* Delta values with better colors */
    .stMetric [data-testid="metric-container"] div[data-testid="metric-delta"] {
        color: #10b981 !important;
        font-weight: 600 !important;
    }
    
    /* Accessible color system */
    .metric-positive {
        background: linear-gradient(135deg, #065f46 0%, #047857 100%);
        border-color: #10b981;
    }
    
    .metric-negative {
        background: linear-gradient(135deg, #7f1d1d 0%, #991b1b 100%);
        border-color: #ef4444;
    }
    
    .metric-neutral {
        background: linear-gradient(135deg, #374151 0%, #4b5563 100%);
        border-color: #6b7280;
    }
    
    /* Enhanced cards */
    .trading-card {
        background: linear-gradient(135deg, #1e293b 0%, #334155 100%);
        border: 1px solid #475569;
        border-radius: 12px;
        padding: 1.5rem;
        margin: 1rem 0;
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.15);
        transition: all 0.3s ease;
        color: #f8fafc;
    }
    
    .trading-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 20px rgba(0, 0, 0, 0.2);
    }
    
    /* Skeleton loader */
    .skeleton {
        background: linear-gradient(90deg, #374151 25%, #4b5563 50%, #374151 75%);
        background-size: 200% 100%;
        animation: loading 1.5s infinite;
        border-radius: 6px;
        height: 1.5rem;
        width: 100%;
    }
    
    @keyframes loading {
        0% { background-position: 200% 0; }
        100% { background-position: -200% 0; }
    }
    
    /* Enhanced watchlist */
    .watchlist-item {
        background: linear-gradient(135deg, #1e293b 0%, #334155 100%);
        border: 1px solid #475569;
        border-radius: 8px;
        padding: 1rem;
        margin: 0.5rem 0;
        transition: all 0.3s ease;
        cursor: pointer;
    }
    
    .watchlist-item:hover {
        transform: translateX(4px);
        border-color: #3b82f6;
        box-shadow: 0 4px 12px rgba(59, 130, 246, 0.15);
    }
    
    /* Signal items with better accessibility */
    .signal-item {
        background: linear-gradient(135deg, #ffffff 0%, #f8fafc 100%);
        border-left: 4px solid #3b82f6;
        padding: 1rem;
        margin: 0.5rem 0;
        border-radius: 8px;
        box-shadow: 0 2px 8px rgba(0, 0, 0, 0.1);
        transition: all 0.3s ease;
        position: relative;
    }
    
    .signal-item::before {
        content: "→";
        position: absolute;
        left: 1rem;
        top: 50%;
        transform: translateY(-50%);
        font-weight: bold;
        font-size: 1.2rem;
    }
    
    .signal-buy {
        border-left-color: #10b981;
        background: linear-gradient(135deg, #ecfdf5 0%, #f0fdf4 100%);
    }
    
    .signal-buy::before {
        content: "↗";
        color: #10b981;
    }
    
    .signal-sell {
        border-left-color: #ef4444;
        background: linear-gradient(135deg, #fef2f2 0%, #fef2f2 100%);
    }
    
    .signal-sell::before {
        content: "↘";
        color: #ef4444;
    }
    
    .signal-hold {
        border-left-color: #f59e0b;
        background: linear-gradient(135deg, #fffbeb 0%, #fefce8 100%);
    }
    
    .signal-hold::before {
        content: "→";
        color: #f59e0b;
    }
    
    /* Compact header */
    .compact-header {
        padding: 0.5rem 0;
        margin-bottom: 1rem;
    }
    
    .compact-header h1 {
        font-size: 1.875rem !important;
        margin: 0 !important;
        font-weight: 700;
    }
    
    /* Status indicators */
    .status-live {
        background: linear-gradient(135deg, #065f46 0%, #047857 100%);
        color: white;
        padding: 0.5rem 1rem;
        border-radius: 20px;
        font-weight: 600;
        font-size: 0.875rem;
        display: inline-flex;
        align-items: center;
        gap: 0.5rem;
    }
    
    .status-paused {
        background: linear-gradient(135deg, #92400e 0%, #b45309 100%);
        color: white;
        padding: 0.5rem 1rem;
        border-radius: 20px;
        font-weight: 600;
        font-size: 0.875rem;
        display: inline-flex;
        align-items: center;
        gap: 0.5rem;
    }
    
    /* Responsive improvements */
    @media (max-width: 768px) {
        .stMetric {
            padding: 1rem;
            margin: 0.5rem 0;
        }
        
        .trading-card {
            padding: 1rem;
            margin: 0.5rem 0;
        }
    }
    
    /* Tab styling */
    .stTabs [data-baseweb="tab-list"] {
        gap: 0.5rem;
    }
    
    .stTabs [data-baseweb="tab"] {
        background: linear-gradient(135deg, #374151 0%, #4b5563 100%);
        color: #d1d5db;
        border-radius: 8px;
        padding: 0.75rem 1.5rem;
        transition: all 0.3s ease;
    }
    
    .stTabs [data-baseweb="tab"]:hover {
        background: linear-gradient(135deg, #4b5563 0%, #6b7280 100%);
        transform: translateY(-1px);
    }
    
    .stTabs [aria-selected="true"] {
        background: linear-gradient(135deg, #3b82f6 0%, #2563eb 100%) !important;
        color: white !important;
    }
</style>
""", unsafe_allow_html=True)

class TradingDashboard:
    """Enhanced Streamlit dashboard for the TIAS trading system"""
    
    def __init__(self, api_base_url: str = "http://localhost:8000"):
        self.api_base_url = api_base_url
        self.refresh_interval = 10  # Refresh every 10 seconds
        self.redis_client = None
        self.signal_queue = queue.Queue()
        
        # Initialize session state
        if 'signals' not in st.session_state:
            st.session_state.signals = []
        if 'filters' not in st.session_state:
            st.session_state.filters = {}
        if 'watchlist' not in st.session_state:
            st.session_state.watchlist = []
        if 'last_refresh' not in st.session_state:
            st.session_state.last_refresh = datetime.now()
        if 'auto_refresh_enabled' not in st.session_state:
            st.session_state.auto_refresh_enabled = True
    
    def run(self):
        """Main dashboard application"""
        # Compact header with better status indicators
        st.markdown('<div class="compact-header">', unsafe_allow_html=True)
        
        col1, col2, col3 = st.columns([4, 2, 2])
        
        with col1:
            st.markdown("# 🚀 TIAS Trading System")
        
        with col2:
            # Enhanced status indicator
            auto_refresh_enabled = st.session_state.get('auto_refresh_enabled', True)
            if auto_refresh_enabled:
                st.markdown('<div class="status-live">🟢 LIVE</div>', unsafe_allow_html=True)
            else:
                st.markdown('<div class="status-paused">⏸️ PAUSED</div>', unsafe_allow_html=True)
        
        with col3:
            # Current time with better styling
            current_time = datetime.now().strftime("%H:%M:%S")
            st.markdown(f"**⏰ {current_time}**")
        
        st.markdown('</div>', unsafe_allow_html=True)
        
        # Sidebar
        self._render_sidebar()
        
        # Main content area with enhanced tabs
        tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs([
            "📊 Overview", 
            "📈 Charts & Watchlist",
            "💼 Positions", 
            "📊 Performance", 
            "🎯 Signals", 
            "📰 News", 
            "🔧 Filters"
        ])
        
        with tab1:
            self._render_overview_tab()
        
        with tab2:
            self._render_charts_watchlist_tab()
        
        with tab3:
            self._render_positions_tab()
        
        with tab4:
            self._render_performance_tab()
        
        with tab5:
            self._render_signals_tab()
        
        with tab6:
            self._render_news_tab()
        
        with tab7:
            self._render_filters_tab()
    
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
        
        # Watchlist
        st.sidebar.subheader("📋 Watchlist")
        watchlist = self._get_watchlist()
        if watchlist:
            st.sidebar.text_area("Current Watchlist", value="\n".join(watchlist), height=100)
        
        # Auto-refresh settings
        st.sidebar.subheader("Settings")
        auto_refresh = st.sidebar.checkbox("Auto Refresh", value=st.session_state.get('auto_refresh_enabled', True))
        
        # Update session state
        st.session_state.auto_refresh_enabled = auto_refresh
        
        if auto_refresh:
            refresh_rate = st.sidebar.selectbox(
                "Refresh Rate",
                [5, 10, 30, 60],
                index=1,
                format_func=lambda x: f"{x} seconds"
            )
            
            # Show last refresh time
            time_since_refresh = (datetime.now() - st.session_state.last_refresh).seconds
            st.sidebar.write(f"Last refresh: {time_since_refresh}s ago")
            
            # Auto refresh logic
            if time_since_refresh >= refresh_rate:
                st.session_state.last_refresh = datetime.now()
                st.rerun()
        
        # Manual refresh button
        if st.sidebar.button("🔄 Refresh Now"):
            st.session_state.last_refresh = datetime.now()
            st.rerun()
    
    def _render_overview_tab(self):
        """Render enhanced system overview tab with market data and charts"""
        st.header("📊 Market Overview & System Status")
        
        # Top section: System metrics and market overview
        col1, col2 = st.columns([2, 1])
        
        with col1:
            # System Performance Metrics
            st.subheader("🚀 System Performance")
            try:
                status = self._get_system_status()
                performance = self._get_performance_metrics()
                
                if status and performance:
                    metric_col1, metric_col2, metric_col3, metric_col4 = st.columns(4)
                    
                    with metric_col1:
                        st.metric(
                            "Total Equity",
                            f"${status['total_equity']:,.2f}",
                            delta=f"${status['daily_pnl']:,.2f}"
                        )
                    
                    with metric_col2:
                        st.metric(
                            "Total Return",
                            f"{performance['total_return_pct']:.2%}",
                            delta=f"{performance['daily_return_pct']:.2%}"
                        )
                    
                    with metric_col3:
                        st.metric(
                            "Sharpe Ratio",
                            f"{performance['sharpe_ratio']:.2f}"
                        )
                    
                    with metric_col4:
                        st.metric(
                            "Open Positions",
                            status['open_positions'],
                            delta=f"{status['positions_change']}"
                        )
                        
                    # Additional trading metrics
                    st.subheader("Trading Activity")
                    activity_col1, activity_col2, activity_col3 = st.columns(3)
                    
                    with activity_col1:
                        st.metric("Signals Processed", status['signals_processed'])
                    
                    with activity_col2:
                        st.metric("Orders Filled", status['orders_filled'])
                    
                    with activity_col3:
                        st.metric("Win Rate", f"{performance['win_rate']:.2%}")
                
                else:
                    st.error("Unable to load system data")
            
            except Exception as e:
                st.error(f"❌ Error loading system metrics: {e}")
        
        with col2:
            # Market Indices
            st.subheader("📈 Market Indices")
            self._render_market_indices()
        
        st.markdown("---")
        
        # Middle section: Latest Headlines and Watchlist
        col3, col4 = st.columns([3, 2])
        
        with col3:
            st.subheader("📰 Latest Market Headlines")
            self._render_latest_headlines()
        
        with col4:
            st.subheader("📈 Market Indices")
            self._render_market_indices()
        
        st.markdown("---")
        
        # Bottom section: Interactive Ticker Charts
        st.subheader("📊 Interactive Ticker Analysis")
        self._render_ticker_charts()
    
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
            # Risk metrics
            risk_metrics = self._get_risk_metrics()
            if risk_metrics:
                st.subheader("Risk Metrics")
                
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    var_1d_color = "inverse" if abs(risk_metrics['var_1d']) > 0.02 else "normal"
                    st.metric("VaR (1-day)", f"{risk_metrics['var_1d']:.2%}", 
                             delta_color=var_1d_color)
                
                with col2:
                    var_5d_color = "inverse" if abs(risk_metrics['var_5d']) > 0.05 else "normal"
                    st.metric("VaR (5-day)", f"{risk_metrics['var_5d']:.2%}", 
                             delta_color=var_5d_color)
                
                with col3:
                    drawdown_color = "inverse" if abs(risk_metrics['max_drawdown']) > 0.10 else "normal"
                    st.metric("Max Drawdown", f"{risk_metrics['max_drawdown']:.2%}", 
                             delta_color=drawdown_color)
                
                with col4:
                    vol_color = "inverse" if risk_metrics['volatility'] > 0.25 else "normal"
                    st.metric("Volatility", f"{risk_metrics['volatility']:.2%}", 
                             delta_color=vol_color)
                
                # Risk alerts
                risk_alerts = self._get_risk_alerts()
                if risk_alerts:
                    st.subheader("🚨 Risk Alerts")
                    for alert in risk_alerts:
                        severity_color = {
                            'LOW': 'info',
                            'MEDIUM': 'warning', 
                            'HIGH': 'error',
                            'CRITICAL': 'error'
                        }.get(alert['severity'], 'info')
                        
                        if severity_color == 'error':
                            st.error(f"**{alert['alert_type']}**: {alert['message']}")
                        elif severity_color == 'warning':
                            st.warning(f"**{alert['alert_type']}**: {alert['message']}")
                        else:
                            st.info(f"**{alert['alert_type']}**: {alert['message']}")
        
        except Exception as e:
            st.error(f"Error loading performance data: {str(e)}")
    
    def _render_charts_watchlist_tab(self):
        """Render combined charts and watchlist tab"""
        st.header("📈 Charts & Watchlist")
        
        # Split into two columns
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.subheader("📊 Interactive Charts")
            self._render_ticker_charts()
        
        with col2:
            st.subheader("📋 Global Watchlist")
            st.caption(f"Last updated: {datetime.now().strftime('%H:%M:%S')}")
            
            # Add filters for global stocks
            filter_col1, filter_col2 = st.columns(2)
            
            with filter_col1:
                sector_filter = st.selectbox(
                    "Filter by Sector",
                    ["All", "Technology", "Financial Services", "Healthcare", "Energy", "Consumer Discretionary", "Consumer Staples", "ETF"],
                    key="sector_filter"
                )
            
            with filter_col2:
                region_filter = st.selectbox(
                    "Filter by Region",
                    ["All", "US", "Europe", "Asia", "Global ETFs"],
                    key="region_filter"
                )
            
            self._render_active_watchlist(sector_filter, region_filter)
    
    def _render_signals_tab(self):
        """Render live signals tab with enhanced readability"""
        st.header("🎯 Live Trading Signals")
        
        # Signal filters in a cleaner layout
        st.markdown("### 🔍 Filter Signals")
        filter_col1, filter_col2, filter_col3 = st.columns([2, 2, 2])
        
        with filter_col1:
            signal_type_filter = st.selectbox(
                "Signal Type",
                ["All", "BUY", "SELL", "HOLD"],
                help="Filter signals by type"
            )
        
        with filter_col2:
            min_confidence = st.slider(
                "Minimum Confidence",
                min_value=0.0,
                max_value=1.0,
                value=0.5,
                step=0.05,
                format="%d%%",
                help="Filter signals by minimum confidence level"
            )
        
        with filter_col3:
            signal_limit = st.selectbox(
                "Show Last N Signals",
                [10, 25, 50, 100],
                index=1,
                help="Number of recent signals to display"
            )
        
        # Get and filter signals
        signals = self._get_recent_signals(limit=signal_limit)
        
        if signals:
            # Filter signals
            filtered_signals = [
                signal for signal in signals
                if (signal_type_filter == "All" or signal.get('signal_type') == signal_type_filter)
                and signal.get('confidence', 0) >= min_confidence
            ]
            
            if filtered_signals:
                # Signal statistics
                st.markdown("### 📊 Signal Summary")
                stats_col1, stats_col2, stats_col3, stats_col4 = st.columns(4)
                
                with stats_col1:
                    st.metric(
                        "Total Signals",
                        len(filtered_signals),
                        help="Number of signals matching current filters"
                    )
                
                with stats_col2:
                    buy_signals = sum(1 for s in filtered_signals if s.get('signal_type') == 'BUY')
                    st.metric(
                        "Buy Signals",
                        buy_signals,
                        help="Number of BUY signals"
                    )
                
                with stats_col3:
                    sell_signals = sum(1 for s in filtered_signals if s.get('signal_type') == 'SELL')
                    st.metric(
                        "Sell Signals",
                        sell_signals,
                        help="Number of SELL signals"
                    )
                
                with stats_col4:
                    avg_confidence = sum(s.get('confidence', 0) for s in filtered_signals) / len(filtered_signals)
                    st.metric(
                        "Avg Confidence",
                        f"{avg_confidence:.1%}",
                        help="Average confidence across all signals"
                    )
                
                # Display signals in an organized layout
                st.markdown("### 📈 Latest Signals")
                
                for signal in filtered_signals:
                    signal_type = signal.get('signal_type', 'UNKNOWN')
                    confidence = signal.get('confidence', 0)
                    ticker = signal.get('ticker', 'N/A')
                    timestamp = signal.get('timestamp', '')
                    
                    # Color coding for different signal types
                    colors = {
                        'BUY': ['#10b981', '#dcfce7'],  # Green
                        'SELL': ['#ef4444', '#fee2e2'],  # Red
                        'HOLD': ['#f59e0b', '#fef3c7'],  # Yellow
                        'UNKNOWN': ['#6b7280', '#f3f4f6']  # Gray
                    }
                    
                    text_color, bg_color = colors.get(signal_type, colors['UNKNOWN'])
                    
                    # Enhanced signal card with better visual hierarchy
                    st.markdown(f"""
                    <div style="
                        background-color: {bg_color};
                        border-left: 4px solid {text_color};
                        border-radius: 8px;
                        padding: 1rem;
                        margin: 0.5rem 0;
                        box-shadow: 0 2px 4px rgba(0,0,0,0.05);
                    ">
                        <div style="display: flex; justify-content: space-between; align-items: center;">
                            <div>
                                <span style="
                                    background-color: {text_color};
                                    color: white;
                                    padding: 0.25rem 0.75rem;
                                    border-radius: 9999px;
                                    font-weight: 600;
                                    font-size: 0.875rem;
                                ">{signal_type}</span>
                                <span style="
                                    font-size: 1.125rem;
                                    font-weight: 600;
                                    margin-left: 0.75rem;
                                ">{ticker}</span>
                            </div>
                            <div style="
                                color: #4b5563;
                                font-size: 0.875rem;
                            ">
                                {timestamp[:19]}
                            </div>
                        </div>
                        <div style="
                            margin-top: 0.5rem;
                            display: flex;
                            align-items: center;
                            gap: 0.5rem;
                            color: #4b5563;
                        ">
                            <span>Confidence:</span>
                            <div style="
                                background-color: white;
                                border-radius: 9999px;
                                height: 6px;
                                width: 100px;
                                overflow: hidden;
                            ">
                                <div style="
                                    background-color: {text_color};
                                    width: {confidence * 100}%;
                                    height: 100%;
                                "></div>
                            </div>
                            <span style="font-weight: 600;">{confidence:.1%}</span>
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
            else:
                st.info("No signals match the current filters")
        else:
            st.info("No signals available")
        
        # Auto-refresh option
        st.markdown("### 🔄 Auto-Refresh")
        auto_refresh = st.checkbox(
            "Enable auto-refresh",
            value=True,
            help="Automatically refresh signals every few seconds"
        )
        
        if auto_refresh:
            time.sleep(5)  # Refresh every 5 seconds
            st.rerun()
    
    def _render_news_tab(self):
        """Render dedicated news analysis tab"""
        st.header("📰 Market News & Analysis")
        
        # News controls
        col1, col2, col3 = st.columns(3)
        
        with col1:
            news_category = st.selectbox(
                "News Category",
                ["business", "technology", "general"],
                index=0
            )
        
        with col2:
            news_country = st.selectbox(
                "Country",
                ["us", "gb", "ca", "au"],
                index=0
            )
        
        with col3:
            if st.button("🔄 Refresh News", key="refresh_news_tab"):
                st.rerun()
        
        # Get news with current settings
        st.subheader(f"📈 Latest {news_category.title()} Headlines")
        
        try:
            # Get news headlines
            headlines = self._get_real_news_headlines_advanced(news_category, news_country)
            
            if headlines:
                # Display in a more detailed format
                for i, headline in enumerate(headlines[:15]):
                    with st.expander(f"📰 {headline['title'][:80]}{'...' if len(headline['title']) > 80 else ''}", expanded=False):
                        col_a, col_b = st.columns([3, 1])
                        
                        with col_a:
                            st.write(f"**Source:** {headline.get('source', 'Unknown')}")
                            st.write(f"**Published:** {headline.get('time', 'Unknown time')}")
                            
                            if headline.get('description'):
                                st.write("**Description:**")
                                st.write(headline['description'])
                            
                            if headline.get('url', '#') != '#':
                                st.markdown(f"[📖 Read Full Article]({headline['url']})")
                        
                        with col_b:
                            # Sentiment and impact indicators
                            sentiment = headline.get('sentiment', 'neutral').title()
                            impact = headline.get('impact', 'medium').title()
                            
                            sentiment_color = {
                                "Positive": "🟢",
                                "Negative": "🔴", 
                                "Neutral": "⚪"
                            }.get(sentiment, "⚪")
                            
                            impact_badge = {
                                "High": "🔥",
                                "Medium": "⚡",
                                "Low": "📊"
                            }.get(impact, "⚡")
                            
                            st.metric("Sentiment", f"{sentiment_color} {sentiment}")
                            st.metric("Impact", f"{impact_badge} {impact}")
            else:
                st.info("No news available. Check your NewsAPI configuration or internet connection.")
                st.markdown("**To enable real news:**")
                st.markdown("1. Get a free API key from [NewsAPI.org](https://newsapi.org/register)")
                st.markdown("2. Set the NEWS_API_KEY environment variable")
                st.markdown("3. Restart the dashboard")
        
        except Exception as e:
            st.error(f"Error loading news: {e}")
    
    def _get_real_news_headlines_advanced(self, category="business", country="us"):
        """Get news headlines with advanced filtering"""
        try:
            if not NEWSAPI_AVAILABLE:
                return None
                
            api_key = os.getenv('NEWS_API_KEY')
            if not api_key or api_key.strip() == '' or api_key == 'demo_key':
                return None
            
            newsapi = NewsApiClient(api_key=api_key)
            
            # Fetch headlines with specified category and country
            news_data = newsapi.get_top_headlines(
                category=category,
                language='en',
                country=country,
                page_size=50
            )
            
            headlines = []
            if news_data and news_data.get('articles'):
                for article in news_data['articles']:
                    # Calculate time ago
                    if article.get('publishedAt'):
                        pub_date = datetime.fromisoformat(article['publishedAt'].replace('Z', '+00:00'))
                        time_diff = datetime.now(pub_date.tzinfo) - pub_date
                        
                        if time_diff.days > 0:
                            time_str = f"{time_diff.days} days ago"
                        elif time_diff.seconds > 3600:
                            hours = time_diff.seconds // 3600
                            time_str = f"{hours} hours ago"
                        else:
                            minutes = time_diff.seconds // 60
                            time_str = f"{minutes} minutes ago"
                    else:
                        time_str = "Unknown time"
                    
                    # Enhanced sentiment analysis - fix None concatenation
                    title = article.get('title') or ''
                    description = article.get('description') or ''
                    title_lower = (title + ' ' + description).lower()
                    positive_words = ['surge', 'gain', 'rise', 'up', 'positive', 'growth', 'profit', 'bull', 'rally', 'soar', 'jump']
                    negative_words = ['fall', 'drop', 'down', 'negative', 'loss', 'decline', 'bear', 'crash', 'plunge', 'dive']
                    
                    positive_count = sum(word in title_lower for word in positive_words)
                    negative_count = sum(word in title_lower for word in negative_words)
                    
                    if positive_count > negative_count:
                        sentiment = 'positive'
                    elif negative_count > positive_count:
                        sentiment = 'negative'
                    else:
                        sentiment = 'neutral'
                    
                    # Enhanced impact analysis - handle None source
                    source_obj = article.get('source')
                    if isinstance(source_obj, dict):
                        source_name = source_obj.get('name', 'Unknown')
                    else:
                        source_name = str(source_obj) if source_obj else 'Unknown'
                    
                    high_impact_sources = ['reuters', 'bloomberg', 'wall street journal', 'financial times', 'cnbc', 'marketwatch']
                    high_impact_keywords = ['fed', 'federal reserve', 'rate', 'inflation', 'gdp', 'unemployment', 'earnings', 'ipo']
                    
                    if any(word in title_lower for word in high_impact_keywords):
                        impact = 'high'
                    elif source_name.lower() in high_impact_sources:
                        impact = 'medium'
                    else:
                        impact = 'low'
                    
                    headlines.append({
                        'title': article.get('title', 'No title'),
                        'description': article.get('description', ''),
                        'source': source_name,
                        'time': time_str,
                        'sentiment': sentiment,
                        'impact': impact,
                        'url': article.get('url', '#')
                    })
            
            return headlines
            
        except Exception as e:
            logger.error(f"Error fetching advanced news: {e}")
            return None
    
    def _render_filters_tab(self):
        """Render filter management tab"""
        st.header("🔧 Filter Management")
        
        # Filter creation
        st.subheader("Create New Filter")
        
        with st.form("filter_form"):
            filter_name = st.text_input("Filter Name")
            filter_type = st.selectbox("Filter Type", ["News", "Price", "Volume", "Sentiment"])
            
            # Dynamic filter fields based on type
            if filter_type == "News":
                field = st.selectbox("Field", ["relevance_score", "sentiment_score", "ticker"])
                operator = st.selectbox("Operator", ["gt", "lt", "eq", "contains"])
                value = st.text_input("Value")
            
            elif filter_type == "Price":
                field = st.selectbox("Field", ["price", "price_change", "volume"])
                operator = st.selectbox("Operator", ["gt", "lt", "gte", "lte"])
                value = st.number_input("Value", value=0.0)
            
            elif filter_type == "Volume":
                field = st.selectbox("Field", ["volume", "volume_ratio"])
                operator = st.selectbox("Operator", ["gt", "lt", "gte", "lte"])
                value = st.number_input("Value", value=0.0)
            
            elif filter_type == "Sentiment":
                field = st.selectbox("Field", ["sentiment_score"])
                operator = st.selectbox("Operator", ["gt", "lt", "gte", "lte"])
                value = st.slider("Sentiment Score", -1.0, 1.0, 0.0, 0.1)
            
            if st.form_submit_button("Create Filter"):
                filter_config = {
                    "field": field,
                    "operator": operator,
                    "value": value
                }
                
                if self._create_filter(filter_name, filter_config):
                    st.success(f"Filter '{filter_name}' created successfully!")
                else:
                    st.error("Failed to create filter")
        
        # Existing filters
        st.subheader("Active Filters")
        
        filters = self._get_active_filters()
        if filters:
            for filter_name, filter_config in filters.items():
                with st.expander(f"🔍 {filter_name}"):
                    st.json(filter_config)
                    
                    col1, col2 = st.columns(2)
                    with col1:
                        if st.button(f"Edit {filter_name}"):
                            # TODO: Implement filter editing
                            st.info("Filter editing coming soon!")
                    
                    with col2:
                        if st.button(f"Delete {filter_name}"):
                            if self._delete_filter(filter_name):
                                st.success(f"Filter '{filter_name}' deleted!")
                                st.rerun()
        else:
            st.info("No active filters")
    
    def _get_system_status(self):
        """Get system status from API"""
        try:
            # Mock data for demo
            return {
                'total_equity': 150000.0,
                'daily_pnl': 2500.0,
                'open_positions': 8,
                'positions_change': "+2",
                'trading_enabled': True,
                'signals_processed': 145,
                'orders_filled': 23
            }
        except Exception as e:
            logger.error(f"Error getting system status: {e}")
            return None
    
    def _get_positions(self):
        """Get current positions"""
        try:
            # Mock data for demo
            return [
                {
                    'ticker': 'AAPL',
                    'quantity': 100,
                    'entry_price': 150.0,
                    'current_price': 155.0,
                    'unrealized_pnl': 500.0,
                    'timestamp': datetime.now().isoformat()
                },
                {
                    'ticker': 'MSFT',
                    'quantity': 50,
                    'entry_price': 300.0,
                    'current_price': 305.0,
                    'unrealized_pnl': 250.0,
                    'timestamp': datetime.now().isoformat()
                }
            ]
        except Exception as e:
            logger.error(f"Error getting positions: {e}")
            return []
    
    def _get_performance_metrics(self):
        """Get performance metrics"""
        try:
            # Mock data for demo
            return {
                'total_return_pct': 0.12,
                'daily_return_pct': 0.015,
                'sharpe_ratio': 1.45,
                'max_drawdown': 0.08,
                'win_rate': 0.65
            }
        except Exception as e:
            logger.error(f"Error getting performance metrics: {e}")
            return None
    
    def _get_risk_metrics(self):
        """Get risk metrics"""
        try:
            # Mock data for demo
            return {
                'var_1d': 0.025,
                'var_5d': 0.045,
                'max_drawdown': 0.08,
                'volatility': 0.18
            }
        except Exception as e:
            logger.error(f"Error getting risk metrics: {e}")
            return None
    
    def _get_risk_alerts(self):
        """Get risk alerts"""
        try:
            # Mock data for demo
            return []
        except Exception as e:
            logger.error(f"Error getting risk alerts: {e}")
            return []
    
    def _get_recent_signals(self, limit: int = 50):
        """Get recent trading signals"""
        try:
            # Mock data for demo
            signals = []
            for i in range(min(limit, 20)):
                signals.append({
                    'signal_id': f'sig_{i}',
                    'ticker': np.random.choice(['AAPL', 'MSFT', 'GOOGL', 'TSLA']),
                    'signal_type': np.random.choice(['BUY', 'SELL', 'HOLD']),
                    'confidence': np.random.uniform(0.5, 0.95),
                    'timestamp': (datetime.now() - timedelta(minutes=i*5)).isoformat()
                })
            return signals
        except Exception as e:
            logger.error(f"Error getting recent signals: {e}")
            return []
    
    def _get_watchlist(self):
        """Get current watchlist"""
        try:
            # Mock data for demo
            return ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA', 'META', 'NVDA']
        except Exception as e:
            logger.error(f"Error getting watchlist: {e}")
            return []
    
    def _get_active_filters(self):
        """Get active filters"""
        try:
            # Mock data for demo
            return {
                'high_confidence': {
                    'field': 'confidence',
                    'operator': 'gt',
                    'value': 0.7
                },
                'major_stocks': {
                    'field': 'ticker',
                    'operator': 'in',
                    'value': ['AAPL', 'MSFT', 'GOOGL']
                }
            }
        except Exception as e:
            logger.error(f"Error getting active filters: {e}")
            return {}
    
    def _create_filter(self, name: str, config: dict):
        """Create a new filter"""
        try:
            # TODO: Implement filter creation via API
            return True
        except Exception as e:
            logger.error(f"Error creating filter: {e}")
            return False
    
    def _delete_filter(self, name: str):
        """Delete a filter"""
        try:
            # TODO: Implement filter deletion via API
            return True
        except Exception as e:
            logger.error(f"Error deleting filter: {e}")
            return False
    
    def _start_trading(self):
        """Start trading"""
        st.success("Trading system started!")
    
    def _stop_trading(self):
        """Stop trading"""
        st.warning("Trading system stopped!")
    
    def _create_equity_curve_chart(self):
        """Create equity curve chart"""
        try:
            # Mock data for demo
            dates = pd.date_range(start='2024-01-01', periods=100, freq='D')
            equity = 100000 * (1 + np.cumsum(np.random.normal(0.001, 0.02, 100)))
            
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=dates,
                y=equity,
                mode='lines',
                name='Equity Curve',
                line=dict(color='#007bff', width=2)
            ))
            
            fig.update_layout(
                title="Portfolio Equity Curve",
                xaxis_title="Date",
                yaxis_title="Equity ($)",
                height=400
            )
            
            return fig
        except Exception as e:
            logger.error(f"Error creating equity curve: {e}")
            return None

    def _render_market_indices(self):
        """Render major market indices with enhanced styling"""
        try:
            # Mock market data - in production, this would come from real API
            indices = [
                {"symbol": "S&P 500", "value": 4185.47, "change": 12.35, "change_pct": 0.30},
                {"symbol": "NASDAQ", "value": 12845.78, "change": -45.23, "change_pct": -0.35},
                {"symbol": "DOW", "value": 33875.12, "change": 89.45, "change_pct": 0.26},
                {"symbol": "VIX", "value": 18.43, "change": 1.25, "change_pct": 6.78}
            ]
            
            for index in indices:
                # Enhanced styling with trend indicators
                if index["change"] >= 0:
                    trend_class = "metric-positive"
                    arrow = "↗"
                    trend_color = "#10b981"
                else:
                    trend_class = "metric-negative"
                    arrow = "↘" 
                    trend_color = "#ef4444"
                
                st.markdown(f"""
                <div class="trading-card {trend_class}">
                    <div style="display: flex; justify-content: space-between; align-items: center;">
                        <div>
                            <div style="font-size: 0.875rem; color: #94a3b8; font-weight: 500;">
                                {index['symbol']}
                            </div>
                            <div style="font-size: 1.5rem; font-weight: bold; color: #f8fafc;">
                                {index['value']:,.2f}
                            </div>
                        </div>
                        <div style="text-align: right;">
                            <div style="font-size: 1.5rem; color: {trend_color};">
                                {arrow}
                            </div>
                            <div style="font-size: 0.875rem; color: {trend_color}; font-weight: 600;">
                                {index['change']:+.2f} ({index['change_pct']:+.2f}%)
                            </div>
                        </div>
                    </div>
                    
                    <!-- Mini sparkline placeholder -->
                    <div style="height: 16px; background: linear-gradient(90deg, {trend_color}20, {trend_color}40, {trend_color}20); 
                         border-radius: 4px; margin-top: 8px; opacity: 0.6;">
                    </div>
                </div>
                """, unsafe_allow_html=True)
                
        except Exception as e:
            st.error(f"Error loading market indices: {e}")

    def _render_latest_headlines(self):
        """Render latest market headlines from NewsAPI"""
        try:
            headlines = self._get_real_news_headlines()
            
            if not headlines:
                # Fallback to enhanced mock data if API not available
                headlines = [
                    {
                        "title": "Fed Chair Powell Hints at Potential Rate Cut in Q2",
                        "source": "Reuters",
                        "time": "2 hours ago",
                        "sentiment": "Positive",
                        "impact": "High",
                        "url": "#"
                    },
                    {
                        "title": "Tesla Announces Record Q4 Deliveries",
                        "source": "CNBC",
                        "time": "4 hours ago", 
                        "sentiment": "Positive",
                        "impact": "Medium",
                        "url": "#"
                    },
                    {
                        "title": "Apple Suppliers Face Supply Chain Disruption",
                        "source": "Bloomberg",
                        "time": "6 hours ago",
                        "sentiment": "Negative", 
                        "impact": "Medium",
                        "url": "#"
                    },
                    {
                        "title": "Bitcoin Surges Past $45,000 on ETF Approval Hopes",
                        "source": "CoinDesk",
                        "time": "8 hours ago",
                        "sentiment": "Positive",
                        "impact": "High",
                        "url": "#"
                    },
                    {
                        "title": "Banking Sector Shows Resilience in Stress Tests",
                        "source": "Financial Times",
                        "time": "10 hours ago",
                        "sentiment": "Positive",
                        "impact": "Medium",
                        "url": "#"
                    }
                ]
            
            # Add refresh button
            if st.button("🔄 Refresh Headlines", key="refresh_headlines"):
                st.rerun()
            
            # Display headlines with enhanced formatting
            for i, headline in enumerate(headlines[:10]):  # Show top 10
                # Better sentiment color coding
                sentiment = headline.get("sentiment", "neutral").lower()
                if sentiment == "positive":
                    sentiment_color = "#00C851"
                    sentiment_bg = "#e8f5e8"
                    sentiment_icon = "📈"
                elif sentiment == "negative":
                    sentiment_color = "#ff4444"
                    sentiment_bg = "#ffe8e8"
                    sentiment_icon = "📉"
                else:
                    sentiment_color = "#ffa500"
                    sentiment_bg = "#fff8e8"
                    sentiment_icon = "📊"
                
                # Impact badge colors
                impact = headline.get("impact", "medium").lower()
                if impact == "high":
                    impact_color = "#ff4444"
                    impact_icon = "🔥"
                elif impact == "medium":
                    impact_color = "#ffa500"
                    impact_icon = "⚡"
                else:
                    impact_color = "#666"
                    impact_icon = "📊"
                
                # Format time
                time_str = headline.get("time", "Unknown time")
                
                st.markdown(f"""
                <div style="
                    margin-bottom: 12px; 
                    padding: 16px; 
                    border-radius: 8px; 
                    background: white; 
                    border: 1px solid #e0e0e0;
                    box-shadow: 0 2px 4px rgba(0,0,0,0.1);
                    transition: all 0.3s ease;
                " onmouseover="this.style.boxShadow='0 4px 8px rgba(0,0,0,0.15)'" onmouseout="this.style.boxShadow='0 2px 4px rgba(0,0,0,0.1)'">
                    <div style="display: flex; justify-content: between; align-items: flex-start; margin-bottom: 8px;">
                        <div style="flex: 1;">
                            <strong style="font-size: 16px; line-height: 1.4;">
                                <a href="{headline.get('url', '#')}" target="_blank" 
                                   style="text-decoration: none; color: #333; hover: color: #1f77b4;">
                                    {headline['title'][:120]}{'...' if len(headline['title']) > 120 else ''}
                                </a>
                            </strong>
                        </div>
                        <div style="margin-left: 12px; text-align: center;">
                            <div style="
                                background: {sentiment_bg}; 
                                color: {sentiment_color}; 
                                padding: 4px 8px; 
                                border-radius: 4px; 
                                font-size: 12px; 
                                font-weight: bold;
                                margin-bottom: 4px;
                            ">
                                {sentiment_icon} {headline.get('sentiment', 'Neutral').title()}
                            </div>
                            <div style="color: {impact_color}; font-size: 14px;">
                                {impact_icon}
                            </div>
                        </div>
                    </div>
                    <div style="display: flex; justify-content: space-between; align-items: center;">
                        <small style="color: #666; font-size: 13px;">
                            <strong>{headline.get('source', 'Unknown')}</strong> • {time_str}
                        </small>
                        <small style="color: {impact_color}; font-weight: bold; font-size: 12px;">
                            {headline.get('impact', 'Medium').title()} Impact
                        </small>
                    </div>
                </div>
                """, unsafe_allow_html=True)
                
        except Exception as e:
            st.error(f"Error loading headlines: {e}")
            st.info("Using fallback news data. Please check your NewsAPI configuration.")
    
    def _get_real_news_headlines(self):
        """Fetch real news headlines from NewsAPI"""
        try:
            # Check if NewsAPI is available and configured
            if not NEWSAPI_AVAILABLE:
                return None
                
            api_key = os.getenv('NEWS_API_KEY')
            if not api_key or api_key.strip() == '':
                return None
            
            # Initialize NewsAPI client
            newsapi = NewsApiClient(api_key=api_key)
            
            # Fetch business headlines
            news_data = newsapi.get_top_headlines(
                category='business',
                language='en',
                country='us',
                page_size=20
            )
            
            headlines = []
            if news_data and news_data.get('articles'):
                for article in news_data['articles']:
                    # Calculate time ago
                    pub_date = datetime.fromisoformat(article['publishedAt'].replace('Z', '+00:00'))
                    time_diff = datetime.now(pub_date.tzinfo) - pub_date
                    
                    if time_diff.days > 0:
                        time_str = f"{time_diff.days} days ago"
                    elif time_diff.seconds > 3600:
                        hours = time_diff.seconds // 3600
                        time_str = f"{hours} hours ago"
                    else:
                        minutes = time_diff.seconds // 60
                        time_str = f"{minutes} minutes ago"
                    
                    # Simple sentiment analysis based on keywords
                    title_lower = article['title'].lower()
                    if any(word in title_lower for word in ['surge', 'gain', 'rise', 'up', 'positive', 'growth', 'profit']):
                        sentiment = 'positive'
                    elif any(word in title_lower for word in ['fall', 'drop', 'down', 'negative', 'loss', 'decline']):
                        sentiment = 'negative'
                    else:
                        sentiment = 'neutral'
                    
                    # Determine impact based on source and keywords
                    source_name = article['source']['name']
                    if any(word in title_lower for word in ['fed', 'federal', 'rate', 'inflation', 'gdp', 'unemployment']):
                        impact = 'high'
                    elif source_name.lower() in ['reuters', 'bloomberg', 'wall street journal', 'financial times']:
                        impact = 'medium'
                    else:
                        impact = 'low'
                    
                    headlines.append({
                        'title': article['title'],
                        'source': source_name,
                        'time': time_str,
                        'sentiment': sentiment,
                        'impact': impact,
                        'url': article['url']
                    })
            
            return headlines
            
        except Exception as e:
            st.error(f"Error fetching news from API: {e}")
            return None

    def _render_active_watchlist(self, sector_filter="All", region_filter="All"):
        """Render enhanced interactive watchlist with real market data and filters"""
        try:
            # Add refresh button with better styling
            col1, col2 = st.columns([3, 1])
            with col2:
                if st.button("🔄 Refresh", key="refresh_watchlist", help="Refresh market data"):
                    st.rerun()
            
            # Show loading state
            with st.spinner("Loading market data..."):
                watchlist = self._get_real_watchlist_data()
            
            if not watchlist:
                # Show skeleton loaders
                st.markdown("**Loading prices...**")
                for i in range(6):
                    st.markdown(f'<div class="skeleton" style="margin: 8px 0; height: 60px;"></div>', unsafe_allow_html=True)
                
                # Fallback to global mock data after timeout
                watchlist = [
                    {"symbol": "AAPL", "price": 178.25, "change": 2.45, "change_pct": 1.39, "volume": 73114100, "sector": "Technology", "country": "US"},
                    {"symbol": "MSFT", "price": 378.85, "change": -1.23, "change_pct": -0.32, "volume": 25847200, "sector": "Technology", "country": "US"},
                    {"symbol": "TSM", "price": 98.45, "change": 1.87, "change_pct": 1.93, "volume": 18934500, "sector": "Technology", "country": "Taiwan"},
                    {"symbol": "ASML", "price": 645.32, "change": -8.76, "change_pct": -1.34, "volume": 1234500, "sector": "Technology", "country": "Netherlands"},
                    {"symbol": "NVDA", "price": 478.23, "change": -8.45, "change_pct": -1.74, "volume": 45123800, "sector": "Technology", "country": "US"},
                    {"symbol": "BABA", "price": 87.34, "change": 3.21, "change_pct": 3.82, "volume": 32456700, "sector": "Consumer Discretionary", "country": "China"},
                    {"symbol": "JPM", "price": 145.67, "change": 0.89, "change_pct": 0.62, "volume": 12345600, "sector": "Financial Services", "country": "US"},
                    {"symbol": "NESN.SW", "price": 108.90, "change": -0.45, "change_pct": -0.41, "volume": 2345600, "sector": "Consumer Staples", "country": "Switzerland"},
                    {"symbol": "TM", "price": 178.45, "change": 2.34, "change_pct": 1.33, "volume": 5678900, "sector": "Consumer Discretionary", "country": "Japan"},
                    {"symbol": "SPY", "price": 418.67, "change": 1.23, "change_pct": 0.29, "volume": 89012300, "sector": "ETF", "country": "US"}
                ]
            
            # Apply filters
            filtered_watchlist = []
            for stock in watchlist:
                # Apply sector filter
                if sector_filter != "All" and stock.get('sector', 'Unknown') != sector_filter:
                    continue
                
                # Apply region filter
                country = stock.get('country', 'Unknown')
                if region_filter != "All":
                    if region_filter == "US" and country != "US":
                        continue
                    elif region_filter == "Europe" and country not in ["Netherlands", "Switzerland", "Germany", "France", "UK"]:
                        continue
                    elif region_filter == "Asia" and country not in ["China", "Taiwan", "Japan", "South Korea", "India"]:
                        continue
                    elif region_filter == "Global ETFs" and stock.get('sector', 'Unknown') != "ETF":
                        continue
                
                filtered_watchlist.append(stock)
            
            # Show filter results and global summary
            if len(filtered_watchlist) != len(watchlist):
                st.info(f"Showing {len(filtered_watchlist)} of {len(watchlist)} stocks")
            
            # Global market summary
            if filtered_watchlist:
                gainers = [s for s in filtered_watchlist if s['change'] > 0]
                losers = [s for s in filtered_watchlist if s['change'] < 0]
                
                summary_col1, summary_col2, summary_col3 = st.columns(3)
                with summary_col1:
                    st.metric("📈 Gainers", len(gainers), f"{len(gainers)/len(filtered_watchlist)*100:.1f}%")
                with summary_col2:
                    st.metric("📉 Losers", len(losers), f"{len(losers)/len(filtered_watchlist)*100:.1f}%")
                with summary_col3:
                    avg_change = sum(s['change_pct'] for s in filtered_watchlist) / len(filtered_watchlist)
                    st.metric("📊 Avg Change", f"{avg_change:+.2f}%")
            
            # Enhanced watchlist display
            for i, stock in enumerate(filtered_watchlist):
                # Accessible color coding with shapes
                if stock["change"] >= 0:
                    trend_class = "metric-positive"
                    arrow = "↗"
                    trend_color = "#10b981"
                else:
                    trend_class = "metric-negative" 
                    arrow = "↘"
                    trend_color = "#ef4444"
                
                volume_str = f"{stock.get('volume', 0):,}" if stock.get('volume') else "N/A"
                sector = stock.get('sector', 'Unknown')
                country = stock.get('country', 'Unknown')
                
                # Country flag mapping
                country_flags = {
                    'US': '🇺🇸', 'China': '🇨🇳', 'Taiwan': '🇹🇼', 'Japan': '🇯🇵', 
                    'Netherlands': '🇳🇱', 'Switzerland': '🇨🇭', 'Germany': '🇩🇪',
                    'France': '🇫🇷', 'UK': '🇬🇧', 'South Korea': '🇰🇷', 
                    'Canada': '🇨🇦', 'Australia': '🇦🇺', 'India': '🇮🇳'
                }
                flag = country_flags.get(country, '🌍')
                
                # Interactive watchlist item with global info
                st.markdown(f"""
                <div class="watchlist-item {trend_class}" onclick="console.log('Clicked {stock['symbol']}')">
                    <div style="display: flex; justify-content: space-between; align-items: center;">
                        <div style="flex: 1;">
                            <div style="display: flex; align-items: center; gap: 8px;">
                                <span style="font-size: 1.2rem; font-weight: bold; color: #f8fafc;">
                                    {stock['symbol']}
                                </span>
                                <span style="font-size: 1rem;">{flag}</span>
                                <span style="font-size: 1.5rem; color: {trend_color};">
                                    {arrow}
                                </span>
                            </div>
                            <div style="font-size: 0.7rem; color: #94a3b8; margin-top: 2px;">
                                {sector[:15]}{'...' if len(sector) > 15 else ''} • Vol: {volume_str}
                            </div>
                        </div>
                        <div style="text-align: right;">
                            <div style="font-size: 1.1rem; font-weight: bold; color: #f8fafc;">
                                ${stock['price']:.2f}
                            </div>
                            <div style="font-size: 0.875rem; color: {trend_color}; font-weight: 600;">
                                {stock['change']:+.2f} ({stock['change_pct']:+.2f}%)
                            </div>
                        </div>
                    </div>
                </div>
                """, unsafe_allow_html=True)
                
                # Add mini sparkline placeholder for future enhancement
                if i < 3:  # Show sparklines for top 3 stocks
                    st.markdown(f"""
                    <div style="height: 20px; background: linear-gradient(90deg, {trend_color}20, {trend_color}40, {trend_color}20); 
                         border-radius: 4px; margin: 4px 0; opacity: 0.6;">
                    </div>
                    """, unsafe_allow_html=True)
                
        except Exception as e:
            st.error(f"Error loading watchlist: {e}")
            # Show error state with skeleton
            for i in range(3):
                st.markdown(f'<div class="skeleton" style="margin: 8px 0; height: 60px;"></div>', unsafe_allow_html=True)
    
    def _get_real_watchlist_data(self):
        """Get real watchlist data using yfinance for global stocks"""
        try:
            # Comprehensive global stock universe
            global_tickers = {
                # US Tech Giants
                "US_TECH": ["AAPL", "MSFT", "GOOGL", "AMZN", "META", "NVDA", "TSLA", "NFLX", "ADBE", "CRM"],
                
                # US Financial
                "US_FINANCE": ["JPM", "BAC", "WFC", "GS", "MS", "C", "BRK-B", "V", "MA", "AXP"],
                
                # US Healthcare & Pharma
                "US_HEALTH": ["JNJ", "PFE", "UNH", "ABBV", "MRK", "TMO", "ABT", "LLY", "MDT", "BMY"],
                
                # US Energy & Industrials
                "US_ENERGY": ["XOM", "CVX", "COP", "SLB", "EOG", "BA", "CAT", "GE", "MMM", "HON"],
                
                # European Stocks
                "EUROPE": ["ASML", "SAP", "NESN.SW", "ROCHE.SW", "NOVN.SW", "TM", "TSLA", "UL", "SHEL", "BP"],
                
                # Asian Stocks (ADRs and direct)
                "ASIA": ["TSM", "BABA", "JD", "NIO", "BIDU", "PDD", "TCEHY", "00700.HK", "2330.TW", "005930.KS"],
                
                # Crypto/Fintech
                "CRYPTO_FINTECH": ["COIN", "SQ", "PYPL", "HOOD", "SOFI", "MSTR", "RIOT", "MARA"],
                
                # ETFs for broader exposure
                "ETFS": ["SPY", "QQQ", "IWM", "VTI", "EFA", "EEM", "GLD", "SLV", "TLT", "HYG"]
            }
            
            # Flatten all tickers and take a sample for performance
            all_tickers = []
            for category, tickers in global_tickers.items():
                all_tickers.extend(tickers)
            
            # Limit to 30 stocks for performance, rotating selection
            import random
            random.seed(42)  # Consistent selection
            selected_tickers = random.sample(all_tickers, min(30, len(all_tickers)))
            
            watchlist = []
            
            # Show loading indicator
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            for i, ticker in enumerate(selected_tickers):
                try:
                    status_text.text(f"Loading {ticker}... ({i+1}/{len(selected_tickers)})")
                    progress_bar.progress((i + 1) / len(selected_tickers))
                    
                    stock = yf.Ticker(ticker)
                    # Get current and previous day data
                    hist = stock.history(period="5d")  # Get more days for reliability
                    
                    if len(hist) >= 2:
                        current_price = hist['Close'].iloc[-1]
                        prev_price = hist['Close'].iloc[-2]
                        change = current_price - prev_price
                        change_pct = (change / prev_price) * 100
                        
                        # Get additional info with error handling
                        try:
                            info = stock.info
                            market_cap = info.get('marketCap', 0)
                            sector = info.get('sector', 'Unknown')
                            country = info.get('country', 'Unknown')
                        except:
                            market_cap = 0
                            sector = 'Unknown'
                            country = 'Unknown'
                        
                        volume = hist['Volume'].iloc[-1] if len(hist) > 0 else 0
                        
                        watchlist.append({
                            "symbol": ticker,
                            "price": float(current_price),
                            "change": float(change),
                            "change_pct": float(change_pct),
                            "volume": int(volume),
                            "market_cap": market_cap,
                            "sector": sector,
                            "country": country
                        })
                except Exception as e:
                    logger.error(f"Error fetching data for {ticker}: {e}")
                    continue
            
            # Clear loading indicators
            progress_bar.empty()
            status_text.empty()
            
            # Sort by market cap (largest first)
            watchlist.sort(key=lambda x: x.get('market_cap', 0), reverse=True)
            
            return watchlist if watchlist else None
            
        except Exception as e:
            logger.error(f"Error fetching watchlist data: {e}")
            return None

    def _render_ticker_charts(self):
        """Render interactive ticker charts"""
        try:
            # Ticker selection
            col1, col2, col3 = st.columns([2, 1, 1])
            
            with col1:
                # Global ticker options organized by region
                global_ticker_options = [
                    # US Tech Giants
                    "AAPL", "MSFT", "GOOGL", "AMZN", "META", "NVDA", "TSLA", "NFLX", "ADBE", "CRM",
                    # US Finance
                    "JPM", "BAC", "WFC", "GS", "MS", "C", "BRK-B", "V", "MA", "AXP",
                    # US Healthcare
                    "JNJ", "PFE", "UNH", "ABBV", "MRK", "TMO", "ABT", "LLY", "MDT", "BMY",
                    # European Stocks
                    "ASML", "SAP", "NESN.SW", "ROCHE.SW", "NOVN.SW", "UL", "SHEL", "BP",
                    # Asian ADRs
                    "TSM", "BABA", "JD", "NIO", "BIDU", "PDD", "TCEHY",
                    # ETFs
                    "SPY", "QQQ", "IWM", "VTI", "EFA", "EEM", "GLD", "SLV", "TLT"
                ]
                
                selected_tickers = st.multiselect(
                    "Select Global Tickers to Display",
                    options=global_ticker_options,
                    default=["AAPL", "TSM", "ASML", "SPY"],
                    max_selections=4,
                    help="Choose up to 4 global stocks/ETFs to chart"
                )
            
            with col2:
                timeframe = st.selectbox(
                    "Timeframe",
                    options=["1D", "5D", "1M", "3M", "6M", "1Y"],
                    index=2
                )
            
            with col3:
                chart_type = st.selectbox(
                    "Chart Type",
                    options=["Candlestick", "Line", "Area"],
                    index=0
                )
            
            if selected_tickers:
                # Create charts in a 2x2 grid
                if len(selected_tickers) == 1:
                    st.plotly_chart(self._create_ticker_chart(selected_tickers[0], timeframe, chart_type), use_container_width=True)
                elif len(selected_tickers) == 2:
                    col1, col2 = st.columns(2)
                    with col1:
                        st.plotly_chart(self._create_ticker_chart(selected_tickers[0], timeframe, chart_type), use_container_width=True)
                    with col2:
                        st.plotly_chart(self._create_ticker_chart(selected_tickers[1], timeframe, chart_type), use_container_width=True)
                else:
                    # 3 or 4 tickers in 2x2 grid
                    col1, col2 = st.columns(2)
                    for i, ticker in enumerate(selected_tickers):
                        with col1 if i % 2 == 0 else col2:
                            st.plotly_chart(self._create_ticker_chart(ticker, timeframe, chart_type), use_container_width=True)
                
        except Exception as e:
            st.error(f"Error rendering ticker charts: {e}")

    def _create_ticker_chart(self, ticker, timeframe, chart_type):
        """Create individual ticker chart with real market data"""
        try:
            # Get real market data using yfinance
            df = self._get_real_market_data(ticker, timeframe)
            
            if df is None or df.empty:
                # Fallback to mock data if real data unavailable
                df = self._generate_mock_data(ticker, timeframe)
            
            fig = go.Figure()
            
            if chart_type == "Candlestick":
                fig.add_trace(go.Candlestick(
                    x=df['Date'],
                    open=df['Open'],
                    high=df['High'],
                    low=df['Low'],
                    close=df['Close'],
                    name=ticker
                ))
            elif chart_type == "Line":
                fig.add_trace(go.Scatter(
                    x=df['Date'],
                    y=df['Close'],
                    mode='lines',
                    name=ticker,
                    line=dict(width=2)
                ))
            else:  # Area
                fig.add_trace(go.Scatter(
                    x=df['Date'],
                    y=df['Close'],
                    mode='lines',
                    name=ticker,
                    fill='tonexty',
                    line=dict(width=2)
                ))
            
            # Calculate current price and change
            current_price = df['Close'].iloc[-1]
            prev_price = df['Close'].iloc[-2] if len(df) > 1 else current_price
            price_change = current_price - prev_price
            price_change_pct = (price_change / prev_price) * 100 if prev_price != 0 else 0
            
            # Color based on price change
            title_color = "green" if price_change >= 0 else "red"
            
            fig.update_layout(
                title=f"{ticker} - ${current_price:.2f} ({price_change:+.2f}, {price_change_pct:+.2f}%)",
                title_font_color=title_color,
                xaxis_title="Date",
                yaxis_title="Price ($)",
                height=400,
                showlegend=False,
                margin=dict(l=0, r=0, t=40, b=0)
            )
            
            return fig
            
        except Exception as e:
            logger.error(f"Error creating chart for {ticker}: {e}")
            return None
    
    def _get_real_market_data(self, ticker, timeframe):
        """Fetch real market data using yfinance"""
        try:
            # Map timeframe to yfinance period
            period_map = {
                "1D": "1d",
                "5D": "5d", 
                "1M": "1mo",
                "3M": "3mo",
                "6M": "6mo",
                "1Y": "1y"
            }
            
            period = period_map.get(timeframe, "1mo")
            
            # Fetch data from yfinance
            stock = yf.Ticker(ticker)
            hist = stock.history(period=period)
            
            if hist.empty:
                return None
            
            # Reset index to get Date as a column
            hist.reset_index(inplace=True)
            
            # Ensure we have the required columns and handle column mismatch
            expected_columns = ['Date', 'Open', 'High', 'Low', 'Close', 'Volume']
            
            if len(hist.columns) == len(expected_columns):
                hist.columns = expected_columns
            elif 'Date' not in hist.columns and len(hist.columns) >= 5:
                # If Date column is missing but we have OHLCV data
                hist.columns = expected_columns[:len(hist.columns)]
            
            # Verify we have the minimum required columns
            required_cols = ['Date', 'Open', 'High', 'Low', 'Close']
            if not all(col in hist.columns for col in required_cols):
                return None
            
            return hist
            
        except Exception as e:
            logger.error(f"Error fetching real data for {ticker}: {e}")
            return None
    
    def _generate_mock_data(self, ticker, timeframe):
        """Generate mock data as fallback"""
        try:
            np.random.seed(hash(ticker) % 1000)  # Consistent random data per ticker
            
            # Generate sample data based on timeframe
            if timeframe == "1D":
                periods = 78  # 1 day = 6.5 hours * 12 (5-min intervals)
                start_date = datetime.now() - timedelta(days=1)
                freq = 'H'
            elif timeframe == "5D":
                periods = 120  # 5 days * 24 hours
                start_date = datetime.now() - timedelta(days=5)
                freq = 'H'
            elif timeframe == "1M":
                periods = 30
                start_date = datetime.now() - timedelta(days=30)
                freq = 'D'
            elif timeframe == "3M":
                periods = 90
                start_date = datetime.now() - timedelta(days=90)
                freq = 'D'
            elif timeframe == "6M":
                periods = 180
                start_date = datetime.now() - timedelta(days=180)
                freq = 'D'
            else:  # 1Y
                periods = 252
                start_date = datetime.now() - timedelta(days=365)
                freq = 'D'
            
            dates = pd.date_range(start=start_date, periods=periods, freq=freq)
            
            # Generate realistic price data for global stocks
            base_price = {
                # US Tech
                "AAPL": 178, "MSFT": 378, "GOOGL": 142, "AMZN": 155, "META": 325, 
                "NVDA": 478, "TSLA": 238, "NFLX": 445, "ADBE": 589, "CRM": 245,
                # US Finance
                "JPM": 145, "BAC": 32, "WFC": 45, "GS": 385, "MS": 87, "C": 48,
                "BRK-B": 345, "V": 245, "MA": 378, "AXP": 156,
                # US Healthcare
                "JNJ": 165, "PFE": 35, "UNH": 489, "ABBV": 145, "MRK": 105,
                "TMO": 534, "ABT": 98, "LLY": 567, "MDT": 87, "BMY": 56,
                # European
                "ASML": 645, "SAP": 134, "NESN.SW": 109, "ROCHE.SW": 267, "NOVN.SW": 89,
                "UL": 56, "SHEL": 67, "BP": 34,
                # Asian ADRs
                "TSM": 98, "BABA": 87, "JD": 34, "NIO": 12, "BIDU": 123,
                "PDD": 145, "TCEHY": 45,
                # ETFs
                "SPY": 418, "QQQ": 384, "IWM": 187, "VTI": 234, "EFA": 67,
                "EEM": 45, "GLD": 187, "SLV": 23, "TLT": 98
            }.get(ticker, 100)
            
            returns = np.random.normal(0.0002, 0.02, periods)  # Returns
            prices = base_price * np.exp(np.cumsum(returns))
            
            # Add some noise for OHLC
            highs = prices * (1 + np.random.uniform(0, 0.02, periods))
            lows = prices * (1 - np.random.uniform(0, 0.02, periods))
            opens = np.roll(prices, 1)
            opens[0] = prices[0]
            
            df = pd.DataFrame({
                'Date': dates,
                'Open': opens,
                'High': highs,
                'Low': lows,
                'Close': prices,
                'Volume': np.random.randint(1000000, 10000000, periods)
            })
            
            return df
            
        except Exception as e:
            logger.error(f"Error generating mock data for {ticker}: {e}")
            return None

# Run the dashboard
if __name__ == "__main__":
    dashboard = TradingDashboard()
    dashboard.run() 