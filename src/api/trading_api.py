"""
Trading API - FastAPI REST endpoints for TIAS

Provides HTTP endpoints for system monitoring, position management,
and trading operations.
"""

from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from typing import Dict, List, Optional, Any
from datetime import datetime, timedelta
import logging
import asyncio
import pandas as pd
import numpy as np

from ..risk_management.risk_manager import RiskManager
from ..execution.execution_agent import ExecutionAgent
from ..backtesting import DataLoader, BacktestEngine

logger = logging.getLogger(__name__)

# Pydantic models for request/response
class SystemStatus(BaseModel):
    """System status response model"""
    timestamp: datetime
    system_health: str
    trading_enabled: bool
    total_equity: float
    unrealized_pnl: float
    open_positions: int
    daily_pnl: float
    signals_processed: int
    orders_filled: int

class Position(BaseModel):
    """Position response model"""
    symbol: str
    side: str
    quantity: float
    entry_price: float
    current_price: float
    unrealized_pnl: float
    timestamp: datetime

class EquityCurvePoint(BaseModel):
    """Equity curve data point"""
    timestamp: datetime
    equity: float
    daily_return: Optional[float] = None

class PerformanceMetrics(BaseModel):
    """Performance metrics response model"""
    total_return: float
    total_return_pct: float
    sharpe_ratio: float
    max_drawdown: float
    volatility: float
    win_rate: float
    total_trades: int

class BacktestRequest(BaseModel):
    """Backtest request model"""
    symbols: List[str]
    start_date: datetime
    end_date: datetime
    initial_capital: float = 100000.0
    signal_threshold: float = 0.7

class TradingAPI:
    """
    Main trading API class providing REST endpoints for the TIAS system.
    """
    
    def __init__(self,
                 risk_manager: RiskManager,
                 execution_agent: Optional[ExecutionAgent] = None,
                 data_loader: Optional[DataLoader] = None):
        
        self.risk_manager = risk_manager
        self.execution_agent = execution_agent
        self.data_loader = data_loader or DataLoader()
        
        # Create FastAPI app
        self.app = FastAPI(
            title="TIAS Trading System API",
            description="Production Trading Intelligence Agent System API",
            version="1.0.0",
            docs_url="/docs",
            redoc_url="/redoc"
        )
        
        # Add CORS middleware
        self.app.add_middleware(
            CORSMiddleware,
            allow_origins=["*"],  # Configure appropriately for production
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"],
        )
        
        # Setup routes
        self._setup_routes()
        
        # Performance tracking
        self.start_time = datetime.now()
        
        logger.info("TradingAPI initialized")
    
    def _setup_routes(self):
        """Setup all API routes"""
        
        @self.app.get("/")
        async def root():
            """Root endpoint"""
            return {
                "message": "TIAS Trading System API",
                "version": "1.0.0",
                "timestamp": datetime.now().isoformat(),
                "uptime_seconds": (datetime.now() - self.start_time).total_seconds()
            }
        
        @self.app.get("/health")
        async def health_check():
            """Health check endpoint"""
            return {
                "status": "healthy",
                "timestamp": datetime.now().isoformat(),
                "system": "TIAS",
                "components": {
                    "risk_manager": "active",
                    "execution_agent": "active" if self.execution_agent else "inactive",
                    "data_loader": "active"
                }
            }
        
        @self.app.get("/status", response_model=SystemStatus)
        async def get_system_status():
            """Get current system status"""
            try:
                # Get basic metrics
                total_equity = self.risk_manager.get_total_equity()
                unrealized_pnl = sum(pos.unrealized_pnl for pos in self.risk_manager.positions.values())
                
                # Calculate daily P&L (simplified)
                daily_pnl = self._calculate_daily_pnl()
                
                # Get execution agent stats
                exec_stats = self.execution_agent.get_status() if self.execution_agent else {}
                
                return SystemStatus(
                    timestamp=datetime.now(),
                    system_health="healthy",
                    trading_enabled=self.execution_agent.running if self.execution_agent else False,
                    total_equity=total_equity,
                    unrealized_pnl=unrealized_pnl,
                    open_positions=len(self.risk_manager.positions),
                    daily_pnl=daily_pnl,
                    signals_processed=exec_stats.get('signals_processed', 0),
                    orders_filled=exec_stats.get('orders_filled', 0)
                )
                
            except Exception as e:
                logger.error(f"Error getting system status: {e}")
                raise HTTPException(status_code=500, detail=str(e))
        
        @self.app.get("/positions", response_model=List[Position])
        async def get_positions():
            """Get all current positions"""
            try:
                positions = []
                for symbol, pos in self.risk_manager.positions.items():
                    positions.append(Position(
                        symbol=symbol,
                        side=pos.side,
                        quantity=pos.quantity,
                        entry_price=pos.entry_price,
                        current_price=pos.current_price,
                        unrealized_pnl=pos.unrealized_pnl,
                        timestamp=pos.timestamp
                    ))
                
                return positions
                
            except Exception as e:
                logger.error(f"Error getting positions: {e}")
                raise HTTPException(status_code=500, detail=str(e))
        
        @self.app.get("/equity-curve", response_model=List[EquityCurvePoint])
        async def get_equity_curve(days: int = 30):
            """Get equity curve data"""
            try:
                # Get equity curve from risk manager
                equity_data = self.risk_manager.equity_curve
                
                if not equity_data:
                    return []
                
                # Filter to requested days
                cutoff_date = datetime.now() - timedelta(days=days)
                filtered_data = [(ts, equity) for ts, equity in equity_data if ts >= cutoff_date]
                
                # Calculate daily returns
                curve_points = []
                prev_equity = None
                
                for timestamp, equity in filtered_data:
                    daily_return = None
                    if prev_equity is not None:
                        daily_return = (equity - prev_equity) / prev_equity
                    
                    curve_points.append(EquityCurvePoint(
                        timestamp=timestamp,
                        equity=equity,
                        daily_return=daily_return
                    ))
                    
                    prev_equity = equity
                
                return curve_points
                
            except Exception as e:
                logger.error(f"Error getting equity curve: {e}")
                raise HTTPException(status_code=500, detail=str(e))
        
        @self.app.get("/performance", response_model=PerformanceMetrics)
        async def get_performance_metrics():
            """Get performance metrics"""
            try:
                risk_metrics = self.risk_manager.get_risk_metrics()
                
                # Calculate additional metrics
                returns = self._calculate_returns()
                win_rate = self._calculate_win_rate()
                total_trades = len(self.risk_manager.closed_positions)
                
                return PerformanceMetrics(
                    total_return=self.risk_manager.get_total_equity() - self.risk_manager.initial_capital,
                    total_return_pct=(self.risk_manager.get_total_equity() - self.risk_manager.initial_capital) / self.risk_manager.initial_capital,
                    sharpe_ratio=risk_metrics.sharpe_ratio,
                    max_drawdown=risk_metrics.max_drawdown,
                    volatility=risk_metrics.volatility,
                    win_rate=win_rate,
                    total_trades=total_trades
                )
                
            except Exception as e:
                logger.error(f"Error getting performance metrics: {e}")
                raise HTTPException(status_code=500, detail=str(e))
        
        @self.app.post("/backtest")
        async def run_backtest(request: BacktestRequest, background_tasks: BackgroundTasks):
            """Run a backtest"""
            try:
                # Validate request
                if request.start_date >= request.end_date:
                    raise HTTPException(status_code=400, detail="Start date must be before end date")
                
                if not request.symbols:
                    raise HTTPException(status_code=400, detail="At least one symbol required")
                
                # Run backtest in background
                background_tasks.add_task(
                    self._run_backtest_task,
                    request.symbols,
                    request.start_date,
                    request.end_date,
                    request.initial_capital,
                    request.signal_threshold
                )
                
                return {
                    "message": "Backtest started",
                    "symbols": request.symbols,
                    "start_date": request.start_date.isoformat(),
                    "end_date": request.end_date.isoformat(),
                    "status": "running"
                }
                
            except Exception as e:
                logger.error(f"Error starting backtest: {e}")
                raise HTTPException(status_code=500, detail=str(e))
        
        @self.app.post("/trading/start")
        async def start_trading():
            """Start live trading"""
            try:
                if not self.execution_agent:
                    raise HTTPException(status_code=400, detail="Execution agent not configured")
                
                if self.execution_agent.running:
                    return {"message": "Trading already active", "status": "running"}
                
                # Start execution agent in background
                asyncio.create_task(self.execution_agent.start())
                
                return {"message": "Trading started", "status": "starting"}
                
            except Exception as e:
                logger.error(f"Error starting trading: {e}")
                raise HTTPException(status_code=500, detail=str(e))
        
        @self.app.post("/trading/stop")
        async def stop_trading():
            """Stop live trading"""
            try:
                if not self.execution_agent:
                    raise HTTPException(status_code=400, detail="Execution agent not configured")
                
                if not self.execution_agent.running:
                    return {"message": "Trading already stopped", "status": "stopped"}
                
                await self.execution_agent.stop()
                
                return {"message": "Trading stopped", "status": "stopped"}
                
            except Exception as e:
                logger.error(f"Error stopping trading: {e}")
                raise HTTPException(status_code=500, detail=str(e))
        
        @self.app.get("/risk-metrics")
        async def get_risk_metrics():
            """Get current risk metrics"""
            try:
                risk_metrics = self.risk_manager.get_risk_metrics()
                
                return {
                    "portfolio_value": risk_metrics.portfolio_value,
                    "total_exposure": risk_metrics.total_exposure,
                    "leverage": risk_metrics.leverage,
                    "var_1d": risk_metrics.var_1d,
                    "max_drawdown": risk_metrics.max_drawdown,
                    "sharpe_ratio": risk_metrics.sharpe_ratio,
                    "volatility": risk_metrics.volatility,
                    "risk_level": risk_metrics.risk_level.value,
                    "timestamp": datetime.now().isoformat()
                }
                
            except Exception as e:
                logger.error(f"Error getting risk metrics: {e}")
                raise HTTPException(status_code=500, detail=str(e))
    
    def _calculate_daily_pnl(self) -> float:
        """Calculate daily P&L"""
        try:
            if not self.risk_manager.equity_curve:
                return 0.0
            
            # Get today's start and current equity
            today_start = datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)
            current_equity = self.risk_manager.get_total_equity()
            
            # Find equity at start of day
            start_equity = self.risk_manager.initial_capital
            for timestamp, equity in self.risk_manager.equity_curve:
                if timestamp >= today_start:
                    break
                start_equity = equity
            
            return current_equity - start_equity
            
        except Exception as e:
            logger.error(f"Error calculating daily P&L: {e}")
            return 0.0
    
    def _calculate_returns(self) -> List[float]:
        """Calculate returns from equity curve"""
        try:
            if len(self.risk_manager.equity_curve) < 2:
                return []
            
            equity_values = [equity for _, equity in self.risk_manager.equity_curve]
            returns = []
            
            for i in range(1, len(equity_values)):
                ret = (equity_values[i] - equity_values[i-1]) / equity_values[i-1]
                returns.append(ret)
            
            return returns
            
        except Exception as e:
            logger.error(f"Error calculating returns: {e}")
            return []
    
    def _calculate_win_rate(self) -> float:
        """Calculate win rate from closed positions"""
        try:
            closed_positions = self.risk_manager.closed_positions
            
            if not closed_positions:
                return 0.0
            
            winning_trades = 0
            for pos in closed_positions:
                if pos.side == 'long':
                    pnl = pos.quantity * (pos.current_price - pos.entry_price)
                else:
                    pnl = pos.quantity * (pos.entry_price - pos.current_price)
                
                if pnl > 0:
                    winning_trades += 1
            
            return winning_trades / len(closed_positions)
            
        except Exception as e:
            logger.error(f"Error calculating win rate: {e}")
            return 0.0
    
    async def _run_backtest_task(self,
                               symbols: List[str],
                               start_date: datetime,
                               end_date: datetime,
                               initial_capital: float,
                               signal_threshold: float):
        """Background task to run backtest"""
        try:
            logger.info(f"Starting backtest for {symbols} from {start_date} to {end_date}")
            
            # Create backtest engine
            backtest_engine = BacktestEngine(initial_capital=initial_capital)
            
            # Create fresh risk manager for backtest
            backtest_risk_manager = RiskManager(initial_capital=initial_capital)
            
            # Run backtest
            result = backtest_engine.run_backtest(
                symbols=symbols,
                start_date=start_date,
                end_date=end_date,
                data_loader=self.data_loader,
                risk_manager=backtest_risk_manager,
                signal_threshold=signal_threshold
            )
            
            # Store results (in production, save to database)
            logger.info(f"Backtest completed: {result.total_return_pct:.2%} return, "
                       f"{result.total_trades} trades, {result.sharpe_ratio:.2f} Sharpe")
            
        except Exception as e:
            logger.error(f"Error in backtest task: {e}")
    
    def get_app(self) -> FastAPI:
        """Get the FastAPI application instance"""
        return self.app 