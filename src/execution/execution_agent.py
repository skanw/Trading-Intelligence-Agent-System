"""
Execution Agent for Live Trading

Handles real-time signal processing, order execution, and position management
for production trading operations.
"""

import asyncio
import json
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Set, Tuple
from dataclasses import dataclass, asdict
from enum import Enum
import redis

from ..risk_management.risk_manager import RiskManager
from .broker_interface import BrokerInterface, MockBroker

logger = logging.getLogger(__name__)

class OrderStatus(Enum):
    """Order status enumeration"""
    PENDING = "pending"
    SUBMITTED = "submitted"
    FILLED = "filled"
    CANCELLED = "cancelled"
    REJECTED = "rejected"

class OrderType(Enum):
    """Order type enumeration"""
    MARKET = "market"
    LIMIT = "limit"

@dataclass
class Order:
    """Represents a trading order"""
    id: str
    symbol: str
    side: str  # 'buy' or 'sell'
    quantity: float
    order_type: OrderType
    status: OrderStatus
    created_at: datetime
    price: Optional[float] = None
    filled_quantity: float = 0.0
    avg_fill_price: float = 0.0
    updated_at: Optional[datetime] = None
    broker_order_id: Optional[str] = None

class ExecutionAgent:
    """Live execution agent for production trading."""
    
    def __init__(self,
                 risk_manager: RiskManager,
                 redis_url: str = "redis://localhost:6379/0",
                 signal_stream: str = "signals"):
        
        self.risk_manager = risk_manager
        self.redis_client = redis.from_url(redis_url, decode_responses=True)
        self.signal_stream = signal_stream
        
        # Internal state
        self.running = False
        self.orders: Dict[str, Order] = {}
        self.last_signal_id = "0-0"
        
        # Performance tracking
        self.signals_processed = 0
        self.orders_submitted = 0
        self.orders_filled = 0
        
        logger.info("ExecutionAgent initialized")
    
    async def start(self):
        """Start the execution agent"""
        if self.running:
            return
        
        self.running = True
        logger.info("Starting ExecutionAgent")
        
        try:
            await self._run_execution_loop()
        except Exception as e:
            logger.error(f"ExecutionAgent error: {e}")
        finally:
            self.running = False
    
    async def stop(self):
        """Stop the execution agent"""
        logger.info("Stopping ExecutionAgent")
        self.running = False
    
    async def _run_execution_loop(self):
        """Main execution loop"""
        while self.running:
            try:
                # Read new signals from Redis stream
                signals = await self._read_new_signals()
                
                # Process each signal
                for signal_id, signal_data in signals:
                    await self._process_signal(signal_id, signal_data)
                
                # Small delay to prevent excessive CPU usage
                await asyncio.sleep(0.1)
                
            except Exception as e:
                logger.error(f"Error in execution loop: {e}")
                await asyncio.sleep(1.0)
    
    async def _read_new_signals(self) -> List[Tuple[str, Dict]]:
        """Read new signals from Redis stream"""
        try:
            # Use XREAD to get only new signals
            streams = {self.signal_stream: self.last_signal_id}
            response = self.redis_client.xread(streams, count=100, block=1000)
            
            signals = []
            for stream_name, stream_signals in response:
                for signal_id, signal_data in stream_signals:
                    signals.append((signal_id, signal_data))
                    self.last_signal_id = signal_id
            
            return signals
            
        except Exception as e:
            logger.error(f"Error reading signals: {e}")
            return []
    
    async def _process_signal(self, signal_id: str, signal_data: Dict):
        """Process a trading signal"""
        try:
            # Parse signal data
            symbol = signal_data.get('symbol')
            signal_type = signal_data.get('signal')
            confidence = float(signal_data.get('confidence', 0.0))
            
            if not symbol or not signal_type:
                return
            
            # Simulate getting current price (in production, get from market data feed)
            current_price = 100.0  # Placeholder
            
            # Process based on signal type
            if signal_type == 'buy':
                await self._handle_buy_signal(symbol, confidence, current_price)
            elif signal_type == 'sell':
                await self._handle_sell_signal(symbol, current_price)
            
            self.signals_processed += 1
            
        except Exception as e:
            logger.error(f"Error processing signal {signal_id}: {e}")
    
    async def _handle_buy_signal(self, symbol: str, confidence: float, price: float):
        """Handle buy signal"""
        try:
            # Check if position already exists
            if symbol in self.risk_manager.positions:
                return
            
            # Calculate position size
            position_size = self.risk_manager.calculate_position_size(
                symbol=symbol,
                signal_strength=confidence,
                price=price
            )
            
            if position_size <= 0:
                return
            
            # Simulate order execution (in production, submit to broker)
            success = self.risk_manager.open_position(
                symbol=symbol,
                side='long',
                quantity=position_size,
                price=price,
                timestamp=datetime.now()
            )
            
            if success:
                self.orders_filled += 1
                logger.info(f"Simulated buy: {position_size} shares of {symbol} at {price}")
                
        except Exception as e:
            logger.error(f"Error handling buy signal for {symbol}: {e}")
    
    async def _handle_sell_signal(self, symbol: str, price: float):
        """Handle sell signal"""
        try:
            # Check if position exists
            if symbol not in self.risk_manager.positions:
                return
            
            # Simulate order execution (in production, submit to broker)
            success = self.risk_manager.close_position(
                symbol=symbol,
                price=price,
                timestamp=datetime.now(),
                reason="signal"
            )
            
            if success:
                self.orders_filled += 1
                logger.info(f"Simulated sell: {symbol} at {price}")
                
        except Exception as e:
            logger.error(f"Error handling sell signal for {symbol}: {e}")
    
    def get_status(self) -> Dict:
        """Get current status of execution agent"""
        return {
            'running': self.running,
            'signals_processed': self.signals_processed,
            'orders_submitted': self.orders_submitted,
            'orders_filled': self.orders_filled,
            'open_positions': len(self.risk_manager.positions),
            'total_equity': self.risk_manager.get_total_equity()
        }
    
    def get_positions(self) -> List[Dict]:
        """Get current positions"""
        positions = []
        for symbol, position in self.risk_manager.positions.items():
            positions.append({
                'symbol': symbol,
                'side': position.side,
                'quantity': position.quantity,
                'entry_price': position.entry_price,
                'current_price': position.current_price,
                'unrealized_pnl': position.unrealized_pnl,
                'timestamp': position.timestamp.isoformat()
            })
        return positions 