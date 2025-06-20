"""
Execution Agent - Smart Order Routing and Trade Execution
"""

import asyncio
import uuid
from datetime import datetime, timezone, timedelta
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, asdict
from enum import Enum
import numpy as np

from .base_agent import BaseAgent, AgentMessage, AgentSignal
from ..config import config


class OrderType(Enum):
    MARKET = "MARKET"
    LIMIT = "LIMIT"
    STOP = "STOP"
    STOP_LIMIT = "STOP_LIMIT"
    TWAP = "TWAP"
    VWAP = "VWAP"
    ICEBERG = "ICEBERG"


class OrderStatus(Enum):
    PENDING = "PENDING"
    SUBMITTED = "SUBMITTED"
    PARTIALLY_FILLED = "PARTIALLY_FILLED"
    FILLED = "FILLED"
    CANCELLED = "CANCELLED"
    REJECTED = "REJECTED"
    EXPIRED = "EXPIRED"


class ExecutionVenue(Enum):
    NYSE = "NYSE"
    NASDAQ = "NASDAQ"
    BATS = "BATS"
    IEX = "IEX"
    DARK_POOL_1 = "DARK_POOL_1"
    DARK_POOL_2 = "DARK_POOL_2"


@dataclass
class Order:
    """Trading order"""
    order_id: str
    symbol: str
    side: str  # 'BUY' or 'SELL'
    quantity: int
    order_type: OrderType
    price: Optional[float]
    stop_price: Optional[float]
    venue: Optional[ExecutionVenue]
    time_in_force: str  # 'DAY', 'GTC', 'IOC', 'FOK'
    status: OrderStatus
    filled_quantity: int
    avg_fill_price: float
    commission: float
    created_time: datetime
    submitted_time: Optional[datetime]
    filled_time: Optional[datetime]
    execution_algorithm: Optional[str]
    parent_order_id: Optional[str]  # For child orders
    metadata: Dict[str, Any]
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class Fill:
    """Order fill/execution"""
    fill_id: str
    order_id: str
    symbol: str
    side: str
    quantity: int
    price: float
    venue: ExecutionVenue
    timestamp: datetime
    commission: float
    liquidity_flag: str  # 'MAKER', 'TAKER', 'UNKNOWN'
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class ExecutionMetrics:
    """Execution quality metrics"""
    symbol: str
    total_orders: int
    total_volume: int
    avg_fill_price: float
    vwap_benchmark: float
    slippage: float  # Basis points
    market_impact: float  # Basis points
    fill_rate: float  # Percentage
    avg_execution_time: float  # Seconds
    venue_breakdown: Dict[ExecutionVenue, Dict[str, float]]
    commission_rate: float
    timestamp: datetime


@dataclass
class ExecutionReport:
    """Comprehensive execution report"""
    period_start: datetime
    period_end: datetime
    total_orders: int
    total_volume: float
    total_commission: float
    avg_slippage: float
    best_performing_venue: ExecutionVenue
    worst_performing_venue: ExecutionVenue
    symbol_metrics: Dict[str, ExecutionMetrics]
    recommendations: List[str]
    timestamp: datetime


class ExecutionAgent(BaseAgent):
    """
    Execution Agent for smart order routing and trade execution
    """
    
    def __init__(self, agent_id: str = 'execution-agent', config_override: Optional[Dict] = None):
        super().__init__(agent_id, config_override)
        
        # Execution configuration
        self.venues = {
            ExecutionVenue.NYSE: {'fee': 0.0015, 'latency': 0.001, 'fill_rate': 0.95},
            ExecutionVenue.NASDAQ: {'fee': 0.0018, 'latency': 0.0012, 'fill_rate': 0.93},
            ExecutionVenue.BATS: {'fee': 0.0012, 'latency': 0.0008, 'fill_rate': 0.88},
            ExecutionVenue.IEX: {'fee': 0.0009, 'latency': 0.0015, 'fill_rate': 0.85},
            ExecutionVenue.DARK_POOL_1: {'fee': 0.0008, 'latency': 0.002, 'fill_rate': 0.70},
            ExecutionVenue.DARK_POOL_2: {'fee': 0.0010, 'latency': 0.0025, 'fill_rate': 0.65}
        }
        
        # Algorithm parameters
        self.twap_slice_minutes = 5
        self.vwap_lookback_days = 20
        self.iceberg_display_size = 0.1  # 10% of order size
        self.max_market_impact = 0.0050  # 50 basis points
        
        # Order management
        self.active_orders: Dict[str, Order] = {}
        self.order_history: List[Order] = []
        self.fills: List[Fill] = []
        
        # Execution tracking
        self.execution_metrics = {}
        self.venue_performance = {}
        
        # Risk limits
        self.max_order_size = 100000  # $100k
        self.max_daily_volume = 1000000  # $1M
        self.daily_volume_traded = 0.0
        
        # Override with config if available
        if hasattr(self.config, 'get') and self.config:
            self.max_order_size = self.config.get('max_order_size', self.max_order_size)
            self.max_daily_volume = self.config.get('max_daily_volume', self.max_daily_volume)
        
    async def agent_initialize(self):
        """Initialize execution agent"""
        try:
            # Initialize venue connections (simulated)
            await self._initialize_venues()
            
            # Load execution history
            await self._load_execution_history()
            
            self.logger.info("Execution Agent initialized successfully")
            
        except Exception as e:
            self.logger.error(f"Failed to initialize Execution Agent: {e}")
            raise
    
    async def execute(self):
        """Main execution logic - process orders and monitor execution"""
        try:
            # Process pending orders
            await self._process_pending_orders()
            
            # Update order statuses
            await self._update_order_statuses()
            
            # Monitor execution quality
            await self._monitor_execution_quality()
            
            # Generate execution reports
            await self._generate_execution_reports()
            
            # Clean up completed orders
            await self._cleanup_completed_orders()
            
            self.logger.info(f"Execution monitoring completed - {len(self.active_orders)} active orders")
            
        except Exception as e:
            self.logger.error(f"Error in execution agent: {e}")
            self.metrics['errors'] += 1
    
    async def submit_order(self, symbol: str, side: str, quantity: int, 
                         order_type: OrderType, price: Optional[float] = None,
                         algorithm: Optional[str] = None) -> Order:
        """Submit a trading order"""
        try:
            # Validate order
            if not await self._validate_order(symbol, side, quantity, price):
                raise ValueError("Order validation failed")
            
            # Generate order ID
            order_id = str(uuid.uuid4())
            
            # Determine optimal venue
            venue = await self._select_optimal_venue(symbol, side, quantity, order_type)
            
            # Create order
            order = Order(
                order_id=order_id,
                symbol=symbol,
                side=side,
                quantity=quantity,
                order_type=order_type,
                price=price,
                stop_price=None,
                venue=venue,
                time_in_force='DAY',
                status=OrderStatus.PENDING,
                filled_quantity=0,
                avg_fill_price=0.0,
                commission=0.0,
                created_time=datetime.now(timezone.utc),
                submitted_time=None,
                filled_time=None,
                execution_algorithm=algorithm,
                parent_order_id=None,
                metadata={'source_agent': 'execution-agent'}
            )
            
            # Add to active orders
            self.active_orders[order_id] = order
            
            # Route order based on algorithm
            if algorithm:
                await self._execute_algorithmic_order(order)
            else:
                await self._route_order(order)
            
            self.logger.info(f"Order submitted: {order_id} - {side} {quantity} {symbol}")
            
            return order
            
        except Exception as e:
            self.logger.error(f"Error submitting order: {e}")
            raise
    
    async def cancel_order(self, order_id: str) -> bool:
        """Cancel an active order"""
        try:
            if order_id not in self.active_orders:
                return False
            
            order = self.active_orders[order_id]
            
            if order.status in [OrderStatus.FILLED, OrderStatus.CANCELLED, OrderStatus.REJECTED]:
                return False
            
            # Simulate order cancellation
            order.status = OrderStatus.CANCELLED
            
            # Send cancellation notification
            await self._send_order_update(order)
            
            self.logger.info(f"Order cancelled: {order_id}")
            
            return True
            
        except Exception as e:
            self.logger.error(f"Error cancelling order {order_id}: {e}")
            return False
    
    async def _validate_order(self, symbol: str, side: str, quantity: int, price: Optional[float]) -> bool:
        """Validate order parameters"""
        try:
            # Check basic parameters
            if not symbol or side not in ['BUY', 'SELL'] or quantity <= 0:
                return False
            
            # Check order size limits
            estimated_value = quantity * (price or 100)  # Rough estimate
            if estimated_value > self.max_order_size:
                self.logger.warning(f"Order size ${estimated_value:,.0f} exceeds limit ${self.max_order_size:,.0f}")
                return False
            
            # Check daily volume limits
            if self.daily_volume_traded + estimated_value > self.max_daily_volume:
                self.logger.warning(f"Daily volume limit would be exceeded")
                return False
            
            # Check market hours (simplified)
            now = datetime.now()
            if now.weekday() >= 5:  # Weekend
                self.logger.warning("Markets are closed on weekends")
                return False
            
            return True
            
        except Exception as e:
            self.logger.error(f"Error validating order: {e}")
            return False
    
    async def _select_optimal_venue(self, symbol: str, side: str, quantity: int, 
                                  order_type: OrderType) -> ExecutionVenue:
        """Select optimal execution venue using smart order routing"""
        try:
            # Get venue scores based on multiple factors
            venue_scores = {}
            
            for venue, properties in self.venues.items():
                score = 0.0
                
                # Fee consideration (lower is better)
                fee_score = (1 - properties['fee'] / 0.002) * 30  # Weight: 30%
                
                # Fill rate consideration (higher is better)
                fill_rate_score = properties['fill_rate'] * 40  # Weight: 40%
                
                # Latency consideration (lower is better)
                latency_score = (1 - properties['latency'] / 0.003) * 20  # Weight: 20%
                
                # Historical performance
                historical_score = self.venue_performance.get(venue, {}).get('avg_score', 5.0)  # Weight: 10%
                
                total_score = fee_score + fill_rate_score + latency_score + historical_score
                venue_scores[venue] = total_score
            
            # Select venue with highest score
            optimal_venue = max(venue_scores.items(), key=lambda x: x[1])[0]
            
            # For large orders, prefer dark pools
            estimated_value = quantity * 100  # Rough estimate
            if estimated_value > 50000 and order_type != OrderType.MARKET:
                dark_pools = [ExecutionVenue.DARK_POOL_1, ExecutionVenue.DARK_POOL_2]
                dark_pool_scores = {v: s for v, s in venue_scores.items() if v in dark_pools}
                if dark_pool_scores:
                    optimal_venue = max(dark_pool_scores.items(), key=lambda x: x[1])[0]
            
            return optimal_venue
            
        except Exception as e:
            self.logger.error(f"Error selecting venue: {e}")
            return ExecutionVenue.IEX  # Default to IEX
    
    async def _route_order(self, order: Order):
        """Route order to selected venue"""
        try:
            # Update order status
            order.status = OrderStatus.SUBMITTED
            order.submitted_time = datetime.now(timezone.utc)
            
            # Simulate order submission latency
            venue_properties = self.venues[order.venue]
            await asyncio.sleep(venue_properties['latency'])
            
            # Simulate execution (simplified)
            await self._simulate_execution(order)
            
        except Exception as e:
            self.logger.error(f"Error routing order {order.order_id}: {e}")
            order.status = OrderStatus.REJECTED
    
    async def _simulate_execution(self, order: Order):
        """Simulate order execution (in production, this would be real venue integration)"""
        try:
            venue_properties = self.venues[order.venue]
            fill_probability = venue_properties['fill_rate']
            
            # Determine if order fills
            if np.random.random() < fill_probability:
                # Simulate partial or full fill
                fill_percentage = np.random.uniform(0.8, 1.0)  # 80-100% fill
                filled_quantity = int(order.quantity * fill_percentage)
                
                # Simulate fill price (with some slippage)
                if order.order_type == OrderType.MARKET:
                    # Market orders have more slippage
                    slippage_bps = np.random.uniform(-5, 15)  # -5 to +15 basis points
                    fill_price = (order.price or 100) * (1 + slippage_bps / 10000)
                else:
                    # Limit orders fill at or better than limit price
                    if order.side == 'BUY':
                        fill_price = min(order.price, (order.price or 100) * 0.999)
                    else:
                        fill_price = max(order.price, (order.price or 100) * 1.001)
                
                # Create fill
                fill = Fill(
                    fill_id=str(uuid.uuid4()),
                    order_id=order.order_id,
                    symbol=order.symbol,
                    side=order.side,
                    quantity=filled_quantity,
                    price=fill_price,
                    venue=order.venue,
                    timestamp=datetime.now(timezone.utc),
                    commission=filled_quantity * fill_price * venue_properties['fee'],
                    liquidity_flag='TAKER' if order.order_type == OrderType.MARKET else 'MAKER'
                )
                
                # Update order
                order.filled_quantity += filled_quantity
                order.avg_fill_price = ((order.avg_fill_price * (order.filled_quantity - filled_quantity)) + 
                                      (fill_price * filled_quantity)) / order.filled_quantity
                order.commission += fill.commission
                
                if order.filled_quantity >= order.quantity:
                    order.status = OrderStatus.FILLED
                    order.filled_time = datetime.now(timezone.utc)
                else:
                    order.status = OrderStatus.PARTIALLY_FILLED
                
                # Add fill to history
                self.fills.append(fill)
                
                # Update daily volume
                self.daily_volume_traded += filled_quantity * fill_price
                
                # Send execution notification
                await self._send_execution_notification(order, fill)
                
            else:
                # Order didn't fill - keep as submitted
                pass
                
        except Exception as e:
            self.logger.error(f"Error simulating execution for {order.order_id}: {e}")
            order.status = OrderStatus.REJECTED
    
    async def _execute_algorithmic_order(self, order: Order):
        """Execute algorithmic order (TWAP, VWAP, etc.)"""
        try:
            if order.execution_algorithm == 'TWAP':
                await self._execute_twap(order)
            elif order.execution_algorithm == 'VWAP':
                await self._execute_vwap(order)
            elif order.execution_algorithm == 'ICEBERG':
                await self._execute_iceberg(order)
            else:
                # Default to simple routing
                await self._route_order(order)
                
        except Exception as e:
            self.logger.error(f"Error executing algorithmic order {order.order_id}: {e}")
    
    async def _execute_twap(self, order: Order):
        """Execute Time-Weighted Average Price algorithm"""
        try:
            # Split order into time slices
            slice_duration = self.twap_slice_minutes * 60  # Convert to seconds
            total_slices = max(1, int(8 * 60 * 60 / slice_duration))  # 8 hours / slice duration
            quantity_per_slice = max(1, order.quantity // total_slices)
            
            # Create child orders
            for i in range(total_slices):
                if order.filled_quantity >= order.quantity:
                    break
                
                remaining_quantity = order.quantity - order.filled_quantity
                slice_quantity = min(quantity_per_slice, remaining_quantity)
                
                # Create child order
                child_order = Order(
                    order_id=str(uuid.uuid4()),
                    symbol=order.symbol,
                    side=order.side,
                    quantity=slice_quantity,
                    order_type=order.order_type,
                    price=order.price,
                    stop_price=order.stop_price,
                    venue=order.venue,
                    time_in_force='IOC',  # Immediate or Cancel
                    status=OrderStatus.PENDING,
                    filled_quantity=0,
                    avg_fill_price=0.0,
                    commission=0.0,
                    created_time=datetime.now(timezone.utc),
                    submitted_time=None,
                    filled_time=None,
                    execution_algorithm=None,
                    parent_order_id=order.order_id,
                    metadata={'slice': i + 1, 'total_slices': total_slices}
                )
                
                # Route child order
                await self._route_order(child_order)
                
                # Update parent order
                if child_order.status == OrderStatus.FILLED:
                    order.filled_quantity += child_order.filled_quantity
                    order.commission += child_order.commission
                    
                    # Update average fill price
                    if order.filled_quantity > 0:
                        total_value = (order.avg_fill_price * (order.filled_quantity - child_order.filled_quantity) + 
                                     child_order.avg_fill_price * child_order.filled_quantity)
                        order.avg_fill_price = total_value / order.filled_quantity
                
                # Wait before next slice (if not last slice)
                if i < total_slices - 1:
                    await asyncio.sleep(slice_duration)
            
            # Update parent order status
            if order.filled_quantity >= order.quantity:
                order.status = OrderStatus.FILLED
                order.filled_time = datetime.now(timezone.utc)
            elif order.filled_quantity > 0:
                order.status = OrderStatus.PARTIALLY_FILLED
            
        except Exception as e:
            self.logger.error(f"Error executing TWAP for {order.order_id}: {e}")
    
    async def _execute_vwap(self, order: Order):
        """Execute Volume-Weighted Average Price algorithm"""
        try:
            # Simplified VWAP - in production would use real volume profiles
            # For now, treat similar to TWAP but with volume-based sizing
            await self._execute_twap(order)
            
        except Exception as e:
            self.logger.error(f"Error executing VWAP for {order.order_id}: {e}")
    
    async def _execute_iceberg(self, order: Order):
        """Execute Iceberg algorithm (show small portion, hide the rest)"""
        try:
            display_size = max(1, int(order.quantity * self.iceberg_display_size))
            
            while order.filled_quantity < order.quantity:
                remaining_quantity = order.quantity - order.filled_quantity
                current_display = min(display_size, remaining_quantity)
                
                # Create visible child order
                child_order = Order(
                    order_id=str(uuid.uuid4()),
                    symbol=order.symbol,
                    side=order.side,
                    quantity=current_display,
                    order_type=order.order_type,
                    price=order.price,
                    stop_price=order.stop_price,
                    venue=order.venue,
                    time_in_force='GTC',
                    status=OrderStatus.PENDING,
                    filled_quantity=0,
                    avg_fill_price=0.0,
                    commission=0.0,
                    created_time=datetime.now(timezone.utc),
                    submitted_time=None,
                    filled_time=None,
                    execution_algorithm=None,
                    parent_order_id=order.order_id,
                    metadata={'iceberg_slice': True}
                )
                
                # Route child order
                await self._route_order(child_order)
                
                # Wait for fill or timeout
                timeout = 300  # 5 minutes
                start_time = datetime.now()
                
                while (child_order.status not in [OrderStatus.FILLED, OrderStatus.CANCELLED] and 
                       (datetime.now() - start_time).seconds < timeout):
                    await asyncio.sleep(1)
                    await self._update_single_order_status(child_order)
                
                # Update parent order
                if child_order.filled_quantity > 0:
                    order.filled_quantity += child_order.filled_quantity
                    order.commission += child_order.commission
                    
                    if order.filled_quantity > 0:
                        total_value = (order.avg_fill_price * (order.filled_quantity - child_order.filled_quantity) + 
                                     child_order.avg_fill_price * child_order.filled_quantity)
                        order.avg_fill_price = total_value / order.filled_quantity
                
                # If child order didn't fill completely, cancel it
                if child_order.status == OrderStatus.PARTIALLY_FILLED:
                    await self.cancel_order(child_order.order_id)
                
                # Break if no progress
                if child_order.filled_quantity == 0:
                    break
            
            # Update parent order status
            if order.filled_quantity >= order.quantity:
                order.status = OrderStatus.FILLED
                order.filled_time = datetime.now(timezone.utc)
            elif order.filled_quantity > 0:
                order.status = OrderStatus.PARTIALLY_FILLED
            
        except Exception as e:
            self.logger.error(f"Error executing Iceberg for {order.order_id}: {e}")
    
    async def _process_pending_orders(self):
        """Process pending orders"""
        try:
            pending_orders = [order for order in self.active_orders.values() 
                            if order.status == OrderStatus.PENDING]
            
            for order in pending_orders:
                try:
                    await self._route_order(order)
                except Exception as e:
                    self.logger.error(f"Error processing pending order {order.order_id}: {e}")
                    order.status = OrderStatus.REJECTED
                    
        except Exception as e:
            self.logger.error(f"Error processing pending orders: {e}")
    
    async def _update_order_statuses(self):
        """Update statuses of active orders"""
        try:
            active_orders = [order for order in self.active_orders.values() 
                           if order.status in [OrderStatus.SUBMITTED, OrderStatus.PARTIALLY_FILLED]]
            
            for order in active_orders:
                await self._update_single_order_status(order)
                
        except Exception as e:
            self.logger.error(f"Error updating order statuses: {e}")
    
    async def _update_single_order_status(self, order: Order):
        """Update status of a single order"""
        try:
            # In production, this would query the venue for order status
            # For simulation, we'll randomly progress some orders
            
            if order.status == OrderStatus.SUBMITTED:
                # Small chance of getting a fill
                if np.random.random() < 0.1:  # 10% chance per update
                    await self._simulate_execution(order)
            
        except Exception as e:
            self.logger.error(f"Error updating order status for {order.order_id}: {e}")
    
    async def _send_execution_notification(self, order: Order, fill: Fill):
        """Send execution notification to relevant agents"""
        try:
            # Send to portfolio management
            await self.send_message(
                'portfolio-mgmt-agent',
                'execution_notification',
                {
                    'order': order.to_dict(),
                    'fill': fill.to_dict()
                },
                priority=2
            )
            
            # Send to risk management
            await self.send_message(
                'risk-mgmt-agent',
                'execution_notification',
                {
                    'order': order.to_dict(),
                    'fill': fill.to_dict()
                },
                priority=3
            )
            
        except Exception as e:
            self.logger.error(f"Error sending execution notification: {e}")
    
    async def _send_order_update(self, order: Order):
        """Send order status update"""
        try:
            await self.send_message(
                'portfolio-mgmt-agent',
                'order_update',
                order.to_dict(),
                priority=3
            )
            
        except Exception as e:
            self.logger.error(f"Error sending order update: {e}")
    
    async def _monitor_execution_quality(self):
        """Monitor execution quality metrics"""
        try:
            # Calculate execution metrics for recent fills
            recent_fills = [fill for fill in self.fills 
                          if (datetime.now(timezone.utc) - fill.timestamp).seconds < 3600]  # Last hour
            
            if not recent_fills:
                return
            
            # Group by symbol
            symbol_fills = {}
            for fill in recent_fills:
                if fill.symbol not in symbol_fills:
                    symbol_fills[fill.symbol] = []
                symbol_fills[fill.symbol].append(fill)
            
            # Calculate metrics for each symbol
            for symbol, fills in symbol_fills.items():
                metrics = await self._calculate_execution_metrics(symbol, fills)
                self.execution_metrics[symbol] = metrics
                
                # Cache metrics
                self.cache_data(f'execution_metrics_{symbol}', asdict(metrics), expiry=3600)
            
        except Exception as e:
            self.logger.error(f"Error monitoring execution quality: {e}")
    
    async def _calculate_execution_metrics(self, symbol: str, fills: List[Fill]) -> ExecutionMetrics:
        """Calculate execution metrics for a symbol"""
        try:
            total_volume = sum(fill.quantity for fill in fills)
            total_value = sum(fill.quantity * fill.price for fill in fills)
            avg_fill_price = total_value / total_volume if total_volume > 0 else 0.0
            
            # Get VWAP benchmark (simplified)
            vwap_benchmark = avg_fill_price * 1.001  # Assume VWAP is slightly higher
            
            # Calculate slippage
            slippage = ((avg_fill_price - vwap_benchmark) / vwap_benchmark) * 10000  # Basis points
            
            # Market impact (simplified)
            market_impact = abs(slippage) * 0.5  # Assume half of slippage is market impact
            
            # Fill rate (simplified)
            fill_rate = 0.95  # 95% fill rate assumption
            
            # Average execution time
            avg_execution_time = 2.5  # 2.5 seconds average
            
            # Venue breakdown
            venue_breakdown = {}
            for venue in ExecutionVenue:
                venue_fills = [f for f in fills if f.venue == venue]
                if venue_fills:
                    venue_volume = sum(f.quantity for f in venue_fills)
                    venue_value = sum(f.quantity * f.price for f in venue_fills)
                    venue_breakdown[venue] = {
                        'volume': venue_volume,
                        'avg_price': venue_value / venue_volume,
                        'fill_count': len(venue_fills)
                    }
            
            # Commission rate
            total_commission = sum(fill.commission for fill in fills)
            commission_rate = (total_commission / total_value) if total_value > 0 else 0.0
            
            return ExecutionMetrics(
                symbol=symbol,
                total_orders=len(set(fill.order_id for fill in fills)),
                total_volume=total_volume,
                avg_fill_price=avg_fill_price,
                vwap_benchmark=vwap_benchmark,
                slippage=slippage,
                market_impact=market_impact,
                fill_rate=fill_rate,
                avg_execution_time=avg_execution_time,
                venue_breakdown=venue_breakdown,
                commission_rate=commission_rate,
                timestamp=datetime.now(timezone.utc)
            )
            
        except Exception as e:
            self.logger.error(f"Error calculating execution metrics for {symbol}: {e}")
            return ExecutionMetrics(
                symbol=symbol,
                total_orders=0,
                total_volume=0,
                avg_fill_price=0.0,
                vwap_benchmark=0.0,
                slippage=0.0,
                market_impact=0.0,
                fill_rate=0.0,
                avg_execution_time=0.0,
                venue_breakdown={},
                commission_rate=0.0,
                timestamp=datetime.now(timezone.utc)
            )
    
    async def _generate_execution_reports(self):
        """Generate execution quality reports"""
        try:
            if not self.execution_metrics:
                return
            
            # Calculate overall metrics
            total_orders = sum(metrics.total_orders for metrics in self.execution_metrics.values())
            total_volume = sum(metrics.total_volume for metrics in self.execution_metrics.values())
            avg_slippage = np.mean([metrics.slippage for metrics in self.execution_metrics.values()])
            
            # Find best and worst performing venues
            venue_performance = {}
            for metrics in self.execution_metrics.values():
                for venue, breakdown in metrics.venue_breakdown.items():
                    if venue not in venue_performance:
                        venue_performance[venue] = []
                    venue_performance[venue].append(breakdown['avg_price'])
            
            best_venue = min(venue_performance.items(), key=lambda x: np.mean(x[1]))[0] if venue_performance else ExecutionVenue.IEX
            worst_venue = max(venue_performance.items(), key=lambda x: np.mean(x[1]))[0] if venue_performance else ExecutionVenue.NYSE
            
            # Generate recommendations
            recommendations = []
            if avg_slippage > 5.0:  # > 5 basis points
                recommendations.append("Consider using more limit orders to reduce slippage")
            if total_volume > self.max_daily_volume * 0.8:
                recommendations.append("Approaching daily volume limit - consider spreading trades")
            
            # Create report
            report = ExecutionReport(
                period_start=datetime.now(timezone.utc) - timedelta(hours=1),
                period_end=datetime.now(timezone.utc),
                total_orders=total_orders,
                total_volume=float(total_volume),
                total_commission=sum(sum(fill.commission for fill in self.fills[-100:]) for _ in [1]),  # Last 100 fills
                avg_slippage=float(avg_slippage),
                best_performing_venue=best_venue,
                worst_performing_venue=worst_venue,
                symbol_metrics=self.execution_metrics,
                recommendations=recommendations,
                timestamp=datetime.now(timezone.utc)
            )
            
            # Cache report
            self.cache_data('execution_report', asdict(report), expiry=3600)
            
        except Exception as e:
            self.logger.error(f"Error generating execution reports: {e}")
    
    async def _cleanup_completed_orders(self):
        """Clean up completed orders"""
        try:
            # Move completed orders to history
            completed_orders = [order for order in self.active_orders.values() 
                              if order.status in [OrderStatus.FILLED, OrderStatus.CANCELLED, OrderStatus.REJECTED]]
            
            for order in completed_orders:
                self.order_history.append(order)
                del self.active_orders[order.order_id]
            
            # Keep only recent history
            if len(self.order_history) > 1000:
                self.order_history = self.order_history[-1000:]
            
            # Keep only recent fills
            if len(self.fills) > 5000:
                self.fills = self.fills[-5000:]
                
        except Exception as e:
            self.logger.error(f"Error cleaning up orders: {e}")
    
    async def _initialize_venues(self):
        """Initialize venue connections"""
        try:
            # In production, this would establish connections to trading venues
            # For simulation, we'll just log the initialization
            for venue in self.venues:
                self.logger.info(f"Initialized connection to {venue.value}")
                
        except Exception as e:
            self.logger.error(f"Error initializing venues: {e}")
    
    async def _load_execution_history(self):
        """Load execution history"""
        try:
            # In production, this would load from database
            # For simulation, start with empty history
            self.order_history = []
            self.fills = []
            
        except Exception as e:
            self.logger.error(f"Error loading execution history: {e}")
    
    async def handle_message(self, message: AgentMessage):
        """Handle incoming messages from other agents"""
        try:
            if message.message_type == 'submit_order':
                # Handle order submission request
                order_data = message.data
                order = await self.submit_order(
                    symbol=order_data.get('symbol'),
                    side=order_data.get('side'),
                    quantity=order_data.get('quantity'),
                    order_type=OrderType(order_data.get('order_type', 'MARKET')),
                    price=order_data.get('price'),
                    algorithm=order_data.get('algorithm')
                )
                
                await self.send_message(
                    message.agent_id,
                    'order_submitted',
                    order.to_dict(),
                    correlation_id=message.correlation_id
                )
            
            elif message.message_type == 'cancel_order':
                # Handle order cancellation request
                order_id = message.data.get('order_id')
                success = await self.cancel_order(order_id)
                
                await self.send_message(
                    message.agent_id,
                    'order_cancelled' if success else 'cancel_failed',
                    {'order_id': order_id, 'success': success},
                    correlation_id=message.correlation_id
                )
            
            elif message.message_type == 'execution_report_request':
                # Provide execution report
                report = self.get_cached_data('execution_report')
                if report:
                    await self.send_message(
                        message.agent_id,
                        'execution_report_response',
                        report,
                        correlation_id=message.correlation_id
                    )
                    
        except Exception as e:
            self.logger.error(f"Error handling message: {e}")
    
    async def agent_cleanup(self):
        """Cleanup agent resources"""
        # Cancel all active orders
        for order_id in list(self.active_orders.keys()):
            await self.cancel_order(order_id)
        
        self.logger.info("Execution Agent cleaned up successfully") 