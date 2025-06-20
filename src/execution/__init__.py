"""
TIAS Execution Framework

Live execution engine for production trading with broker integration,
order management, and risk controls.
"""

from .execution_agent import ExecutionAgent, Order, OrderStatus
from .broker_interface import BrokerInterface, MockBroker, AlpacaBroker
from .order_manager import OrderManager, OrderBook

__all__ = [
    'ExecutionAgent',
    'Order',
    'OrderStatus',
    'BrokerInterface',
    'MockBroker',
    'AlpacaBroker',
    'OrderManager',
    'OrderBook'
]

__version__ = '1.0.0' 