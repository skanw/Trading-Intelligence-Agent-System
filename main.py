#!/usr/bin/env python3
"""
Trading Intelligence Agent System (TIAS) - Main Entry Point

Production-ready multi-agent trading system for financial markets.
This script demonstrates how to start and run the complete agent system.
"""

import asyncio
import argparse
import logging
import signal
import sys
from datetime import datetime
from typing import Optional

from src.agents import AgentSystem, CORE_AGENTS, ENHANCEMENT_AGENTS
from src.config import config


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/tias.log'),
        logging.StreamHandler(sys.stdout)
    ]
)

logger = logging.getLogger('TIAS-Main')


class TradingSystemManager:
    """
    Main system manager for TIAS
    """
    
    def __init__(self, enabled_agents: list = None, config_override: dict = None):
        self.agent_system = AgentSystem(enabled_agents, config_override)
        self.is_running = False
        self.shutdown_event = asyncio.Event()
        
    async def start(self):
        """Start the trading system"""
        try:
            logger.info("🚀 Starting Trading Intelligence Agent System (TIAS)")
            logger.info(f"📅 Start time: {datetime.now()}")
            logger.info(f"🔧 Configuration: {config.get('environment', 'development')}")
            logger.info(f"🤖 Enabled agents: {', '.join(self.agent_system.enabled_agents)}")
            
            # Start agent system
            await self.agent_system.start()
            
            self.is_running = True
            logger.info("✅ Trading Intelligence Agent System is now running")
            
            # Set up signal handlers for graceful shutdown
            self._setup_signal_handlers()
            
            # Start main operation loop
            await self._run_main_loop()
            
        except Exception as e:
            logger.error(f"❌ Failed to start TIAS: {e}")
            raise
    
    async def stop(self):
        """Stop the trading system gracefully"""
        try:
            logger.info("🛑 Shutting down Trading Intelligence Agent System...")
            
            self.is_running = False
            self.shutdown_event.set()
            
            # Stop agent system
            await self.agent_system.stop()
            
            logger.info("✅ TIAS shutdown complete")
            
        except Exception as e:
            logger.error(f"❌ Error during shutdown: {e}")
    
    def _setup_signal_handlers(self):
        """Setup signal handlers for graceful shutdown"""
        def signal_handler(signum, frame):
            logger.info(f"📡 Received signal {signum}, initiating graceful shutdown...")
            asyncio.create_task(self.stop())
        
        signal.signal(signal.SIGINT, signal_handler)
        signal.signal(signal.SIGTERM, signal_handler)
    
    async def _run_main_loop(self):
        """Main operation loop"""
        try:
            logger.info("🔄 Starting main operation loop...")
            
            # Run agent execution cycles
            execution_interval = 30  # 30 seconds between cycles
            
            while self.is_running and not self.shutdown_event.is_set():
                try:
                    # Execute all agents
                    await self._execute_agent_cycle()
                    
                    # Wait for next cycle or shutdown signal
                    try:
                        await asyncio.wait_for(
                            self.shutdown_event.wait(), 
                            timeout=execution_interval
                        )
                        break  # Shutdown signal received
                    except asyncio.TimeoutError:
                        continue  # Normal timeout, continue to next cycle
                        
                except Exception as e:
                    logger.error(f"💥 Error in main loop cycle: {e}")
                    await asyncio.sleep(5)  # Brief pause before retry
            
            logger.info("🔚 Main operation loop ended")
            
        except Exception as e:
            logger.error(f"💥 Critical error in main loop: {e}")
            raise
    
    async def _execute_agent_cycle(self):
        """Execute one cycle of all agents"""
        try:
            # Get all active agents
            active_agents = list(self.agent_system.agents.values())
            
            if not active_agents:
                logger.warning("⚠️ No active agents found")
                return
            
            # Execute all agents concurrently
            tasks = []
            for agent in active_agents:
                if hasattr(agent, 'execute'):
                    tasks.append(agent.execute())
            
            if tasks:
                # Wait for all agents to complete their execution
                await asyncio.gather(*tasks, return_exceptions=True)
                logger.debug(f"🔄 Completed execution cycle for {len(tasks)} agents")
            
        except Exception as e:
            logger.error(f"💥 Error executing agent cycle: {e}")
    
    def get_system_status(self):
        """Get current system status"""
        return {
            'system_running': self.is_running,
            'start_time': datetime.now(),
            'agent_system': self.agent_system.get_system_status()
        }


async def demo_basic_operations(system_manager: TradingSystemManager):
    """
    Demonstrate basic system operations
    """
    try:
        logger.info("🎬 Running basic operations demo...")
        
        # Wait a bit for agents to initialize
        await asyncio.sleep(5)
        
        # Get orchestrator agent
        orchestrator = system_manager.agent_system.get_agent('orchestrator-agent')
        
        if orchestrator:
            logger.info("📊 Requesting system status from orchestrator...")
            
            # Request system status
            await orchestrator.send_message(
                'orchestrator-agent',
                'request_system_status',
                {},
                priority=3
            )
            
            # Wait a bit for processing
            await asyncio.sleep(2)
            
            # Check cached data
            status = orchestrator.get_cached_data('orchestrator_metrics')
            if status:
                logger.info(f"📈 System metrics: {status}")
            else:
                logger.info("📊 System metrics not yet available")
        
        # Get market intelligence agent
        market_agent = system_manager.agent_system.get_agent('market-intel-agent')
        
        if market_agent:
            logger.info("🌍 Checking market regime...")
            
            # Wait for market data
            await asyncio.sleep(3)
            
            regime_data = market_agent.get_cached_data('market_regime')
            if regime_data:
                logger.info(f"📊 Market regime: {regime_data.get('regime_type', 'Unknown')}")
            else:
                logger.info("🌍 Market regime data not yet available")
        
        # Get news intelligence agent
        news_agent = system_manager.agent_system.get_agent('news-intel-agent')
        
        if news_agent:
            logger.info("📰 Checking latest news analysis...")
            
            # Wait for news processing
            await asyncio.sleep(2)
            
            # Check for recent news signals
            recent_signals = news_agent.get_cached_data('recent_news_signals')
            if recent_signals:
                logger.info(f"📰 Found {len(recent_signals)} recent news signals")
            else:
                logger.info("📰 No recent news signals available")
        
        logger.info("✅ Basic operations demo completed")
        
    except Exception as e:
        logger.error(f"💥 Error in demo operations: {e}")


async def run_demo():
    """
    Run a demonstration of the trading system
    """
    try:
        logger.info("🎭 Starting TIAS demonstration...")
        
        # Use core agents plus some enhancements for demo
        demo_agents = CORE_AGENTS + ['news-intel-agent', 'technical-agent']
        
        # Create system manager
        system_manager = TradingSystemManager(
            enabled_agents=demo_agents,
            config_override={
                'environment': 'demo',
                'log_level': 'INFO'
            }
        )
        
        # Start the system
        await system_manager.start()
        
        # Run demo operations
        demo_task = asyncio.create_task(demo_basic_operations(system_manager))
        
        # Let demo run for a few minutes
        await asyncio.sleep(180)  # 3 minutes
        
        # Cancel demo task if still running
        if not demo_task.done():
            demo_task.cancel()
            try:
                await demo_task
            except asyncio.CancelledError:
                pass
        
        # Stop the system
        await system_manager.stop()
        
        logger.info("🎭 TIAS demonstration completed successfully")
        
    except Exception as e:
        logger.error(f"💥 Demo failed: {e}")
        raise


async def run_production():
    """
    Run the production trading system
    """
    try:
        logger.info("🏭 Starting TIAS in production mode...")
        
        # Use all available agents for production
        production_agents = CORE_AGENTS + ENHANCEMENT_AGENTS
        
        # Create system manager with production configuration
        system_manager = TradingSystemManager(
            enabled_agents=production_agents,
            config_override={
                'environment': 'production',
                'log_level': 'WARNING'
            }
        )
        
        # Start and run the system
        await system_manager.start()
        
        logger.info("🏭 TIAS is now running in production mode")
        logger.info("💡 System will run until interrupted (Ctrl+C)")
        
        # Keep running until interrupted
        while system_manager.is_running:
            await asyncio.sleep(1)
        
    except Exception as e:
        logger.error(f"💥 Production system failed: {e}")
        raise


def main():
    """
    Main entry point with argument parsing
    """
    parser = argparse.ArgumentParser(
        description='Trading Intelligence Agent System (TIAS)',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py --mode demo                    # Run demonstration
  python main.py --mode production              # Run production system
  python main.py --agents core                  # Run only core agents
  python main.py --agents all                   # Run all agents
  python main.py --config custom_config.json   # Use custom configuration
        """
    )
    
    parser.add_argument(
        '--mode',
        choices=['demo', 'production'],
        default='demo',
        help='Operation mode (default: demo)'
    )
    
    parser.add_argument(
        '--agents',
        choices=['core', 'all', 'custom'],
        default='core',
        help='Agent selection (default: core)'
    )
    
    parser.add_argument(
        '--config',
        type=str,
        help='Path to custom configuration file'
    )
    
    parser.add_argument(
        '--log-level',
        choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
        default='INFO',
        help='Logging level (default: INFO)'
    )
    
    args = parser.parse_args()
    
    # Set logging level
    logging.getLogger().setLevel(getattr(logging, args.log_level))
    
    try:
        # Create logs directory if it doesn't exist
        import os
        os.makedirs('logs', exist_ok=True)
        
        logger.info("=" * 80)
        logger.info("🤖 TRADING INTELLIGENCE AGENT SYSTEM (TIAS)")
        logger.info("=" * 80)
        logger.info(f"📋 Mode: {args.mode}")
        logger.info(f"🤖 Agents: {args.agents}")
        logger.info(f"📊 Log Level: {args.log_level}")
        
        # Run the appropriate mode
        if args.mode == 'demo':
            asyncio.run(run_demo())
        elif args.mode == 'production':
            asyncio.run(run_production())
        
    except KeyboardInterrupt:
        logger.info("🛑 Interrupted by user")
    except Exception as e:
        logger.error(f"💥 System failed: {e}")
        sys.exit(1)
    finally:
        logger.info("👋 TIAS session ended")


if __name__ == '__main__':
    main() 