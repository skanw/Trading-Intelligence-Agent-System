#!/usr/bin/env python3
"""
Quick start script for Trading Intelligence Agent System (TIAS)
"""

import asyncio
import os
import sys
import logging

# Add project root to Python path
project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, project_root)

from src.agents import AgentSystem, CORE_AGENTS


async def quick_demo():
    """Quick demonstration of TIAS capabilities"""
    
    # Setup basic logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    logger = logging.getLogger('TIAS-QuickStart')
    
    try:
        logger.info("🚀 QUICK START - Trading Intelligence Agent System")
        logger.info("=" * 60)
        
        # Create logs directory
        os.makedirs('logs', exist_ok=True)
        
        # Start with core agents for quick demo
        logger.info("🤖 Starting core agents...")
        system = AgentSystem(enabled_agents=CORE_AGENTS)
        
        # Start the system
        await system.start()
        
        logger.info("✅ System started successfully!")
        logger.info("📊 Running basic operations for 60 seconds...")
        
        # Let it run for 1 minute
        for i in range(6):
            logger.info(f"⏱️  Running... ({(i+1)*10}s / 60s)")
            
            # Execute agent cycles
            for agent in system.agents.values():
                try:
                    if hasattr(agent, 'execute'):
                        await agent.execute()
                except Exception as e:
                    logger.warning(f"⚠️ Agent execution warning: {e}")
            
            await asyncio.sleep(10)
        
        logger.info("🎯 Demo completed! Stopping system...")
        
        # Stop the system
        await system.stop()
        
        logger.info("✅ TIAS Quick Demo completed successfully!")
        logger.info("💡 To run full system: python main.py --mode demo")
        
    except Exception as e:
        logger.error(f"❌ Quick demo failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == '__main__':
    print("🤖 Trading Intelligence Agent System - Quick Start")
    print("=" * 50)
    print("This will run a 60-second demonstration of TIAS core agents.")
    print("Press Ctrl+C to stop early if needed.")
    print("")
    
    try:
        asyncio.run(quick_demo())
    except KeyboardInterrupt:
        print("\n🛑 Demo interrupted by user")
    except Exception as e:
        print(f"\n❌ Demo failed: {e}")
    finally:
        print("👋 Quick start session ended") 