"""
Trading Intelligence Agent System (TIAS) - Agent Registry
"""

from .base_agent import BaseAgent, AgentMessage, AgentSignal
from .news_intelligence import NewsIntelligenceAgent
from .market_intelligence import MarketIntelligenceAgent
from .technical_analysis import TechnicalAnalysisAgent
from .fundamental_analysis import FundamentalAnalysisAgent
from .risk_management import RiskManagementAgent
from .execution import ExecutionAgent
from .orchestrator import OrchestrationAgent

# Agent registry for easy access
AGENT_CLASSES = {
    'news-intel-agent': NewsIntelligenceAgent,
    'market-intel-agent': MarketIntelligenceAgent,
    'technical-agent': TechnicalAnalysisAgent,
    'fundamental-agent': FundamentalAnalysisAgent,
    'risk-mgmt-agent': RiskManagementAgent,
    'execution-agent': ExecutionAgent,
    'orchestrator-agent': OrchestrationAgent
}

# Agent dependency graph
AGENT_DEPENDENCIES = {
    'orchestrator-agent': [],  # No dependencies
    'market-intel-agent': [],
    'news-intel-agent': [],
    'fundamental-agent': [],
    'technical-agent': [],
    'risk-mgmt-agent': ['market-intel-agent', 'fundamental-agent', 'technical-agent'],
    'execution-agent': ['risk-mgmt-agent'],
}

# Agent initialization order (respecting dependencies)
AGENT_INIT_ORDER = [
    'orchestrator-agent',
    'market-intel-agent',
    'news-intel-agent',
    'fundamental-agent',
    'technical-agent',
    'risk-mgmt-agent',
    'execution-agent'
]

# Core production agents (always required)
CORE_AGENTS = [
    'orchestrator-agent',
    'market-intel-agent',
    'fundamental-agent',
    'technical-agent',
    'risk-mgmt-agent',
    'execution-agent'
]

# Optional enhancement agents
ENHANCEMENT_AGENTS = [
    'news-intel-agent'
]

__all__ = [
    'BaseAgent',
    'AgentMessage', 
    'AgentSignal',
    'NewsIntelligenceAgent',
    'MarketIntelligenceAgent',
    'TechnicalAnalysisAgent',
    'FundamentalAnalysisAgent',
    'RiskManagementAgent',
    'ExecutionAgent',
    'OrchestrationAgent',
    'AGENT_CLASSES',
    'AGENT_DEPENDENCIES',
    'AGENT_INIT_ORDER',
    'CORE_AGENTS',
    'ENHANCEMENT_AGENTS'
]


def create_agent(agent_type: str, agent_id: str = None, config_override: dict = None):
    """
    Factory function to create agent instances
    
    Args:
        agent_type: Type of agent to create (key from AGENT_CLASSES)
        agent_id: Optional custom agent ID
        config_override: Optional configuration overrides
    
    Returns:
        Agent instance
    """
    if agent_type not in AGENT_CLASSES:
        raise ValueError(f"Unknown agent type: {agent_type}. Available: {list(AGENT_CLASSES.keys())}")
    
    agent_class = AGENT_CLASSES[agent_type]
    
    if agent_id:
        return agent_class(agent_id=agent_id, config_override=config_override)
    else:
        return agent_class(config_override=config_override)


def get_agent_dependencies(agent_type: str) -> list:
    """
    Get dependencies for an agent type
    
    Args:
        agent_type: Type of agent
        
    Returns:
        List of agent types that this agent depends on
    """
    return AGENT_DEPENDENCIES.get(agent_type, [])


def validate_agent_setup(enabled_agents: list) -> tuple:
    """
    Validate that all dependencies are satisfied for enabled agents
    
    Args:
        enabled_agents: List of agent types to enable
        
    Returns:
        Tuple of (is_valid, missing_dependencies)
    """
    missing_deps = []
    
    for agent_type in enabled_agents:
        deps = get_agent_dependencies(agent_type)
        for dep in deps:
            if dep not in enabled_agents:
                missing_deps.append(f"{agent_type} requires {dep}")
    
    return len(missing_deps) == 0, missing_deps


def get_startup_order(enabled_agents: list) -> list:
    """
    Get the correct startup order for enabled agents based on dependencies
    
    Args:
        enabled_agents: List of agent types to start
        
    Returns:
        List of agent types in correct startup order
    """
    # Filter init order to only include enabled agents
    startup_order = [agent for agent in AGENT_INIT_ORDER if agent in enabled_agents]
    
    # Add any enabled agents not in the predefined order
    for agent in enabled_agents:
        if agent not in startup_order:
            startup_order.append(agent)
    
    return startup_order


class AgentSystem:
    """
    Complete agent system management
    """
    
    def __init__(self, enabled_agents: list = None, config_override: dict = None):
        """
        Initialize agent system
        
        Args:
            enabled_agents: List of agent types to enable (defaults to CORE_AGENTS)
            config_override: Global configuration overrides
        """
        self.enabled_agents = enabled_agents or CORE_AGENTS
        self.config_override = config_override or {}
        self.agents = {}
        self.is_running = False
        
        # Validate setup
        is_valid, missing_deps = validate_agent_setup(self.enabled_agents)
        if not is_valid:
            raise ValueError(f"Invalid agent setup. Missing dependencies: {missing_deps}")
    
    async def start(self):
        """Start all enabled agents in correct order"""
        if self.is_running:
            return
        
        startup_order = get_startup_order(self.enabled_agents)
        
        for agent_type in startup_order:
            try:
                agent = create_agent(agent_type, config_override=self.config_override)
                await agent.agent_initialize()
                self.agents[agent_type] = agent
                print(f"✅ Started {agent_type}")
            except Exception as e:
                print(f"❌ Failed to start {agent_type}: {e}")
                raise
        
        self.is_running = True
        print(f"🚀 Trading Intelligence Agent System started with {len(self.agents)} agents")
    
    async def stop(self):
        """Stop all agents in reverse order"""
        if not self.is_running:
            return
        
        # Stop in reverse order
        for agent_type in reversed(list(self.agents.keys())):
            try:
                agent = self.agents[agent_type]
                await agent.agent_cleanup()
                print(f"🛑 Stopped {agent_type}")
            except Exception as e:
                print(f"⚠️  Error stopping {agent_type}: {e}")
        
        self.agents.clear()
        self.is_running = False
        print("🔴 Trading Intelligence Agent System stopped")
    
    def get_agent(self, agent_type: str):
        """Get agent instance by type"""
        return self.agents.get(agent_type)
    
    def get_system_status(self):
        """Get current system status"""
        return {
            'is_running': self.is_running,
            'enabled_agents': self.enabled_agents,
            'active_agents': list(self.agents.keys()),
            'agent_count': len(self.agents)
        } 