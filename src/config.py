from pathlib import Path
from pydantic_settings import BaseSettings
import os
from dataclasses import dataclass
from typing import Dict, List, Optional
import logging

class Settings(BaseSettings):
    NEWS_API_KEY: str = "c65c95feae5b45fbade19cc987f5e940"
    REDIS_URL: str = "redis://localhost:6379/0"
    TRACKED_TICKERS: str = "AAPL,MSFT,TSLA,NVDA,AMZN"
    DATA_DIR: Path = Path.cwd() / "data"

    class Config:
        env_file = ".env"

settings = Settings()
settings.DATA_DIR.mkdir(exist_ok=True)

@dataclass
class DataSourceConfig:
    """Configuration for data sources"""
    # Market Data
    bloomberg_api_key: Optional[str] = None
    refinitiv_api_key: Optional[str] = None
    iex_cloud_token: Optional[str] = None
    alpha_vantage_key: Optional[str] = None
    polygon_api_key: Optional[str] = None
    
    # News Sources
    news_api_key: Optional[str] = None
    reuters_api_key: Optional[str] = None
    bloomberg_news_key: Optional[str] = None
    
    # Social Data
    twitter_bearer_token: Optional[str] = None
    reddit_client_id: Optional[str] = None
    reddit_client_secret: Optional[str] = None
    
    # Alternative Data
    satellite_api_key: Optional[str] = None
    patent_api_key: Optional[str] = None
    consumer_data_key: Optional[str] = None

@dataclass
class TradingConfig:
    """Trading system configuration"""
    # Coverage Universe
    large_cap_symbols: List[str] = None
    mid_cap_symbols: List[str] = None
    small_cap_symbols: List[str] = None
    international_symbols: List[str] = None
    crypto_symbols: List[str] = None
    
    # Risk Management
    max_position_size: float = 0.05  # 5% max position
    max_sector_exposure: float = 0.20  # 20% max sector exposure
    stop_loss_pct: float = 0.02  # 2% stop loss
    max_daily_var: float = 0.01  # 1% daily VaR limit
    
    # Execution Settings
    execution_venues: List[str] = None
    dark_pool_access: bool = True
    smart_order_routing: bool = True
    
    def __post_init__(self):
        if self.large_cap_symbols is None:
            self.large_cap_symbols = self._get_sp500_symbols()
        if self.execution_venues is None:
            self.execution_venues = ['NYSE', 'NASDAQ', 'BATS', 'IEX']

    def _get_sp500_symbols(self) -> List[str]:
        """Get S&P 500 symbols - would typically fetch from data provider"""
        return ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA', 'META', 'NVDA', 'BRK.B', 'JNJ', 'JPM']  # Sample

@dataclass
class AgentConfig:
    """Configuration for individual agents"""
    # Agent activation flags
    market_intel_enabled: bool = True
    news_intel_enabled: bool = True
    fundamental_enabled: bool = True
    technical_enabled: bool = True
    risk_mgmt_enabled: bool = True
    execution_enabled: bool = True
    portfolio_mgmt_enabled: bool = True
    alt_data_enabled: bool = True
    options_intel_enabled: bool = True
    crypto_intel_enabled: bool = True
    esg_intel_enabled: bool = True
    
    # Agent update frequencies (seconds)
    market_intel_frequency: int = 60
    news_intel_frequency: int = 30
    fundamental_frequency: int = 3600  # 1 hour
    technical_frequency: int = 300     # 5 minutes
    risk_mgmt_frequency: int = 60
    
    # Agent-specific settings
    news_sentiment_threshold: float = 0.7
    technical_signal_threshold: float = 0.6
    risk_alert_threshold: float = 0.8

@dataclass
class InfrastructureConfig:
    """Infrastructure and deployment configuration"""
    # Redis Configuration
    redis_host: str = 'localhost'
    redis_port: int = 6379
    redis_db: int = 0
    
    # Database Configuration
    postgres_host: str = 'localhost'
    postgres_port: int = 5432
    postgres_db: str = 'trading_system'
    postgres_user: str = 'trader'
    postgres_password: str = os.getenv('POSTGRES_PASSWORD', 'changeme')
    
    # Message Queue
    kafka_brokers: List[str] = None
    kafka_topics: Dict[str, str] = None
    
    # Monitoring
    datadog_api_key: Optional[str] = None
    prometheus_endpoint: str = 'http://localhost:9090'
    grafana_endpoint: str = 'http://localhost:3000'
    
    # Cloud Configuration
    aws_region: str = 'us-east-1'
    gcp_project_id: Optional[str] = None
    azure_subscription_id: Optional[str] = None
    
    def __post_init__(self):
        if self.kafka_brokers is None:
            self.kafka_brokers = ['localhost:9092']
        if self.kafka_topics is None:
            self.kafka_topics = {
                'market_data': 'market-data-stream',
                'news_data': 'news-data-stream',
                'signals': 'trading-signals',
                'executions': 'trade-executions',
                'risk_alerts': 'risk-alerts'
            }

@dataclass
class ComplianceConfig:
    """Compliance and regulatory configuration"""
    # Regulatory Requirements
    sec_compliance: bool = True
    finra_compliance: bool = True
    mifid_compliance: bool = False  # For EU operations
    
    # Audit and Logging
    audit_logging: bool = True
    trade_surveillance: bool = True
    position_reporting: bool = True
    
    # Risk Limits
    leverage_limit: float = 2.0  # 2x max leverage
    concentration_limit: float = 0.10  # 10% max single position
    sector_limit: float = 0.25  # 25% max sector exposure

class Config:
    """Main configuration class"""
    
    def __init__(self):
        self.data_sources = DataSourceConfig()
        self.trading = TradingConfig()
        self.agents = AgentConfig()
        self.infrastructure = InfrastructureConfig()
        self.compliance = ComplianceConfig()
        
        # Load from environment variables
        self._load_from_env()
        
        # Setup logging
        self._setup_logging()
    
    def _load_from_env(self):
        """Load configuration from environment variables"""
        # Data Sources
        self.data_sources.bloomberg_api_key = os.getenv('BLOOMBERG_API_KEY')
        self.data_sources.refinitiv_api_key = os.getenv('REFINITIV_API_KEY')
        self.data_sources.iex_cloud_token = os.getenv('IEX_CLOUD_TOKEN')
        self.data_sources.news_api_key = os.getenv('NEWS_API_KEY')
        self.data_sources.twitter_bearer_token = os.getenv('TWITTER_BEARER_TOKEN')
        
        # Infrastructure
        self.infrastructure.redis_host = os.getenv('REDIS_HOST', 'localhost')
        self.infrastructure.postgres_host = os.getenv('POSTGRES_HOST', 'localhost')
        self.infrastructure.datadog_api_key = os.getenv('DATADOG_API_KEY')
        
        # Trading Environment
        trading_env = os.getenv('TRADING_ENV', 'development')
        if trading_env == 'production':
            self._setup_production_config()
    
    def _setup_production_config(self):
        """Configure for production environment"""
        self.agents.news_intel_frequency = 15  # More frequent in production
        self.agents.risk_mgmt_frequency = 30
        self.compliance.audit_logging = True
        self.compliance.trade_surveillance = True
        
        # Enhanced risk controls for production
        self.trading.max_position_size = 0.03  # 3% max position in production
        self.trading.stop_loss_pct = 0.015     # 1.5% stop loss
    
    def _setup_logging(self):
        """Setup logging configuration"""
        log_level = os.getenv('LOG_LEVEL', 'INFO')
        logging.basicConfig(
            level=getattr(logging, log_level),
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler('logs/trading_system.log'),
                logging.StreamHandler()
            ]
        )

# Global configuration instance
config = Config()

# Agent Registry
AGENT_REGISTRY = {
    'market-intel-agent': {
        'class': 'MarketIntelligenceAgent',
        'enabled': config.agents.market_intel_enabled,
        'frequency': config.agents.market_intel_frequency,
        'dependencies': ['data-pipeline-agent']
    },
    'news-intel-agent': {
        'class': 'NewsIntelligenceAgent',
        'enabled': config.agents.news_intel_enabled,
        'frequency': config.agents.news_intel_frequency,
        'dependencies': ['data-pipeline-agent']
    },
    'fundamental-agent': {
        'class': 'FundamentalAnalysisAgent',
        'enabled': config.agents.fundamental_enabled,
        'frequency': config.agents.fundamental_frequency,
        'dependencies': ['data-pipeline-agent']
    },
    'technical-agent': {
        'class': 'TechnicalAnalysisAgent',
        'enabled': config.agents.technical_enabled,
        'frequency': config.agents.technical_frequency,
        'dependencies': ['data-pipeline-agent']
    },
    'risk-mgmt-agent': {
        'class': 'RiskManagementAgent',
        'enabled': config.agents.risk_mgmt_enabled,
        'frequency': config.agents.risk_mgmt_frequency,
        'dependencies': ['portfolio-mgmt-agent']
    },
    'execution-agent': {
        'class': 'ExecutionAgent',
        'enabled': config.agents.execution_enabled,
        'frequency': 1,  # Real-time execution
        'dependencies': ['risk-mgmt-agent']
    },
    'portfolio-mgmt-agent': {
        'class': 'PortfolioManagementAgent',
        'enabled': config.agents.portfolio_mgmt_enabled,
        'frequency': 300,  # 5 minutes
        'dependencies': ['market-intel-agent', 'fundamental-agent', 'technical-agent']
    },
    'alt-data-agent': {
        'class': 'AlternativeDataAgent',
        'enabled': config.agents.alt_data_enabled,
        'frequency': 1800,  # 30 minutes
        'dependencies': ['data-pipeline-agent']
    },
    'options-intel-agent': {
        'class': 'OptionsIntelligenceAgent',
        'enabled': config.agents.options_intel_enabled,
        'frequency': 60,
        'dependencies': ['data-pipeline-agent']
    },
    'crypto-intel-agent': {
        'class': 'CryptoIntelligenceAgent',
        'enabled': config.agents.crypto_intel_enabled,
        'frequency': 30,
        'dependencies': ['data-pipeline-agent']
    },
    'esg-intel-agent': {
        'class': 'ESGIntelligenceAgent',
        'enabled': config.agents.esg_intel_enabled,
        'frequency': 3600,  # 1 hour
        'dependencies': ['data-pipeline-agent']
    },
    'data-pipeline-agent': {
        'class': 'DataPipelineAgent',
        'enabled': True,  # Always enabled
        'frequency': 1,   # Real-time data processing
        'dependencies': []
    },
    'mlops-agent': {
        'class': 'MLOpsAgent',
        'enabled': True,
        'frequency': 900,  # 15 minutes
        'dependencies': ['data-pipeline-agent']
    },
    'monitoring-agent': {
        'class': 'SystemMonitoringAgent',
        'enabled': True,
        'frequency': 30,
        'dependencies': []
    },
    'orchestrator-agent': {
        'class': 'OrchestrationAgent',
        'enabled': True,
        'frequency': 5,  # High frequency coordination
        'dependencies': []  # Coordinates all other agents
    }
}

# News Sources Configuration
NEWS_SOURCES = {
    'tier1': {
        'bloomberg': {
            'url': 'https://api.bloomberg.com/news',
            'api_key': config.data_sources.bloomberg_news_key,
            'priority': 1,
            'latency_sla': 5  # seconds
        },
        'reuters': {
            'url': 'https://api.reuters.com/news',
            'api_key': config.data_sources.reuters_api_key,
            'priority': 1,
            'latency_sla': 10
        }
    },
    'tier2': {
        'newsapi': {
            'url': 'https://newsapi.org/v2/everything',
            'api_key': config.data_sources.news_api_key,
            'priority': 2,
            'latency_sla': 30
        },
        'yahoo_finance': {
            'url': 'https://query1.finance.yahoo.com/v1/finance/search',
            'priority': 2,
            'latency_sla': 60
        }
    },
    'social': {
        'twitter': {
            'url': 'https://api.twitter.com/2/tweets/search/stream',
            'bearer_token': config.data_sources.twitter_bearer_token,
            'priority': 3,
            'latency_sla': 15
        },
        'reddit': {
            'url': 'https://www.reddit.com/r/investing/new.json',
            'priority': 3,
            'latency_sla': 120
        }
    }
}

# Market Data Sources Configuration
MARKET_DATA_SOURCES = {
    'primary': {
        'bloomberg': {
            'url': 'wss://api.bloomberg.com/market-data',
            'api_key': config.data_sources.bloomberg_api_key,
            'latency_sla': 5
        },
        'refinitiv': {
            'url': 'wss://api.refinitiv.com/streaming',
            'api_key': config.data_sources.refinitiv_api_key,
            'latency_sla': 10
        }
    },
    'backup': {
        'iex_cloud': {
            'url': 'https://cloud.iexapis.com/stable',
            'token': config.data_sources.iex_cloud_token,
            'latency_sla': 100
        },
        'alpha_vantage': {
            'url': 'https://www.alphavantage.co/query',
            'api_key': config.data_sources.alpha_vantage_key,
            'latency_sla': 1000
        }
    }
} 