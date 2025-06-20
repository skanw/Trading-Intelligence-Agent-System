"""
News Intelligence Agent - Comprehensive Real-time News Analysis for Trading
"""

import asyncio
import aiohttp
import json
import re
from datetime import datetime, timezone, timedelta
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, asdict
import numpy as np
from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline
import torch
import yfinance as yf
import tweepy
import praw
from bs4 import BeautifulSoup
import feedparser

from .base_agent import BaseAgent, AgentMessage, AgentSignal
from ..config import config, NEWS_SOURCES


@dataclass
class NewsItem:
    """Structured news item with analysis"""
    title: str
    content: str
    source: str
    url: str
    timestamp: datetime
    symbols_mentioned: List[str]
    sentiment_score: float  # -1.0 to 1.0
    relevance_score: float  # 0.0 to 1.0
    urgency_level: int      # 1-5, 5 being most urgent
    category: str           # earnings, m&a, regulatory, etc.
    market_impact: str      # 'HIGH', 'MEDIUM', 'LOW'
    confidence: float       # 0.0 to 1.0
    reasoning: str
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class NewsAlert:
    """High-priority news alert"""
    symbol: str
    news_items: List[NewsItem]
    aggregate_sentiment: float
    alert_type: str  # 'BREAKING', 'EARNINGS', 'MERGER', 'REGULATORY'
    priority: int
    action_recommendation: str
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class NewsIntelligenceAgent(BaseAgent):
    """
    Advanced news intelligence agent for real-time news analysis and trading signals
    """
    
    def __init__(self, agent_id: str = 'news-intel-agent', config_override: Optional[Dict] = None):
        super().__init__(agent_id, config_override)
        
        # News processing models
        self.sentiment_model = None
        self.tokenizer = None
        self.ner_pipeline = None
        
        # Data sources
        self.news_sources = NEWS_SOURCES
        self.session = None
        
        # Twitter API
        self.twitter_client = None
        
        # Reddit API
        self.reddit_client = None
        
        # Coverage universe
        self.covered_symbols = set(config.trading.large_cap_symbols + 
                                 config.trading.mid_cap_symbols + 
                                 config.trading.small_cap_symbols)
        
        # Symbol mappings for better detection
        self.symbol_mappings = {}
        self.company_names = {}
        
        # News processing configuration
        self.min_sentiment_threshold = config.agents.news_sentiment_threshold
        self.max_articles_per_source = 100
        self.lookback_hours = 24
        
        # Real-time tracking
        self.processed_urls = set()
        self.last_fetch_times = {}
        
    async def agent_initialize(self):
        """Initialize news intelligence agent"""
        try:
            # Load sentiment analysis model (FinBERT for financial sentiment)
            self.tokenizer = AutoTokenizer.from_pretrained("ProsusAI/finbert")
            self.sentiment_model = AutoModelForSequenceClassification.from_pretrained("ProsusAI/finbert")
            
            # Setup NER pipeline for entity extraction
            self.ner_pipeline = pipeline(
                "ner", 
                model="dbmdz/bert-large-cased-finetuned-conll03-english",
                aggregation_strategy="simple"
            )
            
            # Initialize HTTP session
            self.session = aiohttp.ClientSession()
            
            # Setup Twitter API
            if config.data_sources.twitter_bearer_token:
                self.twitter_client = tweepy.Client(
                    bearer_token=config.data_sources.twitter_bearer_token,
                    wait_on_rate_limit=True
                )
            
            # Setup Reddit API
            if config.data_sources.reddit_client_id:
                self.reddit_client = praw.Reddit(
                    client_id=config.data_sources.reddit_client_id,
                    client_secret=config.data_sources.reddit_client_secret,
                    user_agent="TradingNewsBot/1.0"
                )
            
            # Build symbol mappings
            await self._build_symbol_mappings()
            
            self.logger.info("News Intelligence Agent initialized successfully")
            
        except Exception as e:
            self.logger.error(f"Failed to initialize News Intelligence Agent: {e}")
            raise
    
    async def execute(self):
        """Main execution logic - fetch and analyze news"""
        try:
            # Fetch news from all sources
            all_news = await self._fetch_all_news()
            
            # Process and analyze news
            processed_news = []
            for news_item in all_news:
                if news_item.url not in self.processed_urls:
                    analyzed_item = await self._analyze_news_item(news_item)
                    if analyzed_item:
                        processed_news.append(analyzed_item)
                        self.processed_urls.add(news_item.url)
            
            # Generate alerts for high-impact news
            alerts = await self._generate_news_alerts(processed_news)
            
            # Send alerts to relevant agents
            for alert in alerts:
                await self._send_news_alert(alert)
            
            # Cache processed news
            self._cache_news_data(processed_news)
            
            # Clean up old URLs from processed set
            self._cleanup_processed_urls()
            
            self.logger.info(f"Processed {len(processed_news)} news items, generated {len(alerts)} alerts")
            
        except Exception as e:
            self.logger.error(f"Error in news intelligence execution: {e}")
            self.metrics['errors'] += 1
    
    async def _fetch_all_news(self) -> List[NewsItem]:
        """Fetch news from all configured sources"""
        all_news = []
        
        # Fetch from Tier 1 sources (Bloomberg, Reuters)
        for source_name, source_config in self.news_sources['tier1'].items():
            try:
                news_items = await self._fetch_premium_news(source_name, source_config)
                all_news.extend(news_items)
            except Exception as e:
                self.logger.error(f"Error fetching from {source_name}: {e}")
        
        # Fetch from Tier 2 sources (NewsAPI, Yahoo Finance)
        for source_name, source_config in self.news_sources['tier2'].items():
            try:
                news_items = await self._fetch_standard_news(source_name, source_config)
                all_news.extend(news_items)
            except Exception as e:
                self.logger.error(f"Error fetching from {source_name}: {e}")
        
        # Fetch from social sources
        if self.twitter_client:
            try:
                twitter_news = await self._fetch_twitter_news()
                all_news.extend(twitter_news)
            except Exception as e:
                self.logger.error(f"Error fetching Twitter news: {e}")
        
        if self.reddit_client:
            try:
                reddit_news = await self._fetch_reddit_news()
                all_news.extend(reddit_news)
            except Exception as e:
                self.logger.error(f"Error fetching Reddit news: {e}")
        
        return all_news
    
    async def _fetch_premium_news(self, source_name: str, source_config: Dict) -> List[NewsItem]:
        """Fetch news from premium sources like Bloomberg, Reuters"""
        news_items = []
        
        if source_name == 'bloomberg' and source_config.get('api_key'):
            # Bloomberg API integration (would require actual Bloomberg Terminal access)
            # For now, we'll simulate with RSS feeds
            news_items = await self._fetch_rss_news(
                'https://feeds.bloomberg.com/markets/news.rss',
                'Bloomberg'
            )
        
        elif source_name == 'reuters' and source_config.get('api_key'):
            # Reuters API integration
            news_items = await self._fetch_rss_news(
                'https://feeds.reuters.com/reuters/businessNews',
                'Reuters'
            )
        
        return news_items
    
    async def _fetch_standard_news(self, source_name: str, source_config: Dict) -> List[NewsItem]:
        """Fetch news from standard sources"""
        news_items = []
        
        if source_name == 'newsapi' and source_config.get('api_key'):
            news_items = await self._fetch_newsapi(source_config)
        
        elif source_name == 'yahoo_finance':
            news_items = await self._fetch_yahoo_finance_news()
        
        return news_items
    
    async def _fetch_newsapi(self, source_config: Dict) -> List[NewsItem]:
        """Fetch news from NewsAPI"""
        news_items = []
        
        try:
            # Build query for covered symbols
            query = ' OR '.join([f'"{symbol}"' for symbol in list(self.covered_symbols)[:50]])
            
            url = source_config['url']
            params = {
                'q': query,
                'apiKey': source_config['api_key'],
                'language': 'en',
                'sortBy': 'publishedAt',
                'pageSize': 100,
                'from': (datetime.now() - timedelta(hours=self.lookback_hours)).isoformat()
            }
            
            async with self.session.get(url, params=params) as response:
                if response.status == 200:
                    data = await response.json()
                    
                    for article in data.get('articles', [])[:self.max_articles_per_source]:
                        if article.get('title') and article.get('description'):
                            news_item = NewsItem(
                                title=article['title'],
                                content=article.get('description', '') + ' ' + article.get('content', ''),
                                source='NewsAPI',
                                url=article.get('url', ''),
                                timestamp=datetime.fromisoformat(article['publishedAt'].replace('Z', '+00:00')),
                                symbols_mentioned=[],
                                sentiment_score=0.0,
                                relevance_score=0.0,
                                urgency_level=1,
                                category='general',
                                market_impact='LOW',
                                confidence=0.0,
                                reasoning=''
                            )
                            news_items.append(news_item)
                
        except Exception as e:
            self.logger.error(f"Error fetching NewsAPI data: {e}")
        
        return news_items
    
    async def _fetch_yahoo_finance_news(self) -> List[NewsItem]:
        """Fetch news from Yahoo Finance"""
        news_items = []
        
        try:
            for symbol in list(self.covered_symbols)[:20]:  # Process in batches
                try:
                    ticker = yf.Ticker(symbol)
                    news = ticker.news
                    
                    for article in news[:5]:  # Top 5 articles per symbol
                        news_item = NewsItem(
                            title=article.get('title', ''),
                            content=article.get('summary', ''),
                            source='Yahoo Finance',
                            url=article.get('link', ''),
                            timestamp=datetime.fromtimestamp(article.get('providerPublishTime', 0), tz=timezone.utc),
                            symbols_mentioned=[symbol],
                            sentiment_score=0.0,
                            relevance_score=0.8,  # High relevance since it's symbol-specific
                            urgency_level=2,
                            category='general',
                            market_impact='MEDIUM',
                            confidence=0.0,
                            reasoning=''
                        )
                        news_items.append(news_item)
                        
                except Exception as e:
                    self.logger.debug(f"Error fetching Yahoo Finance news for {symbol}: {e}")
                    continue
                
        except Exception as e:
            self.logger.error(f"Error in Yahoo Finance news fetch: {e}")
        
        return news_items
    
    async def _fetch_rss_news(self, rss_url: str, source_name: str) -> List[NewsItem]:
        """Fetch news from RSS feeds"""
        news_items = []
        
        try:
            async with self.session.get(rss_url) as response:
                if response.status == 200:
                    content = await response.text()
                    feed = feedparser.parse(content)
                    
                    for entry in feed.entries[:self.max_articles_per_source]:
                        news_item = NewsItem(
                            title=entry.get('title', ''),
                            content=entry.get('summary', ''),
                            source=source_name,
                            url=entry.get('link', ''),
                            timestamp=datetime.fromtimestamp(
                                entry.get('published_parsed') or entry.get('updated_parsed') or 0,
                                tz=timezone.utc
                            ),
                            symbols_mentioned=[],
                            sentiment_score=0.0,
                            relevance_score=0.0,
                            urgency_level=1,
                            category='general',
                            market_impact='LOW',
                            confidence=0.0,
                            reasoning=''
                        )
                        news_items.append(news_item)
                        
        except Exception as e:
            self.logger.error(f"Error fetching RSS news from {rss_url}: {e}")
        
        return news_items
    
    async def _fetch_twitter_news(self) -> List[NewsItem]:
        """Fetch financial news from Twitter"""
        news_items = []
        
        try:
            # Financial Twitter accounts and hashtags
            query = '(from:DeItaone OR from:FirstSquawk OR from:zerohedge OR #earnings OR #breakingnews) -is:retweet'
            
            tweets = tweepy.Paginator(
                self.twitter_client.search_recent_tweets,
                query=query,
                max_results=100,
                tweet_fields=['created_at', 'author_id', 'public_metrics']
            ).flatten(limit=200)
            
            for tweet in tweets:
                news_item = NewsItem(
                    title=tweet.text[:100] + '...' if len(tweet.text) > 100 else tweet.text,
                    content=tweet.text,
                    source='Twitter',
                    url=f'https://twitter.com/user/status/{tweet.id}',
                    timestamp=tweet.created_at,
                    symbols_mentioned=[],
                    sentiment_score=0.0,
                    relevance_score=0.0,
                    urgency_level=3,  # Twitter news can be very timely
                    category='social',
                    market_impact='MEDIUM',
                    confidence=0.0,
                    reasoning=''
                )
                news_items.append(news_item)
                
        except Exception as e:
            self.logger.error(f"Error fetching Twitter news: {e}")
        
        return news_items
    
    async def _fetch_reddit_news(self) -> List[NewsItem]:
        """Fetch financial news from Reddit"""
        news_items = []
        
        try:
            # Monitor key financial subreddits
            subreddits = ['investing', 'stocks', 'SecurityAnalysis', 'ValueInvesting', 'wallstreetbets']
            
            for subreddit_name in subreddits:
                subreddit = self.reddit_client.subreddit(subreddit_name)
                
                # Get hot posts
                for submission in subreddit.hot(limit=20):
                    if submission.score > 50:  # Filter by popularity
                        news_item = NewsItem(
                            title=submission.title,
                            content=submission.selftext[:500] if submission.selftext else submission.title,
                            source=f'Reddit/{subreddit_name}',
                            url=submission.url,
                            timestamp=datetime.fromtimestamp(submission.created_utc, tz=timezone.utc),
                            symbols_mentioned=[],
                            sentiment_score=0.0,
                            relevance_score=0.0,
                            urgency_level=2,
                            category='social',
                            market_impact='LOW',
                            confidence=0.0,
                            reasoning=''
                        )
                        news_items.append(news_item)
                        
        except Exception as e:
            self.logger.error(f"Error fetching Reddit news: {e}")
        
        return news_items
    
    async def _analyze_news_item(self, news_item: NewsItem) -> Optional[NewsItem]:
        """Comprehensive analysis of a news item"""
        try:
            # Extract mentioned symbols
            news_item.symbols_mentioned = self._extract_symbols(news_item.title + ' ' + news_item.content)
            
            # Skip if no relevant symbols found
            if not news_item.symbols_mentioned:
                return None
            
            # Analyze sentiment
            news_item.sentiment_score = await self._analyze_sentiment(news_item.title + ' ' + news_item.content)
            
            # Calculate relevance score
            news_item.relevance_score = self._calculate_relevance(news_item)
            
            # Determine category
            news_item.category = self._categorize_news(news_item.title + ' ' + news_item.content)
            
            # Assess market impact
            news_item.market_impact, news_item.urgency_level = self._assess_market_impact(news_item)
            
            # Calculate confidence
            news_item.confidence = self._calculate_confidence(news_item)
            
            # Generate reasoning
            news_item.reasoning = self._generate_reasoning(news_item)
            
            return news_item
            
        except Exception as e:
            self.logger.error(f"Error analyzing news item: {e}")
            return None
    
    def _extract_symbols(self, text: str) -> List[str]:
        """Extract stock symbols from text"""
        symbols = []
        
        # Direct symbol matching
        symbol_pattern = r'\b([A-Z]{1,5})\b'
        potential_symbols = re.findall(symbol_pattern, text.upper())
        
        for symbol in potential_symbols:
            if symbol in self.covered_symbols and len(symbol) <= 5:
                symbols.append(symbol)
        
        # Company name matching
        for symbol, company_name in self.company_names.items():
            if company_name.lower() in text.lower():
                symbols.append(symbol)
        
        return list(set(symbols))
    
    async def _analyze_sentiment(self, text: str) -> float:
        """Analyze sentiment using FinBERT"""
        try:
            # Tokenize and analyze
            inputs = self.tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
            
            with torch.no_grad():
                outputs = self.sentiment_model(**inputs)
                predictions = torch.nn.functional.softmax(outputs.logits, dim=-1)
            
            # FinBERT outputs: negative, neutral, positive
            negative, neutral, positive = predictions[0].tolist()
            
            # Convert to sentiment score (-1 to 1)
            sentiment_score = positive - negative
            
            return sentiment_score
            
        except Exception as e:
            self.logger.error(f"Error in sentiment analysis: {e}")
            return 0.0
    
    def _calculate_relevance(self, news_item: NewsItem) -> float:
        """Calculate relevance score based on multiple factors"""
        relevance = 0.0
        
        # Source reliability weight
        source_weights = {
            'Bloomberg': 1.0,
            'Reuters': 1.0,
            'Yahoo Finance': 0.8,
            'NewsAPI': 0.7,
            'Twitter': 0.6,
            'Reddit': 0.5
        }
        
        relevance += source_weights.get(news_item.source, 0.5)
        
        # Symbol mention weight
        if len(news_item.symbols_mentioned) > 0:
            relevance += 0.3
        
        # Recency weight
        hours_old = (datetime.now(timezone.utc) - news_item.timestamp).total_seconds() / 3600
        recency_weight = max(0, 1 - (hours_old / 24))  # Decay over 24 hours
        relevance += recency_weight * 0.2
        
        return min(relevance, 1.0)
    
    def _categorize_news(self, text: str) -> str:
        """Categorize news into different types"""
        text_lower = text.lower()
        
        earnings_keywords = ['earnings', 'quarterly', 'q1', 'q2', 'q3', 'q4', 'eps', 'revenue', 'guidance']
        if any(keyword in text_lower for keyword in earnings_keywords):
            return 'earnings'
        
        ma_keywords = ['merger', 'acquisition', 'takeover', 'buyout', 'deal']
        if any(keyword in text_lower for keyword in ma_keywords):
            return 'merger_acquisition'
        
        regulatory_keywords = ['fda', 'sec', 'regulatory', 'approval', 'investigation', 'lawsuit']
        if any(keyword in text_lower for keyword in regulatory_keywords):
            return 'regulatory'
        
        leadership_keywords = ['ceo', 'cfo', 'president', 'management', 'executive', 'appointed', 'resigned']
        if any(keyword in text_lower for keyword in leadership_keywords):
            return 'management'
        
        product_keywords = ['launch', 'product', 'innovation', 'patent', 'technology']
        if any(keyword in text_lower for keyword in product_keywords):
            return 'product'
        
        analyst_keywords = ['analyst', 'upgrade', 'downgrade', 'target price', 'recommendation']
        if any(keyword in text_lower for keyword in analyst_keywords):
            return 'analyst'
        
        return 'general'
    
    def _assess_market_impact(self, news_item: NewsItem) -> Tuple[str, int]:
        """Assess potential market impact and urgency"""
        impact_score = 0
        
        # Category impact weights
        category_weights = {
            'earnings': 3,
            'merger_acquisition': 4,
            'regulatory': 3,
            'management': 2,
            'product': 2,
            'analyst': 2,
            'general': 1
        }
        
        impact_score += category_weights.get(news_item.category, 1)
        
        # Sentiment impact
        impact_score += abs(news_item.sentiment_score) * 2
        
        # Source impact
        if news_item.source in ['Bloomberg', 'Reuters']:
            impact_score += 2
        elif news_item.source in ['Yahoo Finance', 'NewsAPI']:
            impact_score += 1
        
        # Determine impact level and urgency
        if impact_score >= 6:
            return 'HIGH', 5
        elif impact_score >= 4:
            return 'MEDIUM', 3
        else:
            return 'LOW', 1
    
    def _calculate_confidence(self, news_item: NewsItem) -> float:
        """Calculate confidence in the analysis"""
        confidence = 0.5  # Base confidence
        
        # Source reliability
        if news_item.source in ['Bloomberg', 'Reuters']:
            confidence += 0.3
        elif news_item.source in ['Yahoo Finance', 'NewsAPI']:
            confidence += 0.2
        
        # Content length (more content = higher confidence)
        content_length = len(news_item.content)
        if content_length > 500:
            confidence += 0.1
        elif content_length > 200:
            confidence += 0.05
        
        # Symbol mention clarity
        if len(news_item.symbols_mentioned) == 1:
            confidence += 0.1  # Clear single symbol mention
        elif len(news_item.symbols_mentioned) > 3:
            confidence -= 0.1  # Too many symbols might be noise
        
        return min(confidence, 1.0)
    
    def _generate_reasoning(self, news_item: NewsItem) -> str:
        """Generate human-readable reasoning for the analysis"""
        reasons = []
        
        # Sentiment reasoning
        if news_item.sentiment_score > 0.3:
            reasons.append(f"Positive sentiment ({news_item.sentiment_score:.2f})")
        elif news_item.sentiment_score < -0.3:
            reasons.append(f"Negative sentiment ({news_item.sentiment_score:.2f})")
        
        # Category reasoning
        if news_item.category in ['earnings', 'merger_acquisition']:
            reasons.append(f"High-impact {news_item.category.replace('_', ' ')} news")
        
        # Source reasoning
        if news_item.source in ['Bloomberg', 'Reuters']:
            reasons.append("Premium news source")
        
        # Market impact reasoning
        if news_item.market_impact == 'HIGH':
            reasons.append("Expected high market impact")
        
        return "; ".join(reasons) if reasons else "Standard news analysis"
    
    async def _generate_news_alerts(self, news_items: List[NewsItem]) -> List[NewsAlert]:
        """Generate high-priority alerts from processed news"""
        alerts = []
        
        # Group news by symbol
        symbol_news = {}
        for item in news_items:
            for symbol in item.symbols_mentioned:
                if symbol not in symbol_news:
                    symbol_news[symbol] = []
                symbol_news[symbol].append(item)
        
        # Generate alerts for symbols with significant news
        for symbol, items in symbol_news.items():
            high_impact_items = [item for item in items if item.market_impact == 'HIGH']
            
            if high_impact_items:
                # Calculate aggregate sentiment
                avg_sentiment = np.mean([item.sentiment_score for item in high_impact_items])
                
                # Determine alert type
                alert_type = 'BREAKING'
                if any(item.category == 'earnings' for item in high_impact_items):
                    alert_type = 'EARNINGS'
                elif any(item.category == 'merger_acquisition' for item in high_impact_items):
                    alert_type = 'MERGER'
                elif any(item.category == 'regulatory' for item in high_impact_items):
                    alert_type = 'REGULATORY'
                
                # Generate action recommendation
                action_recommendation = self._generate_action_recommendation(symbol, high_impact_items, avg_sentiment)
                
                alert = NewsAlert(
                    symbol=symbol,
                    news_items=high_impact_items,
                    aggregate_sentiment=avg_sentiment,
                    alert_type=alert_type,
                    priority=max(item.urgency_level for item in high_impact_items),
                    action_recommendation=action_recommendation
                )
                
                alerts.append(alert)
        
        return alerts
    
    def _generate_action_recommendation(self, symbol: str, news_items: List[NewsItem], sentiment: float) -> str:
        """Generate trading action recommendation based on news analysis"""
        if sentiment > 0.5:
            return f"POSITIVE: Consider buying {symbol} - strong positive sentiment from {len(news_items)} high-impact news items"
        elif sentiment < -0.5:
            return f"NEGATIVE: Consider selling/avoiding {symbol} - strong negative sentiment from {len(news_items)} high-impact news items"
        else:
            return f"NEUTRAL: Monitor {symbol} closely - mixed signals from {len(news_items)} high-impact news items"
    
    async def _send_news_alert(self, alert: NewsAlert):
        """Send news alert to relevant agents"""
        try:
            # Send to risk management
            await self.send_message(
                'risk-mgmt-agent',
                'news_alert',
                alert.to_dict(),
                priority=alert.priority
            )
            
            # Send to portfolio management
            await self.send_message(
                'portfolio-mgmt-agent',
                'news_alert',
                alert.to_dict(),
                priority=alert.priority
            )
            
            # Send to orchestrator for coordination
            await self.send_message(
                'orchestrator-agent',
                'news_alert',
                alert.to_dict(),
                priority=alert.priority
            )
            
            # Generate trading signal if confidence is high
            if len(alert.news_items) > 0 and alert.news_items[0].confidence > 0.7:
                signal_type = 'BUY' if alert.aggregate_sentiment > 0.3 else 'SELL' if alert.aggregate_sentiment < -0.3 else 'HOLD'
                
                signal = AgentSignal(
                    symbol=alert.symbol,
                    signal_type=signal_type,
                    strength=min(abs(alert.aggregate_sentiment), 1.0),
                    confidence=np.mean([item.confidence for item in alert.news_items]),
                    time_horizon='SHORT',
                    reasoning=f"News-based signal: {alert.action_recommendation}",
                    data_sources=[item.source for item in alert.news_items]
                )
                
                await self.broadcast_signal(signal)
            
        except Exception as e:
            self.logger.error(f"Error sending news alert: {e}")
    
    def _cache_news_data(self, news_items: List[NewsItem]):
        """Cache processed news data for other agents"""
        try:
            # Cache by symbol
            symbol_news = {}
            for item in news_items:
                for symbol in item.symbols_mentioned:
                    if symbol not in symbol_news:
                        symbol_news[symbol] = []
                    symbol_news[symbol].append(item.to_dict())
            
            for symbol, items in symbol_news.items():
                self.cache_data(f"news:{symbol}", items, expiry=7200)  # 2 hours
            
            # Cache overall market sentiment
            if news_items:
                market_sentiment = np.mean([item.sentiment_score for item in news_items])
                self.cache_data("market_sentiment", {
                    'sentiment': market_sentiment,
                    'news_count': len(news_items),
                    'timestamp': datetime.now(timezone.utc).isoformat()
                }, expiry=1800)  # 30 minutes
                
        except Exception as e:
            self.logger.error(f"Error caching news data: {e}")
    
    def _cleanup_processed_urls(self):
        """Clean up old URLs from processed set to prevent memory bloat"""
        if len(self.processed_urls) > 10000:
            # Keep only recent URLs (this is a simplified cleanup)
            self.processed_urls = set(list(self.processed_urls)[-5000:])
    
    async def _build_symbol_mappings(self):
        """Build mappings between symbols and company names"""
        try:
            for symbol in list(self.covered_symbols)[:100]:  # Process in batches
                try:
                    ticker = yf.Ticker(symbol)
                    info = ticker.info
                    
                    company_name = info.get('longName') or info.get('shortName')
                    if company_name:
                        self.company_names[symbol] = company_name
                        # Add common variations
                        short_name = company_name.split()[0] if company_name else ''
                        if short_name and len(short_name) > 3:
                            self.symbol_mappings[short_name.lower()] = symbol
                            
                except Exception as e:
                    self.logger.debug(f"Could not get info for {symbol}: {e}")
                    continue
                    
        except Exception as e:
            self.logger.error(f"Error building symbol mappings: {e}")
    
    async def handle_message(self, message: AgentMessage):
        """Handle incoming messages from other agents"""
        try:
            if message.message_type == 'request_news_analysis':
                # Provide specific news analysis for a symbol
                symbol = message.data.get('symbol')
                if symbol:
                    cached_news = self.get_cached_data(f"news:{symbol}")
                    if cached_news:
                        await self.send_message(
                            message.agent_id,
                            'news_analysis_response',
                            {'symbol': symbol, 'news_data': cached_news},
                            correlation_id=message.correlation_id
                        )
            
            elif message.message_type == 'market_sentiment_request':
                # Provide overall market sentiment
                market_sentiment = self.get_cached_data("market_sentiment")
                if market_sentiment:
                    await self.send_message(
                        message.agent_id,
                        'market_sentiment_response',
                        market_sentiment,
                        correlation_id=message.correlation_id
                    )
                    
        except Exception as e:
            self.logger.error(f"Error handling message: {e}")
    
    async def agent_cleanup(self):
        """Cleanup agent resources"""
        if self.session:
            await self.session.close()
        
        self.logger.info("News Intelligence Agent cleaned up successfully") 