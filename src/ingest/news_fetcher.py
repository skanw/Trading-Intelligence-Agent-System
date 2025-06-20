import asyncio, logging
from datetime import datetime, timezone, timedelta
from typing import Dict, List

from newsapi import NewsApiClient
from redis.asyncio import Redis
from src.config import settings

log = logging.getLogger(__name__)

class NewsFetcher:
    def __init__(self, redis: Redis):
        self.client = NewsApiClient(api_key=settings.NEWS_API_KEY)
        self.redis = redis
        self.tickers = settings.TRACKED_TICKERS.split(",")

    async def fetch_loop(self, interval: int = 60):
        """Poll NewsAPI every `interval` seconds and push JSON into Redis stream."""
        while True:
            try:
                since = (datetime.now(timezone.utc) - timedelta(minutes=15)).isoformat()
                q = " OR ".join(self.tickers)
                res = self.client.get_everything(
                    q=q,
                    from_param=since,
                    language="en",
                    sort_by="publishedAt",
                    page_size=100,
                )
                for art in res["articles"]:
                    await self.redis.xadd("news_raw", art)
                log.info("Fetched %s new articles", len(res["articles"]))
            except Exception as e:
                log.exception("Fetcher error: %s", e)
            await asyncio.sleep(interval)

async def main():
    redis = Redis.from_url(settings.REDIS_URL, decode_responses=True)
    await NewsFetcher(redis).fetch_loop()

if __name__ == "__main__":
    asyncio.run(main()) 