import asyncio
import os
from src.crawlers import DeepCrawler
from src.config import CrawlerConfig

async def main():
    url = "https://docs.pydantic.dev/latest/"
    crawler = DeepCrawler(
        # chromium_profile=CrawlerConfig.CHROMIUM_PROFILE,
    )
    docs = []

    async for doc in crawler.crawl(url, max_depth=3, max_pages=3, min_score=0.0):
        docs.append(doc)
    
    print("X")

if __name__ == "__main__":
    asyncio.run(main())