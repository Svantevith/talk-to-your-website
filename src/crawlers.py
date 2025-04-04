from typing import Union
from collections.abc import AsyncGenerator
from langchain.docstore.document import Document
from crawl4ai import AsyncWebCrawler, BFSDeepCrawlStrategy, BestFirstCrawlingStrategy, BrowserConfig, CacheMode, CrawlResult, CrawlerRunConfig, KeywordRelevanceScorer, LXMLWebScrapingStrategy


class DeepCrawler:
    def __init__(
        self,
        start_url: str
    ):
        """
        DeepCrawler objects are used to explore websites beyond a single page and extract relevant content.

        Parameters
        ----------
            start_url : str
                Website to start crawling from.
        """
        # Basic configuration
        self.start_url = start_url

        # Configure browsing environment
        self.__browser_config = BrowserConfig(
            # Headless mode (invisible browser)
            headless=True,
            # Prints extra logs (helpful for debugging)
            verbose=False,
            # Disables images, possibly speeding up text-only crawls.
            text_mode=True,
            # Turns off certain background features for performance.
            light_mode=True
        )

    async def crawl(
        self,
        stream_mode: bool = True,
        max_depth: int = 0,
        max_pages: Union[None, int] = 1,
        min_score: Union[None, float] = 0.3,
        kw_weight: float = 0.7,
        kw_list: list[str] = []
    ) -> AsyncGenerator[Document]:
        """
        Initialize asynchronous crawl run to extract relevant content.

        Parameters
        ----------
            stream_mode : bool
                Whether to use real-time (True) or batch (False) processing.
            max_depth : int
                Number of pages to crawl beyond (be cautious with values > 3) the starting page.
            max_pages : int
                Limit total number of pages crawled or use None to crawl all pages. 
            min_score : float
                Skip pages with score below this value or use None to crawl any page.
            kw_list : list[str]
                List of keywords to use in relevance score calculation.
            kw_weight : float
                Prioritize keywords in overall score.

        Returns
        -------
            AsyncGenerator[Document]
                Generator of langchain's documents.
        """
        # Debugging
        print("\n=== Crawling starts ===")

        # Debugging
        print(
            f"=== URL: {self.start_url}, Streaming mode: {stream_mode}, Max depth: {max_depth}, Max pages: {max_pages}, Min score: {min_score}, Keywords list: {kw_list}, Keywords weight: {kw_weight} ===\n"
        )

        if kw_list:
            # Explore higher-scoring pages first
            crawl_strategy = BestFirstCrawlingStrategy(
                max_depth=max_depth,
                max_pages=max_pages,
                include_external=False,
                # Prioritize the most relevant pages
                url_scorer=KeywordRelevanceScorer(
                    # Calculate relevance based on various signals
                    keywords=kw_list,
                    # Control importance of the keywords to the overall score
                    weight=kw_weight
                )
            )

        else:
            # Eexplore all links at one depth (breadth-first) before moving deeper
            crawl_strategy = BFSDeepCrawlStrategy(
                # Be cautious with values > 3, which can exponentially increase crawl size
                max_depth=max_depth,
                # Maximum number of pages to crawl
                max_pages=max_pages,
                # Stay within the same domain
                include_external=False,
                # Skip URLs with scores below this value
                score_threshold=min_score
            )

        # Configure how each crawl run should behave
        run_config = CrawlerRunConfig(
            # Streaming (real-time one at a time, True) or Batch (all at once, False) processing
            stream=stream_mode,
            # Skip cache for this operation
            cache_mode=CacheMode.BYPASS,
            # Respect robots.txt for each URL
            check_robots_txt=True,
            # Remove all links pointing outside of the current domain.
            exclude_external_links=True,
            # Extract text-only content
            only_text=True,
            # Logs additional runtime details (overlaps with the BrowserConfig’s verbosity)
            verbose=True,
            # Use lxml library for faster HTML parsing to improve scraping performance, especially for large or complex pages
            scraping_strategy=LXMLWebScrapingStrategy(),
            # Configure crawling strategy to extract content precisely
            deep_crawl_strategy=crawl_strategy
        )

        # Initialize asynchronous crawler to run
        async with AsyncWebCrawler(config=self.__browser_config) as crawler:
            if stream_mode:
                # When streaming use 'async for' to process each result as soon as it is available
                async for result in await crawler.arun(self.start_url, config=run_config):
                    yield await self.__process_result(result)
            else:
                # Processing all results in batch after completion
                for result in await crawler.arun(self.start_url, config=run_config):
                    yield await self.__process_result(result)

        # Debugging
        print("=== Crawling finished ===\n")

    async def __process_result(self, result: CrawlResult) -> Document:
        """
        Process an individual crawling result. 

        Parameters
        ----------
            result : CrawlResult
                Crawl result to extract data from.

        Returns
        -------
            Document
                Langchain's document.
        """
        # Initialize page content
        page_content = ""

        if result.success:
            # Retrieve markdown
            page_content = result.markdown

        else:
            print(f"Failed to crawl {result.url}: {result.error_message}")

        # Structure metadata for the document
        structured_metadata = {
            "url": result.url,
            **{
                key: result.metadata[key]
                for key in ("title", "description")
                if result.metadata.get(key, "")
            }
        }

        # Return langchain's document
        return Document(
            page_content,
            metadata=structured_metadata
        )
