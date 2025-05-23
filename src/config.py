import os
from dotenv import load_dotenv
from typing import Final, Union

# Load environmental variables
load_dotenv()

class ImmutableMeta(type):
    """
    Metaclass that enforces immutability on class attributes and prevents instantiation.
    - Prevents instantiation of the class.
    - Prevents modifying class attributes.
    - Prevents deleting class attributes.
    """

    def __setattr__(cls, key, value):
        """Raises an error when trying to modify a class attribute."""
        raise TypeError(f"Cannot modify {cls.__name__}.{key}")

    def __delattr__(cls, key):
        """Raises an error when trying to delete a class attribute."""
        raise TypeError(f"Cannot delete {cls.__name__}.{key}")

    def __call__(cls, *args, **kwargs):
        """Raises an error when trying to instantiate the class."""
        raise TypeError(
            f"{cls.__name__} is a static class and cannot be instantiated.")


class ChatConfig(metaclass=ImmutableMeta):
    """
    Static configuration class for chat settings.
    This class is immutable and cannot be instantiated.
    """
    # Bot message displayed after crawling website pages
    WELCOME_MESSAGE: Final[str] = """
        Hello, I am your personal assistant. 
       
        I can answer many general questions right away! 
        If you're looking for specific webpage content, enable web search retrieval.

        How may I help you?
    """

    # Wait time in seconds (must be greater than or equal to 0.0) between each token in the response stream
    STREAM_INTERVAL: Final[float] = 0.05

    # Example chat response essentially used when debugging the application to avoid more time-consiming text generation with LLM
    LOREM_IPSUM: Final[str] = """
        Lorem ipsum dolor sit amet, consectetur adipiscing elit. Aenean gravida tincidunt eros, vitae mattis tellus fermentum sit amet.
        Suspendisse potenti. Vivamus elementum, magna eget tempor ornare, diam ex egestas lectus, accumsan pellentesque lorem lorem sed justo.
        Pellentesque sit amet tortor at enim consequat sollicitudin. Nunc et lacus ac tellus pellentesque consectetur id a velit.
        Orci varius natoque penatibus et magnis dis parturient montes, nascetur ridiculus mus. Integer congue vel quam nec auctor.
    """


class CrawlerConfig(metaclass=ImmutableMeta):
    """
    Static configuration class for crawler settings.
    This class is immutable and cannot be instantiated.

    Documentation
    -------------
        Deep crawling: https://docs.crawl4ai.com/core/deep-crawling/

    Requirements
    ------------
        Crawl4AI requires Microsoft Visual C++ 14.0 or higher version to be installed
    """
    # Streaming mode (True), Batch mode (False)
    STREAM_PROCESSING: Final[bool] = True

    # Number of levels to crawl.
    # Sets upper limit (must be greater than 1) for the slider.
    # Be cautious with values > 3, which can exponentially increase crawl size.
    MAX_DEPTH: Final[int] = 5

    # Maximum number of pages to crawl.
    # Sets upper limit (must be positive value greater or equal to 1) for the slider.
    MAX_PAGES: Final[Union[None, int]] = 50

    # Minimum score (breadth-first search) for pages to be crawled.
    # Sets default value (must be between 0.0 and 0.8) for the slider.
    MIN_SCORE: Final[int] = 0.2

    # Experiment with keyword weights (best-first search) for optimal page prioritization.
    # Sets default value (must be between 0.0 and 1.0) for the slider.
    KW_WEIGHT: Final[float] = 0.7

    # Minimum number (must be greater than or equal to 1) of keywords to extract.
    KW_NUM: Final[int] = 3

    # Proportion (must be between 0.0 and 0.3) of the extracted keywords to the number of tokens inside the prompt.
    KW_PROP: Final[float] = 0.05

    # Number of tokens (must be between 1 and 3) to extract as a keyword.
    KW_NGRAM: Final[int] = 3

    # Deduplication factor (must be between 0.0 and 1.0) for the keyword extraction. 
    # Lower values are better for deterministic search.
    KW_DEDUP: Final[float] = 0.2

    # User data directory to persist profile session data with managed Chromium browser.
    USER_DATA_DIR: Final[str] = os.getenv("CHROMIUM_PROFILE")


class RAGConfig(metaclass=ImmutableMeta):
    """
    Static configuration class for RAG settings.
    This class is immutable and cannot be instantiated.

    Documentation
    -------------
        Chroma clients: https://docs.trychroma.com/reference/python/client
        Chroma vector store: https://python.langchain.com/docs/integrations/vectorstores/chroma/
        Ollama embedding models: https://ollama.com/search?c=embedding

    Requirements
    ------------
        Chroma vector storage requires Ollama to be installed 
        Referenced Ollama models have to be pulled
    """
    # Timestamp is appended to ensure unique collection per client session
    COLLECTION_NAME: Final[str] = "web_search_llm"

    # Collections are automatically saved & loaded from the memory.
    # Persistent client allows to handle changing context retrieval parameters without the need to create new client session.
    PERSIST_DIRECTORY: Final[str] = os.getenv("CHROMA_DIRECTORY")

    # Return top-k relevant documents to the LLM.
    # Sets upper limit (must be greater than or equal to 3) for the slider.
    TOP_K: Final[int] = 10

    # Amount of documents to pass to MMR algorithm.
    # Sets upper limit (must be greater than or equal to 30) for the slider.
    FETCH_K: Final[int] = 40


class LLMConfig(metaclass=ImmutableMeta):
    """
    Static configuration class for LLM settings.
    This class is immutable and cannot be instantiated.

    Documentation
    -------------
        Langchain chat model: https://python.langchain.com/api_reference/ollama/chat_models/langchain_ollama.chat_models.ChatOllama.html#
        Ollama models: https://ollama.com/search/

    Requirements
    ------------
        Referenced Ollama models have to be pulled
    """

    # System prompt for the assistant to control its response behaviour
    SYSTEM_PROMPT: Final[str] = """
        Task description: 
            - You are an AI assistant tasked with providing detailed answers based primarily on the given context.
            - Your goal is to analyze the information provided and formulate a comprehensive, well-structured response to the question.

        Prompt format:
            - Context will be passed as "Context:"
            - Question will be passed as "Question:"

        To answer the question:
            - Organize your thoughts and plan your response to ensure a logical flow of information. 
            - If context is sufficient, rely on it to generate a detailed and coherent response:
                - Thoroughly analyze the context and identify key information relevant to the question.
                - Do not include any external knowledge or assumptions not present in the given text that could mislead the user.
            - If context is missing or it does not contain enough information, use your general knowledge to complete the response:
                - Explicitly state that the context was not provided or lacks sufficient detail.
                - Emphasize that the answer relies on your general knowledge and is not supported by the context.

        Format your response as follows:
            - Use clear, concise language.
            - Organize your answer into paragraphs for readability.
            - Use bullet points or numbered lists where appropriate to break down complex information.
            - If relevant, include headings or subheadings to structure your response.
            - Ensure proper grammar, punctuation, and spelling throughout your answer.
            - Avoid mentioning what you received in the context, just focus on answering based on the context (if present).
    """

    # Increasing the temperature (default 0.8) will make the model answer more creatively.
    # Lower values lead to more deterministic answers.
    # Sets default value (must be between 0.0 and 1.0) for the slider.
    TEMPERATURE: Final[float] = 0.3

    # Maximum number of tokens to predict when generating text.
    # Sets upper limit (must be a multiple of 128 greater than or equal to 128) for the slider.
    MAX_TOKENS: Final[int] = 2048

    # Timeout (must be greater than or equal to 1) in seconds for the request stream.
    # Requests are sent to the local Ollama client, avoiding external API calls.
    TIMEOUT: Final[int] = 30

    # By default models are kept in memory for 5 minutes before being unloaded.
    # This allows for quicker response times if you're making numerous requests (like in a chat conversation) to the LLM.
    KEEP_ALIVE: Final[int] = 3600
