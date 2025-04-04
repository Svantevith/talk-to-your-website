from typing import Final, Union


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
    WELCOME_MESSAGE = """
        Hello, I am your personal assistant. 
       
        I can answer many general questions right away! 
        If you're looking for specific webpage content, enable web search retrieval.

        How may I help you?
    """

    # Example chat response essentially used when debugging the application to avoid more time-consiming text generation with LLM
    LOREM_IPSUM = """
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

    # Number of levels to crawl. Be cautious with values > 3, which can exponentially increase crawl size.
    # Sets upper limit for the slider.
    MAX_DEPTH: Final[int] = 5

    # Maximum number of pages to crawl.
    # Use None to crawl all top-level pages.
    # Sets upper limit for the slider.
    MAX_PAGES: Final[Union[None, int]] = 50

    # Minimum score (breadth-first search) for pages to be crawled.
    # Sets default value between 0.0 and 1.0 for slider.
    MIN_SCORE: Final[int] = 0.2

    # Experiment with keyword weights (best-first search) for optimal page prioritization.
    # Sets default value between 0.0 and 1.0 for the slider.
    KW_WEIGHT: Final[float] = 0.7

    # Minimum number of keywords to extract.
    KW_NUM: Final[int] = 3

    # Proportion of the extracted keywords to the number of tokens inside the prompt.
    KW_PROP: Final[float] = 0.05

    # Number of tokens to extract as a keyword.
    KW_NGRAM: Final[int] = 3

    # Deduplication factor for the keyword extraction. Lower values are better for deterministic search.
    KW_DEDUP: Final[float] = 0.2


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
    COLLECTION_NAME = "web_search_llm"

    # Collections are automatically saved & loaded from the memory.
    # Persistent client allows to handle changing context retrieval parameters without the need to create new client session.
    PERSIST_DIRECTORY: Final[str] = "./data/chroma_db"

    # Ollama embeddings model.
    EMBEDDINGS_MODEL: Final[str] = "all-minilm"

    # Size of the segment in sliding window chunking.
    # In Ollama, default context window size is 2048 tokens.
    WINDOW_SIZE: Final[int] = 256

    # Proportion of the overlap (10-30%) to the window size.
    # A slightly higher overlap (0.2) helps preserve continuity.
    WINDOW_OVERLAP: Final[float] = 0.2

    # Return top-k relevant documents to the LLM.
    # Sets upper limit for the slider.
    TOP_K: Final[int] = 12

    # Amount of documents to pass to MMR algorithm.
    # Sets upper limit for the slider.
    FETCH_K: Final[int] = 40

    # Enable to delete collection and associated directories when RAG object is destroyed.
    # Keep in mind that RAG object operates on single collection, thus only artifacts for that particular collection are deleted.
    # Be aware that if there is no collection available, offline capabilities are restricted to the general LLM knowledge.
    CLEANUP_COLLECTION: Final[bool] = False


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
            - You are an AI assistant tasked with providing detailed answers based solely on the given context.
            - Your goal is to analyze the information provided and formulate a comprehensive, well-structured response to the question.

        Prompt format:
            - Context will be passed as "Context:"
            - Question will be passed as "Question:"

        To answer the question:
            - Thoroughly analyze the context, identifying key information relevant to the question.
            - Organize your thoughts and plan your response to ensure a logical flow of information.
            - Formulate a detailed answer that directly addresses the question, using only the information provided in the context.
            - When the context supports an answer, ensure your response is clear, concise, and directly addresses the question.
            - When there is no context, just say you have no context and stop immediately.
            - If the context doesn't contain sufficient information to fully answer the question, state this clearly in your response.
            - Avoid explaining why you cannot answer or speculating about missing details. Simply state that you lack sufficient context when necessary.

        Format your response as follows:
            - Use clear, concise language.
            - Organize your answer into paragraphs for readability.
            - Use bullet points or numbered lists where appropriate to break down complex information.
            - If relevant, include any headings or subheadings to structure your response.
            - Ensure proper grammar, punctuation, and spelling throughout your answer.
            - Do not mention what you received in context, just focus on answering based on the context.

        Important: 
            - Base your entire response solely on the information provided in the context. 
            - Do not include any external knowledge or assumptions not present in the given text.
    """

    # Ollama LLM model variant.
    # Meta's Llama 3.2 is a very lightweight, instruction-tuned model with only 1B parameters.
    # English, German, French, Italian, Portuguese, Hindi, Spanish, and Thai are officially supported.
    MODEL_VARIANT: Final[str] = "llama3.2:1b"

    # Increasing the temperature (default 0.8) will make the model answer more creatively.
    # Lower values lead to more deterministic answers.
    # Sets default value between 0.0 and 1.0 for the slider.
    TEMPERATURE: Final[float] = 0.3

    # Maximum number of tokens to predict when generating text.
    # Sets upper limit for the slider.
    MAX_TOKENS: Final[int] = 2048
