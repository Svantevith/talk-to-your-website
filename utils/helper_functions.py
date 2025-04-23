import re
import httpx
import socket
import requests
from datetime import datetime
from yake import KeywordExtractor


def list_ollama_models(families: set[str] = {}) -> list[str]:
    """
    Use Ollama's API to retrieve list of locally available (i.e. pulled) models belonging to the specified families. 
    Architecture details (including the family) for each model can be found in the library: https://ollama.com/search

    Parmeters
    ---------
        families : set[str]
            Set of model families to filter results.
    
    Returns
    -------
        List of model names.
    """
    # The /api/tags endpoint is specified in the documentation: https://github.com/ollama/ollama/blob/main/docs/api.md
    endpoint = "http://localhost:11434/api/tags"
    try:
        # Send GET request
        response = httpx.get(endpoint)
        response.raise_for_status()
        return [
            model["name"] for model in response.json().get("models", [])
            if not families or model["details"]["family"] in families
        ]
    except (ConnectionRefusedError, httpx.HTTPStatusError) as e:
        print(
            f"=== Cannot retrieve list of models from the http://localhost:11434/api/tags endpoint: {e}")
        return []


def is_valid_collection_name(name: str) -> bool:
    """
    Validates if the given name is a proper ChromaDB collection name.

    Rules:
    - Only lowercase letters, numbers, and underscores
    - Must start and end with alphanumeric character
    - Length between 1 and 63 characters
    - No consecutive underscores

    Parameters
    ----------
    name : str
        The collection name to validate.

    Returns
    -------
    bool
        True if valid, False otherwise.
    """
    if not (1 <= len(name) <= 63):
        return False

    # Must match pattern: starts/ends with alphanumeric, only _ allowed in between
    pattern = r'^[a-z0-9](?!.*__)[a-z0-9_]*[a-z0-9]$'
    return re.match(pattern, name) is not None


def connected_to_internet(host: str = "8.8.8.8", port: int = 443, timeout: float = 1.0) -> bool:
    """
    Check if connection to internet can be established.

    Parameters
    ----------
        host: str 
            Host to connect with, by default 8.8.8.8 (google-public-dns-a.google.com)
        port : int 
            Port to use, by default 443 (firewalls, proxies, and VPNs must allow this port to allow secure HTTP traffic).
        timeout : float
            The maximum number of seconds to wait while trying to connect to a host.

    Returns
    -------
        bool
            Indicates internet connectivity.
    """
    try:
        with socket.create_connection(address=(host, port), timeout=timeout):
            return True
    except socket.error as e:
        print(f"=== Connection with the host {host} over port {port} could not be established: {e} ===")
        return False


def get_timestamp() -> str:
    """
    Retrieve timestamp with precision to miliseconds. 

    Returns 
    -------
        str
            Formatted timestamp.
    """
    return datetime.now().strftime(r'%Y%m%d%H%M%S%f')[:16]


def validate_url(url: str) -> tuple[int, str]:
    """
    Validate whether URL address exists.

    Parameters
    ----------
        url : str
            Link to a website.

    Returns
    -------
        int 
            Http status code or -1 if connection was not established.

        str
            Http reason or exception message if caught.
    """
    try:
        # With 'stream' option enabled we avoid body being immediately downloaded unless explicitly requested
        r = requests.get(url, stream=True, allow_redirects=True, verify=True)

        if "text/html" in r.headers.get("Content-Type", "").lower():
            # Indicate success
            return r.status_code, r.reason
        
        # Playwright's page.goto() i.e. Crawl4AI's navigation expects to load a webpage (HTML)
        return -1, "URL navigation aborted, webpage (HTML) expected"

    except (ConnectionRefusedError, requests.ConnectionError, requests.exceptions.MissingSchema) as e:
        return -1, f"Connection with the URL could not be established: {e}"


def extract_keywords(text, ngram_size: int = 1, dedup_factor: float = 0.2, kw_prop: float = 0.05, kw_num: int = 3) -> list[str]:
    """
    Extract keywords from user's prompt.

    Parameters
    ----------
        text : str
            Text to extract keywords from.
        ngram_size : int
            Number of tokens in a keyword.
        dedup_factor : float
            Deduplication threshold.
        kw_prop : float
            Proportion of keywords to the number of tokens in the prompt.  
        kw_num : int
            Minimum number of keywords to extract.

    Returns
    -------
        list[str]
            List of keywords.
    """
    # Initialize YAKE keyword extractor with default settings
    kw_extractor = KeywordExtractor(
        # Do not specify any language
        lan="",
        # Size of the ngram.
        n=ngram_size,
        # Lower threshold facilitates distinct keywords, while higher values are to keep similar keywords.
        dedupLim=dedup_factor,
        # Max number of extracted keywords
        top=max(kw_num, round(kw_prop * len(text.split())))
    )

    # Extract keywords and their scores. The lower the score, the more relevant the keyword is.
    keywords = [
        keyword for keyword, _
        in kw_extractor.extract_keywords(text)
    ]

    # Return keywords
    return keywords
