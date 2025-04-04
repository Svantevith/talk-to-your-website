import os
import time
import socket
import requests
import shutil
import sqlite3
from datetime import datetime
from collections.abc import Generator
from yake import KeywordExtractor


def delete_collection_directories(persist_directory: str) -> None:
    """
    Remove remaining directories after a collection was deleted, but some artifacts still exist in the database.

    Parameters
    ----------
        persist_directory : str
            Path to the persist directory.
    """
    # Browse persist directory
    chroma_collections = os.listdir(persist_directory)

    # Check if persist directory has a running Chroma DB instance
    if "chroma.sqlite3" not in chroma_collections:
        print(
            f"=== No Chroma database running in {persist_directory} ==="
        )
        return

    # Remove database instance from paths
    chroma_collections.remove("chroma.sqlite3")

    # Check if there are any artifacts in the persist directory
    if not chroma_collections:
        # Debugging
        print(
            f"=== No Chroma artifacts in {persist_directory} to remove ==="
        )
        return

    # Retrieve vector store ids used by Chroma DB to identify a collection
    vector_store_ids = get_vector_store_ids(persist_directory)

    # Iterate over collection directories
    for collection in chroma_collections:

        # Collection was not deleted and is still in the vector scope
        if collection in vector_store_ids:
            # Debugging
            print(
                f"=== Collection {collection} was not deleted by the client and is still in the vector scope ==="
            )
            continue

        try:
            # Remove directory associated with an unused collection
            collection_path = os.path.join(persist_directory, collection)
            shutil.rmtree(collection_path)

            # Debugging
            print(
                f"=== Remaining directory {collection_path} was removed ==="
            )

        except PermissionError:
            # Debugging
            print(
                f"=== Chroma DB hasn't released the {collection} collection yet ==="
            )


def get_vector_store_ids(persist_directory: str) -> set[str]:
    """
    Retrieve vector store IDs in the specified persist directory.

    Parameters
    ----------
        persist_directory : str
            Path to the persistent directory.

    Returns
    -------
        set[str]
            Set of vector store IDs.
    """
    # Connect to the database
    db = sqlite3.connect(os.path.join(persist_directory, "chroma.sqlite3"))

    # Get locations where Chroma stores specific collections
    cursor = db.cursor()
    cursor.execute("SELECT id FROM segments WHERE scope = 'VECTOR'")

    # Return retrieved locations
    return {
        id[0] for id in cursor.fetchall()
    }


def connected_to_internet(host: str = "8.8.8.8", port: int = 53, timeout: float = 3.0) -> bool:
    """
    Check if connection to internet can be established.

    Parameters
    ----------
        host: str 
            Host to connect with, by default 8.8.8.8 (google-public-dns-a.google.com)
        port : int 
            Port to use, by default 53 (DNS uses TCP and UDP on port 53, so connections are simple and fast)
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
        print(f"=== Connection with the host {host} over port {port} could not be established: {e}")
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


def modified_metadata(
    old_metadata: dict[str, any],
    new_metadata: dict[str, any],
    excluded_prefix: str = "",
    excluded_keys: set[str] = {}
) -> set[str]:
    """
    Retrieve keys of modified metadata.

    Parameters
    ----------
        old_metadata : dict[str, any]
            Single-level dictionary with previous values.
        old_metadata : dict[str, Any]
            Single-level dictionary with current values.
        excluded_prefix : str[str]
            Prefix to exclude from comparison.
        excluded_keys : set[str]
            Set of keys to exclude from comparison.

    Returns
    -------
        set[str] 
            Set of keys corresponding to the modified metadata.
    """

    return {
        key for key, new_value in new_metadata.items()
        if (
            key not in excluded_keys and
            (excluded_prefix == "" or not key.startswith(excluded_prefix)) and
            old_metadata.get(key, "") != new_value
        )
    }


def stream_response(text: str, seconds: float = 0.05) -> Generator[str]:
    """
    Yield one word at a time to the streamlit's write_stream function.

    Parameters
    ----------
        text : str
            Message to stream.
        seconds : float
            Seconds to wait between each iteration.

    Returns
    -------
        Generator[str]
            Word generator.
    """
    for word in text.split(" "):
        yield word + " "
        time.sleep(seconds)


def validate_url(url: str) -> str:
    """
    Validate URL address.

    Parameters
    ----------
        url : str
            Link to a website.

    Returns
    -------
        str
            Exception message if errors occurred.
    """
    try:
        requests.get(url)
        return ""
    except (requests.ConnectionError, requests.exceptions.MissingSchema) as e:
        return e


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
