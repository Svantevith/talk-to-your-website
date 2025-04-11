from hashlib import sha256
import os
import shutil
import sqlite3
from typing import Literal, Union
from chromadb.api.client import Client
from chromadb.config import Settings
from chromadb.errors import DuplicateIDError
from langchain.docstore.document import Document
from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter


class RAG(Client):
    def __init__(
        self,
        persist_directory: str = ""
    ) -> None:
        """
        RAG objects use Ollama embeddings model to retrieve context for the LLMs from ephemeral memory or persisted Chroma vector store. 

        Parameters
        ----------   
            persist_directory : str
                Directory where to persist the collection. 
        """
        # Set attributes
        self.persist_directory = persist_directory

        # Initially prepare settings for ephemeral client
        client_settings: Settings = Settings(
            is_persistent=False,
            # Do not allow Chroma to collect anonymous usage data
            anonymized_telemetry=False,
            # Do not allow flushing database
            allow_reset=False
        )

        if persist_directory:
            # Use persistent client
            client_settings.is_persistent = True
            client_settings.persist_directory = persist_directory

        # Instantiate client session
        super().__init__(settings=client_settings)

        # Chroma integrates as local vector store
        self.__vector_store: Union[None, Chroma] = None

    def get_context(
        self,
        query: str,
        collection_name: str,
        embedding_model: str = "all-minilm:latest",
        search_function: Literal[
            "mmr",
            "similarity_score_threshold"
        ] = "mmr",
        top_k: int = 3,
        fetch_k: int = 20,
        min_diversity: float = 0.5,
        min_similarity: float = 0.6
    ) -> str:
        """
        Search for context in the vector store.

        Parameters
        ----------
            query : str
                Query to respond.
            collection_name : str
                Collection to search for the context.
            embedding_model : str
                Ollama model to generate embeddings.  
            search_function : Literal["mmr", "similarity_score_threshold"]
                Search function for document retrieval.
            top_k : int
                Amount of documents to return.
            fetch_k : int
                Number of documents filtered by the MMR.
            min_diversity : float
                Minimum diversity of documents returned by the MMR.
            min_similarity : float
                Minimum similarity to exclude unrelated documents. 

        Returns
        -------
            str
                Retrieved context for the LLM.
        """
        # Debugging
        print(
            f"=== Query: {query}, Collection name: {collection_name}, Embedding model: {embedding_model}, Search function: {search_function}, Top K: {top_k}, Fetch K: {fetch_k}, Min Diversity: {min_diversity}, Min Similarity: {min_similarity} ===\n"
        )

        # Create, update or preserve collection
        self.__set_vector_store(collection_name, embedding_model)

        # Get number of documents in the collection
        num_docs = len(self.__vector_store.get()['ids'])

        if num_docs == 0:
            # Debugging
            print(
                f"=== Cannot retrieve context from empty {collection_name} collection, please use add_to_collection() method to populate it ==="
            )

            # Return no context
            return ""

        # Documents retrieval
        search_params = {
            "k": top_k
        }

        if search_function == "mmr":
            # Use MMR to control trade-off between relevancy and diversity
            search_params.update({
                "fetch_k": fetch_k,
                "lambda_mult": min_diversity
            })

        else:
            # Enhance simiarity search with score threshold to exclude irrelevant documents
            search_params.update({
                "score_threshold": min_similarity
            })

        # Retrieve the top-k documents based on the search configuration.
        retriever = self.__vector_store.as_retriever(
            search_type=search_function,
            search_kwargs=search_params
        )

        # Debugging
        print(
            f"=== Number of documents in the {collection_name} collection: {num_docs} ==="
        )

        # Retrieve relevant documents
        retrieved_documents = retriever.invoke(query)

        # Debugging
        print(
            f"=== Number of documents retrieved: {len(retrieved_documents)} ==="
        )

        # Join retrieved content into a single string.
        context = ' '.join(
            [doc.page_content for doc in retrieved_documents]
        )

        # Return context to the LLM.
        return context

    async def add_to_collection(
        self,
        document: Document,
        collection_name: str,
        embedding_model: str = "all-minilm:latest",
        window_size: int = 256,
        window_overlap: float = 0.2
    ) -> None:
        """
        Add document to the vector store. 
        Asynchronous execution does not block the main thread, hence smaller documents can be processed independently of larger ones still being processed.

        Parameters
        ----------
            document : Document
                Langchain's document.
            collection_name : str
                Collection to populate with the document.
            embedding_model : str
                Ollama model to generate embeddings.  
            window_size : int
                Window size for text segmentation.
            window_overlap : float
                Proportion of the overlap to the window size.
        """
        # Debugging
        print(
            f"=== Collection name: {collection_name}, Embedding model: {embedding_model}, Window size: {window_size}, Window overlap: {window_overlap} ===\n"
        )

        # Create, update or preserve collection
        self.__set_vector_store(collection_name, embedding_model)

        # RAGs favor segmented documents as input.
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=window_size,
            chunk_overlap=round(window_size * window_overlap),
            separators=[
                # Split by paragraphs and line breaks
                "\n\n", "\n",
                # Split at sentence boundaries (., ?, !)
                ".", ",", "?", "!",
                # Zero-width space
                "\u200b",
                # Fullwidth comma
                "\uff0c",
                # Ideographic comma
                "\u3001",
                # Fullwidth full stop
                "\uff0e",
                # Ideographic full stop
                "\u3002",
                # Ensure that if no larger splits work, words and characters get split last
                " ", "",
            ],
        )

        # Sliding window generates overlapping chunks for better contextual coherence
        for doc in text_splitter.split_documents([document]):
            try:
                # Asynchronously add text segments to the collection
                await self.__vector_store.aadd_documents(
                    # Function is expecting list of documents as input
                    documents=[doc,],
                    # Hash of page content serves as unique id to avoid duplicate entries
                    ids=[sha256(doc.page_content.encode()).hexdigest(),]
                )

            except DuplicateIDError as e:
                # Duplicate documents are skipped, but exception is raised
                print(
                    f"=== Adding document to {collection_name} collection failed. {e} ===")

    def delete_collection_artifacts(self, collection_name: str) -> None:
        """
        Delete collection and associated artifacts from the Chroma database.

        Parameters
        ----------
            collection_name: str
              Collection name to delete.
        """
        try:
            # Delete collection
            self.delete_collection(collection_name)
            print(
                f"=== Successfully deleted the {collection_name} collection ==="
            )
        
        except ValueError:
            print(
                f"=== Collection {collection_name} has not been found in the {self.persist_directory} database ==="
            )

        finally:
            # Delete unused directories
            self.__delete_unused_directories()

    def describe_collection(self, docs_limit: Union[None, int] = None) -> None:
        """
        Print details about the collection's content.

        Parameters
        ----------
            docs_limit : Union[None, int]
                Number of documents to consider.
        """
        # Get collections content
        collection = self.__collection.get(
            limit=docs_limit
        )

        print(
            f"\n=== Displaying {len(collection["ids"])} documents from the {self.collection_name} collection ===\n"
        )

        print(
            *[
                f"ID: {doc_id}\nContent: {doc_content}\nMetadata: {doc_meta}"
                for doc_id, doc_content, doc_meta in zip(collection["ids"], collection["documents"], collection["metadatas"])
            ],
            sep="\n\n",
            end="\n\n"
        )

    def __set_vector_store(
        self,
        collection_name: str,
        embedding_model: str = "all-minilm:latest",
    ) -> None:
        """
        Create, update or preserve vector store for populating collection and retrieving context.

        Parameters
        ----------
            collection_name : str
                Collection to populate with the document.
            embedding_model : str
                Ollama model to generate embeddings.  
        """
        # Get vector store only when collection changes to prevent numerous loads from database when retrieving context
        if self.__vector_store is None or self.__vector_store._collection.name != collection_name:

            # Clear system cache
            self.clear_system_cache()

            # Set vector store only when collection changes to prevent numerous loads from database while retrieving context
            self.__vector_store = Chroma(
                client=self,
                collection_name=collection_name,
                embedding_function=OllamaEmbeddings(model=embedding_model),
                # Prevent negative scores
                collection_metadata={"hnsw:space": "cosine"}
            )

    def __delete_unused_directories(self) -> None:
        """
        Remove remaining directories after a collection is deleted, but some artifacts still exist in the database.
        """
        # Browse persist directory
        chroma_collections = os.listdir(self.persist_directory)

        # Check if persist directory has a running Chroma DB instance
        if "chroma.sqlite3" not in chroma_collections:
            print(
                f"=== No Chroma database running in {self.persist_directory} ==="
            )
            return

        # Remove database instance from paths
        chroma_collections.remove("chroma.sqlite3")

        # Check if there are any artifacts in the persist directory
        if not chroma_collections:
            # Debugging
            print(
                f"=== No Chroma artifacts in {self.persist_directory} to remove ==="
            )
            return

        # Retrieve vector store ids used by Chroma DB to identify a collection
        vector_store_ids = self.__get_vector_store_ids()

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
                collection_path = os.path.join(self.persist_directory, collection)
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

    def __get_vector_store_ids(self) -> set[str]:
        """
        Retrieve vector store IDs from the persist directory.

        Returns
        -------
            set[str]
                Set of unique vector store IDs.
        """
        # Connect to the database
        db = sqlite3.connect(
            os.path.join(self.persist_directory, "chroma.sqlite3")
        )

        # Get locations where Chroma stores specific collections
        cursor = db.cursor()
        cursor.execute("SELECT id FROM segments WHERE scope = 'VECTOR'")

        # Return retrieved locations
        return {
            id[0] for id in cursor.fetchall()
        }