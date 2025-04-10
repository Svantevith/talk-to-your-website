from hashlib import sha256
from typing import Literal, Union
from chromadb import EphemeralClient, PersistentClient
from chromadb.config import Settings
from chromadb.errors import DuplicateIDError
from langchain.docstore.document import Document
from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter


class RAG():
    def __init__(
        self,
        embedding_model: str = "all-minilm:latest",
        collection_name: str = "",
        persist_directory: str = ""
    ) -> None:
        """
        RAG objects use Ollama embeddings model to retrieve context for the LLMs based on the collected knowledgebase. 

        Parameters
        ---------- 
            embedding_model : str
                Ollama model to generate embeddings.    
            collection_name : str
                Name for the collection.
            persist_directory : str
                Directory where to persist the collection. 
        """
        # Collection configuration
        self.embedding_model = embedding_model
        self.collection_name = collection_name[:63]
        self.persist_directory = persist_directory

        # Chroma client instance
        self.__client = self.__get_chroma_client()

        # Clear system cache
        self.__client.clear_system_cache()

        # Chroma integrates as local vector store
        self.__collection = Chroma(
            collection_name=self.collection_name,
            embedding_function=OllamaEmbeddings(model=self.embedding_model),
            client=self.__client,
            # Prevent negative scores
            collection_metadata={"hnsw:space": "cosine"}
        )

    def __get_chroma_client(self) -> Union[PersistentClient, EphemeralClient]:
        """
        Return appropriate Chroma client configuration. 

        Returns
        -------
            Union[PersistentClient, EphemeralClient]
                Chroma Client instance.
        """
        # Configure client settings
        client_settings = Settings(
            # Do not allow Chroma to collect anonymous usage data
            anonymized_telemetry=False,
            allow_reset=True
        )

        if self.persist_directory:
            # Return persistent instance
            return PersistentClient(
                path=self.persist_directory,
                settings=client_settings
            )

        # Return in-memory instance
        return EphemeralClient(
            settings=client_settings
        )

    async def add_to_collection(
        self,
        document: Document,
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
            window_size : int
                Window size for text segmentation.
            window_overlap : float
                Proportion of the overlap to the window size.
        """
        # Debugging
        print(
            f"=== Window size: {window_size}, Window overlap: {window_overlap} ===\n"
        )

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
                await self.__collection.aadd_documents(
                    # Function is expecting list of documents as input
                    documents=[doc,],
                    # Hash of page content serves as unique id to avoid duplicate entries
                    ids=[sha256(doc.page_content.encode()).hexdigest(),]
                )
                
            except DuplicateIDError as e:
                # Duplicate documents are skipped, but exception is raised
                print(f"=== Adding document to {self.collection_name} collection failed. {e} ===")

    def get_context(
        self,
        query: str,
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
            f"=== Query: {query}, Search function: {search_function}, Top K: {top_k}, Fetch K: {fetch_k}, Min Diversity: {min_diversity}, Min Similarity: {min_similarity} ===\n"
        )

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
        retriever = self.__collection.as_retriever(
            search_type=search_function,
            search_kwargs=search_params
        )

        # Debugging
        print(
            f"=== Number of documents in the {self.collection_name} collection: {len(self.__collection.get()['ids'])} ==="
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
