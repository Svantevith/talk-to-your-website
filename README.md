# Talk to your website

This project delivers Retrieval-Augmented Generation (RAG) agent enhanced with LLM-optimized website crawler built using Crawl4AI, Langchain, ChromaDB and Ollama. The agent can crawl websites, store extracted content in a local vector database, and provide context-aware answers to user queries by retrieving and analyzing the most relevant pieces of textual information.

At its core lays a user-friendly, interactive chat interface built with Streamlit, ensuring a seamless and responsive user experience. The chat application allows users to configure behaviour of the assistant as needed to receive contextually coherent responses.

By leveraging ChromaDB and Ollama on a local machine, this system offers several key advantages:
- Privacy & Security – No external API calls, ensuring data remains private.
- Cost-Effective – No reliance on paid APIs or cloud subscriptions.
- Offline Functionality – Works without an internet connection retrieving context from locally persisted collection.
- Multilangage Support – Various models are trained on a broad collection of languages, however it must be taken into account that the leading language is English.

This makes the application an excellent choice for research, knowledge management, and AI-powered text analysis without compromising control over your data. 

## Features

- Deep crawling solution from Crawl4AI precisely extracts AI-optimized content from beyond a single webpage. 
- Splitting documents into chunks with Langchain keeps context intact, while ensuring token importance.
- Chroma database client persists vector stores on disk as locally available knowledgebase.
- Build efficient vectors stores with Ollama embedding models trained on very large sentence-level datasets.
- Question answering combines retrieval of semantically relevant documents with the generative power of local Ollama LLM models.
- Offline-suported Streamlit application with responsive UI to control agent's behaviour to ensure seamless chat experience.

## Prerequisites

- Python 3.8+ `https://www.python.org/downloads/`
- Ollama client, a local runtime for LLMs `https://ollama.com/download`
- Referenced Ollama models `https://ollama.com/search` have to be pulled, for example `ollama pull llama3.2:1b`

## Installation

1. Clone the repository:
```bash
git clone https://github.com/Svantevith/talk-to-your-website.git
cd talk-to-your-website
```

2. Install dependencies (recommended to use a Python virtual environment):
```bash
pip install -r requirements.txt
```

## Usage

Launch the chat application with the following command in the console:

```bash
streamlit run main.py
```

The chat application will be available at `http://localhost:8501`

## Configuration

Modules and scripts share the `src/config.py` configuration file, where developers can control constant attributes in the static configuration classes. 

## Project Structure

Modules:
- `src/app.py`: Interactive chat interaface (UI)
- `src/config.py`: Configuration file for managing constants
- `src/crawlers.py`: Website crawling and page content processing
- `src/retrievers.py`: Context retrieval with RAG implementation
- `src/models.py`: Multilingual dialogue agent with LLM implementation
- `utils/helper_functions.py`: Helper functions used across the modules

Data:
- `data/chroma_db` (Default unless `RAGConfig.PERSISTENT_DIRECTORY` from `src/config.py` is modified): Storage location for vector database files

Scripts:
- `main.py` Main file to launch the chat application

Other files:
- `requirements.txt`: Project dependencies