# ResearchRAG

## Overview

The goal of this project is to build a RAG pipeline to answer queries over source information such as PDFs and webpages.

These sources in particular are rich not just in text but also in figures, tables and can have complex layouts.
To handle this, we use Docling to convert the source into markdown format.

We convert this into markdown which later will be used for chunking using LangChain's MarkdownTextSplitter where each section will be their own chunk.

These chunks alongside their embeddings will then be ingested into the ChromaDB vector database.

We leverage query expansion where we generate multiple variations of the original query to try to cover all bases and retrieve the most relevant documents.

The first retrieval step will be a simple vector similarity search over the document chunks. Here we will return around 15 document chunks most relevant to the query.

Then we leverage a cross encoder reranker being Qwen/Qwen3-Reranker-0.6B to further refine the retrieved documents.
This takes in a (query, document) pair and outputs a score indicating how relevant the document is to the query.
We then sort the documents based on this score and return the top n documents. top n is typically 5 in this case.

These refined documents will be used as context and we use the LLM augmented with this context to generate a response to the query.

## Setup

Create your virtual environment and install the dependencies.

```bash
python3.11 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### Download the Reranker model

```bash
hf download Qwen/Qwen3-Reranker-0.6B
```

## Usage

Then run the following command to start the RAG pipeline:

```bash
python3 research_rag.py --source <url_to_source_path or local_path_to_source>
```