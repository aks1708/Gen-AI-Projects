# ResearchRAG

## Overview

The goal of this project is to build a RAG pipeline to answer queries over a document.
Specifically we are taking research papers in the form of PDFs.

These documents are rich not just in text but also in figures, tables and can have complex layouts.
Thus to process these documents, we use IBM's Granite-Docling-258M model to convert the PDF into markdown format.
This is an ultra lightweight open source VLM (Vision Language Model) that can preserve the structure of the document.

We convert this into markdown which later will be used for chunking using LangChain's MarkdownTextSplitter where each section will be their own chunk.

These chunks alongside their embeddings will then be ingested into the ChromaDB vector database.

We leverage query expansion where we generate multiple variations of the original query in order to retrieve the most relevant documents.

The first retrieval step will be a simple vector similarity search over the document chunks. Here we will return around 15 document chunks most relevant to the query.

Then we leverage a cross encoder reranker being Qwen/Qwen3-Reranker-0.6B to further refine the retrieved documents.
This takes in a (query, document) pair and outputs a score indicating how relevant the document is to the query.
We then sort the documents based on this score and return the top n documents. top n is typically 5 in this case.

These refined documents will be used as context and we use Llama 3.1 8B to generate a response to the query.

This project does not use proprietary APIs. We download the models from Hugging Face and Ollama and use them locally.

Embedding model: qwen3-embedding:4b
Reranker model: Qwen/Qwen3-Reranker-0.6B
VLM model: Granite-Docling-258M
LLM model: Llama 3.1 8B

## Setup

Create your virtual environment and install the dependencies.

```bash
python3.11 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Make sure you have Ollama installed and running.
To download Llama 3.1 8B and qwen3-embedding:4b, run the following commands:

```bash
ollama pull llama3.1:8b
ollama pull qwen3-embedding:4b
```

### Download the Reranker and VLM Models

1. First, download the Qwen3-Reranker model:

```bash
hf download Qwen/Qwen3-Reranker-0.6B
```

2. Then, download the appropriate Granite-Docling model based on your operating system:

**For macOS (with MPS support):**
```bash
hf download ibm-granite/granite-docling-258M-mlx
```

**For Linux/Windows/Other OS:**
```bash
hf download ibm-granite/granite-docling-258M
```

## Usage

Then run the following command to start the RAG pipeline:

```bash
python3 research_rag.py --source <url_to_pdf_path or local_pdf_path>
```