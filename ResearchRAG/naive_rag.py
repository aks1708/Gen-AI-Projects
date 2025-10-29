from typing import List

import chromadb
from chromadb.utils.embedding_functions.ollama_embedding_function import OllamaEmbeddingFunction
from colorama import Fore, Style
from docling.document_converter import DocumentConverter
import json
from langchain_text_splitters import MarkdownHeaderTextSplitter
from litellm import completion
from pydantic import BaseModel

import reranking
from prompts import QUERY_EXPANSION_PROMPT, RAG_SYSTEM_PROMPT

import logging
logging.getLogger("LiteLLM").setLevel(logging.ERROR)

class QueryExpansion(BaseModel):
    variations: List[str]

class NaiveRAG:
    
    headers_to_split_on: list = [
        ("#", "Header 1"),
        ("##", "Header 2"),
        ("###", "Header 3")
    ]

    chroma_client = chromadb.Client()

    def __init__(
        self, 
        source: str, 
        llm: str,
        embedding_model: str,
        query_expander_llm: str = "ollama/llama3.1:8b"
        ):
        
        self.source = source
        self.llm = llm
        self.chroma_embedding_fn = OllamaEmbeddingFunction(model_name=embedding_model)
        self.query_expander_llm = query_expander_llm
        self.collection = self._ingest_into_vector_db()
    
    def _convert_to_markdown(self):
        if self.source.endswith(".md"):
            with open(self.source, 'r') as f:
                return f.read()
       
        doc = DocumentConverter().convert(source=self.source).document
        print(doc.export_to_markdown(image_placeholder=""))
        return doc.export_to_markdown(image_placeholder="")
    
    def _ingest_into_vector_db(self):
        markdown = self._convert_to_markdown()
        markdown_splitter = MarkdownHeaderTextSplitter(NaiveRAG.headers_to_split_on)
        md_header_splits = markdown_splitter.split_text(markdown)
        refined_splits = [doc for doc in md_header_splits if doc.metadata.get('Header 2') != 'References']
        print(Fore.GREEN + "Markdown chunks created!" + Style.RESET_ALL)
        
        if 'collection' in [collection.name for collection in NaiveRAG.chroma_client.list_collections()]:
            NaiveRAG.chroma_client.delete_collection(name="collection")

        collection = NaiveRAG.chroma_client.create_collection(name="collection", embedding_function=self.chroma_embedding_fn)
        doc_ids = [f"id{i+1}" for i in range(len(refined_splits))]
        documents = [document.page_content for document in refined_splits]
        metadatas = [{"section": document.metadata["Header 2"] if "Header 2" in document.metadata else ""} for document in refined_splits]
    
        collection.add(ids=doc_ids,documents=documents,metadatas=metadatas)
        print(Fore.GREEN + "Ingestion completed successfully" + Style.RESET_ALL)
        
        return collection
    
    def _expand_query(self, query, n=3):
        
        queries = completion(
            model=self.query_expander_llm, 
            messages=[{"role": "user", "content": QUERY_EXPANSION_PROMPT.format(n=3, query=query)}],
            response_format=QueryExpansion)
        
        query_list = json.loads(queries.choices[0].message.content)['variations']
        
        augmented_queries = [query] + query_list
        
        return augmented_queries
    
    def query(self, query, k=15):
        
        augmented_queries = self._expand_query(query)
        draft_docs = self.collection.query(query_texts=augmented_queries, n_results=k)['documents'][0]

        refined_context = reranking.reranked_context(query, draft_docs)

        messages = [
            {"role": "system", "content": RAG_SYSTEM_PROMPT.format(context=refined_context)},
            {"role": "user", "content": query}
        ]

        response = completion(
            model=self.llm, 
            messages=messages)
        
        return response.choices[0].message.content