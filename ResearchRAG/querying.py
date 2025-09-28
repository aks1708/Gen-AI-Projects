from pydantic import BaseModel
from typing import List

from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama import ChatOllama

import reranking

class QueryExpansion(BaseModel):
    variations: List[str]

query_expansion_prompt = """
Please provide 3 additional search keywords or phrases for each of the key aspects of the following query that makes it easier to find relevant documents:
Query: {query}
"""

rag_prompt = ChatPromptTemplate.from_messages(
    [
       ("system",
        """You are provided with the following context.
        Context: {context}
        
        Use the context to give a comprehensive answer to the user's query.
        Only if the given context does not answer the question then say 'I don't know'.
        Do not include phrases such as `according to the context` or `based on the context`. Just give the answer."""),
        ("human", "{query}")

    ])

query_expander_llm = ChatOllama(model="llama3.1:8b", temperature=0).bind_tools([QueryExpansion])

gemma = ChatOllama(model="gemma3:1b", temperature=0)

rag_chain = rag_prompt | gemma

def query_expander(query):
    queries = query_expander_llm.invoke(query_expansion_prompt.format(query=query))
    augmented_queries = [query] + queries.tool_calls[0]['args']['variations']
    return augmented_queries

def get_relevant_docs(augmented_queries, collection, k=10):
    return collection.query(query_texts=augmented_queries, n_results=k)['documents'][0]

def generate_response(query, collection):
    augmented_queries = query_expander(query)
    draft_documents = get_relevant_docs(augmented_queries, collection)

    refined_context = reranking.reranked_context(query, draft_documents)
    
    response = rag_chain.invoke({"query": query, "context": refined_context})
    return response.content