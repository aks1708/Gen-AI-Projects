from pydantic import BaseModel
from typing import List

from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate

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
        
        ONLY use the context to give a comprehensive answer to the user's query.
        Do not use your own knowledge to answer the question.
        Only if the given context does not answer the question then say 'I don't know'.
        Do not include phrases such as `according to the context` or `based on the context`. Just give the answer."""),
        ("human", "{query}")

    ])

llm = ChatOllama(model="llama3.1:8b", temperature=0)
query_expander = llm.bind_tools([QueryExpansion])

rag_chain = rag_prompt | llm

def expand_query(query):
    queries = query_expander.invoke(query_expansion_prompt.format(query=query))
    augmented_queries = [query] + queries.tool_calls[0]['args']['variations']
    return augmented_queries

def get_relevant_docs(augmented_queries, collection, k=15):
    return collection.query(query_texts=augmented_queries, n_results=k)['documents'][0]

def generate_response(query, collection):
    augmented_queries = expand_query(query)
    draft_documents = get_relevant_docs(augmented_queries, collection)

    refined_context = reranking.reranked_context(query, draft_documents)
    
    response = rag_chain.invoke({"query": query, "context": refined_context})
    return response.content