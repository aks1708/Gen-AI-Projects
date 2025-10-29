QUERY_EXPANSION_PROMPT = """
You have to expand a given query into {n} queries that are semantically similar to improve retrieval recall.

Examples:
1.  Query: "climate change effects"
    ["impact of climate change", "consequences of global warming", "effects of environmental changes"]

2.  Query: "machine learning algorithms"
    ["neural networks", "clustering techniques", "supervised learning methods", "deep learning models"]

3.  Query: "open source NLP frameworks"
    ["natural language processing tools", "free nlp libraries", "open-source NLP platforms"]

Guidelines:
- Generate queries that use different words and phrasings
- Include synonyms and related terms
- Maintain the same core meaning and intent
- Make queries that are likely to retrieve relevant information the original might miss
- Focus on variations that would work well with keyword-based search
- Respond in the same language as the input query

Query: {query}
"""

RAG_SYSTEM_PROMPT = """
You are provided with the following context.
Context: {context}
        
ONLY use the context to give a comprehensive answer to the user's query.
Do not use your own knowledge to answer the question.
Only if the given context does not answer the question then say 'I don't know'.
Do not include phrases such as `according to the context` or `based on the context`. Just give the answer."""