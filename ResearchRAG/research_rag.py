import argparse
from colorama import Fore, Style

from dotenv import load_dotenv
load_dotenv()

import os

from naive_rag import NaiveRAG

##### Parse Command Line Arguments #####
parser = argparse.ArgumentParser(description='Process a document and answer questions about it.')
parser.add_argument('-s', '--source', type=str, required=True, 
                    help='Path to the source document (URL or local file path)')
args = parser.parse_args()

##### Logic Starts Here #####
source_document = args.source

naive_rag = NaiveRAG(
    source_document,
    llm=os.getenv("LLM_MODEL"),
    embedding_model=os.getenv("EMBEDDING_MODEL"),
    query_expander_llm=os.getenv("QUERY_EXPANDER_MODEL"))

while True:
    query = input(Fore.BLUE + "Yo, what's up? (Type 'exit' to quit): " + Style.RESET_ALL + Fore.YELLOW)
    print(Style.RESET_ALL, end='')
    if query == "exit":
        break
    print(Fore.RED + "\nThinking..." + Style.RESET_ALL)
    response = naive_rag.query(query)
    print("\n")
    print(f"{Fore.MAGENTA} {response}{Style.RESET_ALL}")