from preprocessing import ingest_into_vector_db
from querying import generate_response

import argparse
from colorama import Fore, Style

##### Parse Command Line Arguments #####
parser = argparse.ArgumentParser(description='Process a document and answer questions about it.')
parser.add_argument('-s', '--source', type=str, required=True, 
                    help='Path to the source document (URL or local file path)')
args = parser.parse_args()

##### Logic Starts Here #####
source_document = args.source

collection = ingest_into_vector_db(source_document)

while True:
    query = input(Fore.BLUE + "Yo, what's up? (Type 'exit' to quit): " + Style.RESET_ALL + Fore.YELLOW)
    print(Style.RESET_ALL, end='')
    if query == "exit":
        break
    print(Fore.RED + "\nThinking..." + Style.RESET_ALL)
    response = generate_response(query, collection)
    print("\n")
    print(f"{Fore.MAGENTA} {response}{Style.RESET_ALL}")