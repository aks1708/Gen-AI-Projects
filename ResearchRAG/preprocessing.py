from docling.datamodel import vlm_model_specs
from docling.datamodel.base_models import InputFormat
from docling.datamodel.pipeline_options import VlmPipelineOptions
from docling.datamodel.accelerator_options import AcceleratorDevice, AcceleratorOptions
from docling.document_converter import DocumentConverter, PdfFormatOption
from docling.pipeline.vlm_pipeline import VlmPipeline

from langchain_text_splitters import MarkdownHeaderTextSplitter

import chromadb
from chromadb.utils.embedding_functions.ollama_embedding_function import OllamaEmbeddingFunction

from colorama import Fore, Style

pipeline_options = VlmPipelineOptions(
    vlm_options=vlm_model_specs.GRANITEDOCLING_MLX,
    accelerator_options=AcceleratorOptions(
        device=AcceleratorDevice.MPS,
    ),
)

converter = DocumentConverter(
    format_options={
        InputFormat.PDF: PdfFormatOption(
            pipeline_cls=VlmPipeline,
            pipeline_options=pipeline_options,
        ),
    }
)

headers_to_split_on = [
    ("#", "Header 1"),
    ("##", "Header 2"),
    ("###", "Header 3"),
]

ollama_ef = OllamaEmbeddingFunction(model_name="qwen3-embedding:4b", timeout=120)

chroma_client = chromadb.Client()

def convert_to_markdown(source):
    doc = converter.convert(source=source).document
    return doc.export_to_markdown(image_placeholder="")

def split_markdown(markdown):
    markdown_splitter = MarkdownHeaderTextSplitter(headers_to_split_on)
    md_header_splits = markdown_splitter.split_text(markdown)
    refined_splits = [doc for doc in md_header_splits if doc.metadata.get('Header 2') != 'References']
    print(Fore.GREEN + "Markdown chunks created!" + Style.RESET_ALL)
    return refined_splits

def ingest_into_vector_db(source):
    chunks = split_markdown(convert_to_markdown(source))
    if 'collection' in [collection.name for collection in chroma_client.list_collections()]:
        chroma_client.delete_collection(name="collection")

    collection = chroma_client.create_collection(name="collection", embedding_function=ollama_ef)
    doc_ids = [f"id{i+1}" for i in range(len(chunks))]
    documents = [document.page_content for document in chunks]
    metadatas = [{"section": document.metadata["Header 2"] if "Header 2" in document.metadata else ""} for document in chunks]
    
    collection.add(ids=doc_ids,documents=documents,metadatas=metadatas)
    print(Fore.GREEN + "Ingestion completed successfully" + Style.RESET_ALL)
    return collection