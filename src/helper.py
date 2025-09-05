from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.schema import Document

from src import DEVICE


def extract_text_from_pdf(path: str) -> list[Document]:
    loader = DirectoryLoader(
        path=path,
        glob="*.pdf",
        loader_cls=PyPDFLoader 
    )

    document = loader.load()

    return document


def filter_source_and_page_content(docs: list[Document]) -> list[Document]:
    filtered_docs: list[Document] = []

    for doc in docs:
        src = doc.metadata.get("source")
        filtered_docs.append(
            Document(
                metadata={"source": src},
                page_content=doc.page_content
            )
        )
    
    return filtered_docs


def get_document_chunks(doc: list[Document]) -> list[Document]:
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=512,
        chunk_overlap=64,
        length_function=len
    )

    text_chunks = text_splitter.split_documents(doc)
    return text_chunks


def download_embedding() -> HuggingFaceEmbeddings:
    model_name = "sentence-transformers/all-MiniLM-L6-v2"
    embeddings = HuggingFaceEmbeddings(
        model_name=model_name,
        model_kwargs={"device": DEVICE}
    )

    return embeddings