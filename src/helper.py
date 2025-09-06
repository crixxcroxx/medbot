import os
from dotenv import load_dotenv

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


def load_env_vars() -> tuple[str, str]:
    load_dotenv()

    PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
    GROQ_API_KEY = os.getenv("GROQ_API_KEY")

    os.environ["PINECONE_API_KEY"] = PINECONE_API_KEY
    os.environ["GROQ_API_KEY"] = GROQ_API_KEY

    if "GROQ_API_KEY" not in os.environ or "PINECONE_API_KEY" not in os.environ:
        print("Please set the GROQ_API_KEY & PINECONE_API_KEY environment variable.")
        exit()

    return PINECONE_API_KEY, GROQ_API_KEY