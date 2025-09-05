import os
from dotenv import load_dotenv

from src.helper import extract_text_from_pdf, filter_source_and_page_content, \
    get_document_chunks, download_embedding

from pinecone import Pinecone, ServerlessSpec, CloudProvider, AwsRegion
from langchain_pinecone import PineconeVectorStore


load_dotenv()

# API key set
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

os.environ["PINECONE_API_KEY"] = PINECONE_API_KEY
os.environ["GROQ_API_KEY"] = GROQ_API_KEY

if "GROQ_API_KEY" not in os.environ or "PINECONE_API_KEY" not in os.environ:
    print("Please set the GROQ_API_KEY & PINECONE_API_KEY environment variable.")
    exit()


# Data
extracted_data = extract_text_from_pdf(path="data/")
filtered_data = filter_source_and_page_content(extracted_data)
text_chunks = get_document_chunks(filtered_data)

embedding = download_embedding()


# Pinecone 
pc = Pinecone(api_key=PINECONE_API_KEY)
INDEX_NAME = "medbot"

if not pc.has_index(INDEX_NAME):
    pc.create_index(name=INDEX_NAME,
                    dimension=384,
                    metric="cosine",
                    spec=ServerlessSpec(
                        cloud=CloudProvider.AWS, 
                        region=AwsRegion.US_EAST_1 # pinecone free tier only have this region available
                    ) 
    )

index = pc.Index(INDEX_NAME)

# store to pinecone index
docsearch = PineconeVectorStore.from_documents(
    documents=text_chunks,
    embedding=embedding,
    index_name=INDEX_NAME
)