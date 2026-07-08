from langchain_experimental.text_splitter import SemanticChunker
from langchain_text_splitters import RecursiveCharacterTextSplitter
from app.embeddings.embedding_model import embeddings

Semantic_splitter = SemanticChunker(
    embeddings,
    breakpoint_threshold_type= "gradient"
)

recursive_splitter = RecursiveCharacterTextSplitter(
    chunk_size = 1000,
    chunk_overlap = 200,
    separators = ["\n\n", "\n", " ", ""]
)