from langchain_experimental.text_splitter import SemanticChunker
from app.embeddings.embedding_model import embeddings

splitter = SemanticChunker(
    embeddings,
    breakpoint_threshold_type= "gradient"
)