"""Split document text into overlapping chunks for embedding."""

from langchain_text_splitters import RecursiveCharacterTextSplitter

from src import config

_splitter = RecursiveCharacterTextSplitter(
    chunk_size=config.CHUNK_SIZE,
    chunk_overlap=config.CHUNK_OVERLAP,
)


def chunk_text(text: str) -> list[str]:
    """Split text into overlapping chunks using recursive character splitting."""
    chunks = _splitter.split_text(text)
    return [c for c in chunks if c.strip()]
