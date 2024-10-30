from dataclasses import dataclass
from datetime import datetime


@dataclass
class Prompt:
    id: str
    prompt: str
    model: str
    temperature: float
    top_p: float
    top_k: float
    context_size: int
    embeddings_model: str
    chunk_size: int
    chunk_overlap: int
    created_at: datetime

@dataclass
class Response:
    id: str
    prompt_id: str
    response: str
    quality: int
    document_id: str
    created_at: datetime

@dataclass
class Document:
    id: str
#    kind: str

@dataclass
class DocumentKind:
    id: int
    kind: str

@dataclass
class Result:
    id: str
    prompt_id: str
    response_id: str
    created_at: datetime

