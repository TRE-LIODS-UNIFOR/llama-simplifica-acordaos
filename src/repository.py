from uuid import uuid4
from langchain_ollama import ChatOllama
from config import Config
from database import PromptDB
from models import Prompt


class Repository:
    db: PromptDB

    def __init__(self):
        self.db = PromptDB()

    def save_prompt(self, chat_ollama: ChatOllama, prompt: str):
        self.db.insert_prompt(
            Prompt(
                id=str(uuid4()),
                prompt=prompt,
                model=chat_ollama.model,
                chunk_overlap=Config.chunk_overlap,
                chunk_size=Config.chunk_size,
                context_size=Config.context_size,
                created_at=None,
                embeddings_model=Config.OLLAMA_EMBEDDINGS_MODEL,
                temperature=chat_ollama.temperature,
                top_k=chat_ollama.top_k,
                top_p=chat_ollama.top_p,
            )
        )
