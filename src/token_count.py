from langchain_ollama import ChatOllama

from config import Config


def get_token_count(text: str) -> int:
    llm: ChatOllama = ChatOllama(
        model=Config.OLLAMA_MODEL
    )
    return llm.get_num_tokens(text)
