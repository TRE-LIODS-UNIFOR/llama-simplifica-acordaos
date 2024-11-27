# type: ignore

from dataclasses import dataclass
from dotenv import load_dotenv
from os import environ

load_dotenv()

@dataclass
class Config:
    load_dotenv()

    OLLAMA_MODEL = environ.get('OLLAMA_MODEL')
    OLLAMA_BASE_URL = environ.get('OLLAMA_BASE_URL')
    OLLAMA_CONTEXT_SIZE = environ.get('OLLAMA_CONTEXT_SIZE')
    OLLAMA_KEEP_ALIVE = environ.get('OLLAMA_KEEP_ALIVE')
    OLLAMA_TEMPERATURE = environ.get('OLLAMA_TEMPERATURE')
    OLLAMA_TOP_P = environ.get('OLLAMA_TOP_P')
    OLLAMA_TOP_K = environ.get('OLLAMA_TOP_K')
    OLLAMA_EMBEDDINGS_BASE_URL = environ.get('OLLAMA_EMBEDDINGS_BASE_URL')
    OLLAMA_EMBEDDINGS_MODEL = environ.get('OLLAMA_EMBEDDINGS_MODEL', 'nomic-embed-text')
    OLLAMA_REPEAT_LAST_N = environ.get('OLLAMA_REPEAT_LAST_N')

    OLLAMA_BASE_URL_POOL = environ.get('OLLAMA_BASE_URL_POOL').split(',')

    OLLAMA_MODEL_2 = environ.get('OLLAMA_MODEL_2')
    OLLAMA_BASE_URL_2 = environ.get('OLLAMA_BASE_URL_2')
    OLLAMA_CONTEXT_SIZE_2 = environ.get('OLLAMA_CONTEXT_SIZE_2')
    OLLAMA_KEEP_ALIVE_2 = environ.get('OLLAMA_KEEP_ALIVE_2')
    OLLAMA_TEMPERATURE_2 = environ.get('OLLAMA_TEMPERATURE_2')
    OLLAMA_TOP_P_2 = environ.get('OLLAMA_TOP_P_2')
    OLLAMA_TOP_K_2 = environ.get('OLLAMA_TOP_K_2')
    OLLAMA_EMBEDDINGS_BASE_URL_2 = environ.get('OLLAMA_EMBEDDINGS_BASE_URL_2')
    OLLAMA_EMBEDDINGS_MODEL_2 = environ.get('OLLAMA_EMBEDDINGS_MODEL_2')

    SPLITTER_CHUNK_SIZE = int(environ.get('SPLITTER_CHUNK_SIZE'))
    SPLITTER_CHUNK_OVERLAP = int(environ.get('SPLITTER_CHUNK_OVERLAP'))

    BERTSCORE_MODEL = environ.get('BERTSCORE_MODEL', 'bert-base-multilingual-cased')
    BERTSCORE_MODEL_N_LAYERS = int(environ.get('BERTSCORE_MODEL_N_LAYERS'))
