# type: ignore

from dataclasses import dataclass
from dotenv import load_dotenv
from os import environ

load_dotenv()

@dataclass
class Config:
    load_dotenv()

    OLLAMA_MODEL = environ.get('OLLAMA_MODEL', 'llama3.1:8b')
    OLLAMA_BASE_URL = environ.get('OLLAMA_BASE_URL', 'http://localhost:11434')
    OLLAMA_CONTEXT_SIZE = int(environ.get('OLLAMA_CONTEXT_SIZE', '8192'))
    OLLAMA_KEEP_ALIVE = environ.get('OLLAMA_KEEP_ALIVE', '5m0s')
    OLLAMA_TEMPERATURE = float(environ.get('OLLAMA_TEMPERATURE', '0.1'))
    OLLAMA_TOP_P = float(environ.get('OLLAMA_TOP_P', '0.05'))
    OLLAMA_TOP_K = int(environ.get('OLLAMA_TOP_K', '5'))
    OLLAMA_EMBEDDINGS_BASE_URL = environ.get('OLLAMA_EMBEDDINGS_BASE_URL', 'http://localhost:11434')
    OLLAMA_EMBEDDINGS_MODEL = environ.get('OLLAMA_EMBEDDINGS_MODEL', 'nomic-embed-text')
    OLLAMA_REPEAT_LAST_N = int(environ.get('OLLAMA_REPEAT_LAST_N', '64'))

    OLLAMA_BASE_URL_POOL = environ.get('OLLAMA_BASE_URL_POOL', 'http://localhost:11434').split(',')

    SKIP_POSTPROCESS = int(environ.get('SKIP_POSTPROCESS', '0')) == 1

    SPLITTER_CHUNK_SIZE = int(environ.get('SPLITTER_CHUNK_SIZE', '2560'))
    SPLITTER_CHUNK_OVERLAP = int(environ.get('SPLITTER_CHUNK_OVERLAP', '256'))

    BERTSCORE_MODEL = environ.get('BERTSCORE_MODEL', 'bert-base-multilingual-cased')
    BERTSCORE_MODEL_N_LAYERS = int(environ.get('BERTSCORE_MODEL_N_LAYERS', '12'))

    N_FACTOR = int(environ.get('N_FACTOR', '2'))
