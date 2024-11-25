from langchain_core.documents import Document
from pprint import pprint
import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from semantic_similarity import get_similarity_score
from prompts.rag_prompt import RAGPrompt
from prompts.prompts import SimplePrompt

from call_llms import call_llms
from config import Config
from llms import get_llama

def test_model_configurations():
    model_configs = {
        'base_url': 'http://192.168.0.254:11434',
        'model': 'llama3.2:3b',
        'temperature': 0.5,
        # 'stream': True,
        'top_k': 100,
        'top_p': 0.95,
        'num_ctx': 256,
        'verbose': False,
        'keep_alive': 1,
        'callbacks': None,
        'repeat_last_n': 1,
    }
    llm = get_llama(model_configuration=model_configs)

    for key, value in model_configs.items():
        assert getattr(llm, key) == value

def test_hosts():
    n = len(Config.OLLAMA_BASE_URL_POOL)
    response = call_llms([
        {
            'model': 'llama3.1:8b',
            'prompt': SimplePrompt('Hello.'),
            'options': {'temperature': 0.0, 'top_k': 2, 'top_p': 0.05},
            'key': i
        } for i in range(n)
    ])

    pprint(response)

    assert len(response) == n

def test_rag_prompt():
    documents = [Document(page_content='In 1991, there was a man named Jerry. Jerry was a race car driver.')]
    response = call_llms([
        {
            'model': 'llama3.1:8b',
            'prompt': RAGPrompt('Based on the following text: {context}. What is the answer to the question: {question}?'),
            'options': {
                'temperature': 0.0,
                'top_k': 2,
                'top_p': 0.05,
                'documents': documents,
                'input_dict': {'question': 'What was Jerry\'s profession?'},
                'verbose': True,
            },
            'key': 0,
        }
    ])

    print(response[0][0])
    score = get_similarity_score(response[0][0], documents[0].page_content, method='bertscore')
    pprint(response)
    print(score)

    assert len(response) == 1

def test_rag_prompt_3():
    response = call_llms([
        {
            'model': 'llama3.1:8b',
            'prompt': RAGPrompt('Based on the following text: {context}. What is the answer to the question: {question}?'),
            'options': {
                'temperature': 0.0,
                'top_k': 2,
                'top_p': 0.05,
                'documents': [Document(page_content='In 1991, there was a man named Jerry. Jerry was a race car driver. He drove so goddamn fast. He never did win no checkered flags. But he never did come in last')], 'input_dict': {'question': 'What was Jerry\'s profession?'}
            },
        } for _ in range(3)
    ])

    pprint(response)

    assert len(response) == 3
