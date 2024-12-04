import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from prompts.prompts import SimplePrompt
from call_llms import call_llms
from config import Config

def test_call_llms():
    n = len(Config.OLLAMA_BASE_URL_POOL)
    results = call_llms([
        {
            'model': 'llama3.1:8b',
            'prompt': SimplePrompt('The quick brown fox jumps over the lazy dog.'),
            'options': {
                'temperature': 0.5,
                'max_tokens': 100
            },
        } for _ in range(n)
    ],
        n_workers=n
    )

    assert len(results) == n

def test_call_llms_with_keys():
    results = call_llms([
        {
            'model': 'llama3.1:8b',
            'prompt': 'The quick brown fox jumps over the lazy dog.',
            'options': {
                'temperature': 0.5,
                'max_tokens': 100
            },
            'key': 0
        },
        {
            'model': 'llama3.1:8b',
            'prompt': 'The quick brown fox jumps over the lazy dog.',
            'options': {
                'temperature': 0.5,
                'max_tokens': 100
            },
            'key': 1
        },
    ],
        n_workers=2
    )

    sorted_results = sorted(results, key=lambda x: x[1]) # type: ignore

    assert len(results) == 2
    assert sorted_results[0][1] == 0
    assert sorted_results[1][1] == 1

def test_call_llms_with_retry_fail():
    base_request = lambda x: {
        'model': 'llama3.1:8b',
        'prompt': SimplePrompt('The quick brown fox jumps over the lazy dog.'),
        'options': {
            'num_ctx': x
        }
    }
    results = call_llms([
        base_request(256), # deve passar em todas as tentativas
        base_request(40960), # deve falhar em todas as tentativas
    ])

    print(results)

    assert len(results) == 1

def test_call_llms_with_retry_success():
    base_request = lambda x: {
        'model': x,
        'prompt': SimplePrompt('The quick brown fox jumps over the lazy dog.'),
        'options': {
            'num_ctx': 256
        }
    }
    results = call_llms([
        base_request('llama3.1:8b'), # deve passar em todas as tentativas
        base_request('llama3'), # deve falhar em todas as tentativas
    ])

    print(results)

    assert len(results) == 2
