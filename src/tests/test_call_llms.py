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

    sorted_results = sorted(results, key=lambda x: x[1])

    assert len(results) == 2
    assert sorted_results[0][1] == 0
    assert sorted_results[1][1] == 1
