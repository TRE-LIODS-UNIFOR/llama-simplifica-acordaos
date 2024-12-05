import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from postprocessing import postprocess

def test_postprocess():
    processed_result = """Teste."""
    original = """Testando."""
    result = postprocess(processed_result, original)

    assert result
