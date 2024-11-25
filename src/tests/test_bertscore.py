import sys
import os

import torch

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import bert_score

from config import Config

def test_load_model():
    bert_score.BERTScorer(lang='pt', model_type=Config.BERTSCORE_MODEL, num_layers=Config.BERTSCORE_MODEL_N_LAYERS)

def test_chunks():
    chunks = bert_score.score(['a', 'b', 'c'], ['d'] * 3, lang='pt', model_type=Config.BERTSCORE_MODEL, num_layers=Config.BERTSCORE_MODEL_N_LAYERS)
    assert len(chunks) == 3
    assert len(chunks[0]) == 3
