import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from split_documents import split_text
from ner import NER

def test_get_topics():
    text = 'Ana foi ao mercado. Bruno comprou um carro. Carlos foi ao cinema. Daniel comprou um celular.'
    document = split_text(text)
    ner = NER()
    topics = ner.get_topics(document)

    assert type(topics) == list
    assert 'Ana' in topics
