from pprint import pprint
import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from segment_sentences import join_paragraphs, paragraph_mask, segment

def test_segment():
    sentence = """Esta é uma frase. Esta é outra frase.

Um novo parágrafo. Um pouco de texto.

Mais uma frase."""
    sentences = segment(sentence)

    assert len(sentences) == 5
    assert sentences[0] == "Esta é uma frase."
    assert sentences[1] == "Esta é outra frase.\n\n"
    assert sentences[2] == "Um novo parágrafo."
    assert sentences[3] == "Um pouco de texto.\n\n"
    assert sentences[4] == "Mais uma frase."

def test_paragraph_mask():
    segments = [
        "Esta é uma frase.",
        "Esta é outra frase.\n\n",
        "Um novo parágrafo.",
        "Um pouco de texto.\n\n",
        "Mais uma frase."
    ]
    mask = paragraph_mask(segments)

    assert mask == [1, 3]

def test_join_paragraphs():
    segments = [
        "Esta é uma frase.", "Esta é outra frase.\n\n",
        "Um novo parágrafo.", "Um pouco de texto.\n\n",
        "Mais uma frase."
    ]
    mask = [1, 3]
    paragraphs = join_paragraphs(segments, mask)

    assert len(paragraphs) == 3
    assert paragraphs[0] == "Esta é uma frase. Esta é outra frase."
    assert paragraphs[1] == "Um novo parágrafo. Um pouco de texto."
    assert paragraphs[2] == "Mais uma frase."
