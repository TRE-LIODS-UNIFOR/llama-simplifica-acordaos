from pprint import pprint
import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from langchain_core.documents import Document
from stuff import most_similar, n_stuff

def test_n_stuff():
    docs = [Document(page_content='In 1991, there was a man named Jerry. Jerry was a race car driver. He drove so goddamn fast. He never did win no checkered flags. But he never did come in last')]
    res, scores = n_stuff(n=4, docs=docs, prompt="Based on the following text: \"{context}\", What was Jerry's profession?")
    best = most_similar(res, scores)

    print(res)
    print("best:", best)
