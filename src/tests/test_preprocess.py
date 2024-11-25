from pprint import pprint
import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from preprocess import find_repetitions
from preprocess import extract_text_from_pdf

def test_find_repetitions():
    with open("documentos/acordaos/0600012-49_REl_28052024_1.pdf", "rb") as f:
        doc = f.read()
    repetitions = find_repetitions(doc, 7)

    # print(repetitions)

def test_extract_text_from_pdf():
    footer_lines = 2

    pdf_path = "documentos/acordaos/0600012-49_REl_28052024_1.pdf"
    with open(pdf_path, "rb") as f:
        pdf = f.read()
    content = extract_text_from_pdf(pdf, footer_lines=footer_lines)

    print(content)

    # pdf_path_2 = "documentos/acordaos/0600059-75_REl_03102023_1.pdf"
    # with open(pdf_path_2, "rb") as f:
    #     pdf_2 = f.read()
    # content_2 = extract_text_from_pdf(pdf_2, footer_lines=footer_lines)

    pages = content.split("\n-----\n")
    cabecalho = pages[0]
    print(cabecalho)
    relatorio = pages[1:3]
    lines = [line for line in relatorio[0].split("\n")] + [line for line in relatorio[1].split("\n")]
    print(lines)
    # print(content_2.split("\n-----\n")[0])

    lines = content.split("\n")

    assert len([line for line in lines if len(line) == 0]) == 0
