from collections import Counter
from io import BytesIO
import os
from pathlib import Path
import pymupdf


def find_repetitions(doc: bytes | BytesIO, cluster_size: int) -> list[str]:
    pages = extract_text_from_pdf(doc).split("\n-----\n")
    n = len(pages)

    content = "\n".join(pages)
    lines = content.split("\n")
    clusters = []
    for j in range(len(lines) - cluster_size):
        clusters.append("\n".join([line for line in lines[j:j+cluster_size]]))
    counter = Counter(clusters)
    common = counter.most_common(4)
    for i, (cluster, count) in enumerate(common):
        print(f"Cluster {i+1} ({count} repetitions):")
        print(cluster)
        print()

    return counter
    # repetitions = []
    # for page in pages:
    #     text = page.get_text()
    #     lines = text.split("\n")
    #     for line in lines:
    #         if line.strip() in repetitions:
    #             continue
    #         if lines.count(line) > 1:
    #             repetitions.append(line.strip())
    # return repetitions

# def get_header(doc: bytes | BytesIO) -> str:
#     content = pymupdf.Document(stream=doc)
#     cabecalho = content[0].get_text()
#     lines = cabecalho.split("\n")
#     lines = [line.strip() for line in lines if len(line.strip())]

def partition(document: bytes | BytesIO | None, start: int, end: int) -> str:
    if document is None:
        raise ValueError("Document is None")
    pages = extract_text_from_pdf(document).split("\n-----\n")
    return "\n".join(pages[start:end])

def auto_partition(document: str, start_keyword: str, end_keyword: str, page_separator = "\n-----\n") -> dict[str, str]:
    start = end = 0
    pages = document.split(page_separator)
    for i, page in enumerate(pages):
        lines = page.split("\n")
        for line in lines:
            if line.strip().upper() == start_keyword and start_keyword != "":
                start = i
            elif line.strip().upper() == end_keyword and end_keyword != "":
                end = i
                break
    if start_keyword == "":
        start = 0
    if end_keyword == "":
        end = len(pages)
    return start, end

def save_document(out_path: Path, contents: str):
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w") as f:
        f.write(contents)
    print(f"Saved {out_path}")

def extract_text_from_pdf(pdf: bytes | BytesIO, save_document=False, footer_lines=2) -> str:
    """
    Extrai o conteúdo do PDF em 'pdf_path' para texto. Quebras de página serão demarcadas com '\n-----\n'
    """
    doc = pymupdf.Document(stream=pdf)

    contents = []
    for i, page in enumerate(doc):
        text = page.get_text(sort=True)
        text = text.replace(" \n", " ")
        lines = text.split("\n")
        lines = [line.strip() for line in lines]
        lines = [line for line in lines if len(line)]
        if footer_lines:
            text = "\n".join(lines[:-(footer_lines)])
        else:
            text = "\n".join(lines)
        if i < len(doc) - 1:
            text = text + "\n-----\n"
        contents.append(text)
    text_contents = "".join(contents)

    return text_contents

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Extract text from PDF")
    parser.add_argument("pdf_path", type=Path, help="Path to the PDF file")
    args = parser.parse_args()

    with open(args.pdf_path, "rb") as pdf:
        text = extract_text_from_pdf(pdf.read(), False)
    path = Path(args.pdf_path)
    filename = path.stem

    with open(f"documentos/acordaos/{filename}.txt", "w") as f:
        f.write(text)
