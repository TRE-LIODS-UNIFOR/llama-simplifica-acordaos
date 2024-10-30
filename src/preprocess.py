import os
from pathlib import Path
from typing import List
import pandas as pd
import pymupdf

from collections import Counter


def separar(document: str, start_keyword: str, end_keyword: str, page_separator = "\n-----\n") -> dict[str, str]:
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

def extract_text_from_pdf(pdf_path: Path, save_document=False) -> Path:
    """
    Extrai o conteúdo do PDF em 'pdf_path' para um arquivo de texto que será
    criado em 'out_dir'
    Quebras de página serão demarcadas com '\n-----\n'
    """
    doc = pymupdf.open(pdf_path)

    contents = []
    for page in doc:
        text = page.get_text(sort=True).encode("utf8")
        decoded = text.decode("utf8")
        decoded = decoded.replace(" \n", " ")
        decoded = decoded.strip()
        decoded = "\n".join(list(filter(lambda line: len(line), decoded.split("\n"))))
        decoded = "\n".join(decoded.split("\n"))
        decoded = decoded + "\n-----\n"
        contents.append(decoded)
    text_contents = "\n".join(contents)
    # lines = [[line.strip().lower() for line in lines.split("\n")] for lines in contents]
    lines = [line.strip().lower() for lines in contents for line in lines.split("\n")]
    # print(lines)

    counter = Counter(lines)
    top = counter.most_common(20)
    keys = [k[:25] for k in list(counter.keys())]
    values = list(counter.values())
    for k, v in top:
        print(k, v)

    return text_contents

def read_tables(pdf_path: Path, vertical_lines: List[float]) -> List:
    doc: pymupdf.Document = pymupdf.open(pdf_path)
    table: pd.DataFrame = pd.DataFrame()
    for page in doc:
        tabs = page.find_tables(vertical_lines=vertical_lines)
        if tabs.tables:
            for tab in tabs:
                tab.insert(0, [i for i in range(len(tab[0]))])
                df = tab.to_pandas()
                filtered = df.fillna(value='')
                print(filtered)
                table.append(tab.extract()[0])
    return table

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Extract text from PDF")
    parser.add_argument("pdf_path", type=Path, help="Path to the PDF file")
    parser.add_argument("out_dir", type=Path, help="Path to the output directory", nargs='*')
    args = parser.parse_args()

    text = extract_text_from_pdf(args.pdf_path, args.out_dir, False)
    filename = Path(args.pdf_path).stem

    cabecalho = separar(text, "", "RELATÓRIO")
    relatorio = separar(text, "RELATÓRIO", "VOTO")
    voto = separar(text, "VOTO", "DISPOSITIVO")
    dispositivo = separar(text, "DISPOSITIVO", "")

    print(cabecalho[0], cabecalho[1])
    print(relatorio[0], relatorio[1])
    print(voto[0], voto[1])
    print(dispositivo[0], dispositivo[1])

    pages = text.split("\n-----\n")
    out_dir = Path(os.path.dirname(args.pdf_path))
    save_document(out_dir / filename / 'cabecalho.txt', "\n".join(pages[cabecalho[0]:cabecalho[1]]))
    save_document(out_dir / filename / 'relatorio.txt', "\n".join(pages[relatorio[0]:relatorio[1]]))
    save_document(out_dir / filename / 'voto.txt', "\n".join(pages[voto[0]:voto[1]]))
    save_document(out_dir / filename / 'dispositivo.txt', "\n".join(pages[dispositivo[0]:dispositivo[1]]))

    # extract_text_from_pdf(args.pdf_path, args.out_dir)
    # print(f"Text extracted from {args.pdf_path} to {args.out_dir}")

