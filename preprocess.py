from functools import reduce
from pathlib import Path
from typing import List
import pandas as pd
import pymupdf


def preprocess(pdf_path: Path, out_dir: Path) -> None:
    doc = pymupdf.open(pdf_path)

    with open(out_dir / f"{pdf_path.stem}.txt", "w") as out:
        for page in doc:
            text = page.get_text().encode("utf8")
            decoded = text.decode("utf8")
            out.writelines("\n".join(decoded.split("\n")[:-4]))
            out.write("\n-----\n")

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

def print_tables(pdf_path):
    doc = pymupdf.open(pdf_path)
    for page in doc:
        tabs = page.find_tables(vertical_lines=[42, 178, 462])
        if tabs.tables:
            for tab in tabs:
                print(tab.extract())


if __name__ == "__main__":
    import sys
    import pprint

    dicionario_juridico_vertical_lines = [42, 178, 462]
    pdf_path = Path(sys.argv[1])
    # out_dir = Path(sys.argv[2])
    # preprocess(pdf_path, out_dir)
    # print_tables(pdf_path)

    filterNone = lambda table: [i for i in list(filter(lambda j: not j is None, table))]

    table = read_tables(pdf_path, dicionario_juridico_vertical_lines)
    # pprint.pprint(filterNone(table[1:]))
    # pprint.pprint(table[1:])
    # formatted = map(
    #     lambda row:
    #         [
    #             row[0],
    #             "".join(
    #                 row[1:]
    #             ).replace("\n", " ")
    #         ],
    #     table[1:]
    # )
    # formatted = list(formatted)
    # print(formatted)
    # pprint.pprint(formatted)
