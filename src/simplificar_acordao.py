from pathlib import Path
import time
import pandas as pd
import sys

from api import summarize

BASE_DIR = Path(__file__).resolve().parent

def simplificar_acordao(file_path: str | Path, sections: list[int]) -> str:
    file = open(Path(file_path), 'rb')
    result = summarize(file, sections)
    file.close()
    return result

def from_csv(file_path: str | Path) -> pd.DataFrame:
    file_path = Path(file_path)
    df: pd.DataFrame = pd.read_csv(file_path, index_col=0)
    return df

if __name__ == "__main__":
    dataset_path = BASE_DIR / sys.argv[1]
    results_path = BASE_DIR / "results"
    documents_dir = BASE_DIR / "documentos" / "acordaos"

    if not results_path.exists():
        results_path.mkdir()

    acordaos = from_csv(dataset_path).iterrows()

    for i, acordao in acordaos:
        file_path = documents_dir / acordao["path"]
        sections = [0, acordao["relatorio_pagina"], acordao["voto_pagina"], acordao["decisao_pagina"], acordao["num_paginas"]]
        a = time.monotonic()
        print(f"Processing {file_path}.", end=" ")
        result = simplificar_acordao(file_path, sections)
        with open(results_path / f"{file_path.stem}.txt", "w") as f:
            f.write(result)
        b = time.monotonic()
        with open(results_path / f"{file_path.stem}_time.txt", "w") as f:
            f.write(f"Time: {(b - a) / 60:.2f} minutes\n\n")
        print(f"Done. Time: {(b - a) / 60:.2f} minutes.")
