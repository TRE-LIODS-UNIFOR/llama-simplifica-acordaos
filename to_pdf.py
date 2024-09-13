import os
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent

def to_pdf(source: Path):
    OUT_DIR = BASE_DIR / f'out/pdf/{source.stem}.pdf'
    os.system(f'pandoc -s {str(source)} -o {str(OUT_DIR)} -v "mainfont:OpenSerif.ttf"')
