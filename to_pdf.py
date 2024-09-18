from pathlib import Path
import subprocess

BASE_DIR = Path(__file__).parent

def to_pdf(source: Path):
    OUT_DIR = BASE_DIR / f'out/pdf/{source.stem}.pdf'
    subprocess.call([f'pandoc -s {str(source)} -o {str(OUT_DIR)} -V "mainfont:OpenSerif.ttf"'])
