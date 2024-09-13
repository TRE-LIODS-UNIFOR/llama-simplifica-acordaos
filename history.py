from pathlib import Path
import time

BASE_DIR = Path(__file__).resolve().parent
OUT_DIR = BASE_DIR / 'out'

def save(name: str, prompt: str, response: str) -> None:
    now = time.strftime("%H-%M-%S_%d-%m-%y")
    out_name: str = f"{name}_{now}"
    prompt_name: str = f"{out_name}_p.txt"
    res_name: str = f"{out_name}_r.md"

    print(f"Salvando {prompt_name}...")
    with open(OUT_DIR / prompt_name, "w") as f:
        f.write(prompt)
    print(f"Salvando {res_name}...")
    with open(OUT_DIR / res_name, "w") as f:
        f.write(response)
