from io import BytesIO
from flask import Flask, jsonify, request
from werkzeug.datastructures import FileStorage

import preprocess
from simplify import simplify
from summarize import Prompts, summarize_section

def summarize(doc: FileStorage, sections: list[int] | None = None) -> str:
    pdf: BytesIO = BytesIO(doc.read())
    if sections is None:
        raise ValueError("Sections must be provided")
    # cabecalho = preprocess.partition(pdf, sections[0], sections[1])
    relatorio: str = preprocess.partition(pdf, sections[1], sections[2])
    # voto = preprocess.partition(pdf, sections[2], sections[3])
    # decisao = preprocess.partition(pdf, sections[3], sections[4])
    # document = {
    #     "cabecalho": cabecalho,
    #     "relatorio": relatorio,
    #     "voto": voto,
    #     "decisao": decisao
    # }
    print("\n\nSUMMARIZING\n\n")
    summaries: dict[str, str] = {
        "relatorio": summarize_section(relatorio, prompt=Prompts.RELATORIO, verbose=True, n_factor=2),
        # "voto": summarize_section(voto, prompt=Prompts.VOTO),
        # "decisao": summarize_section(decisao, prompt=Prompts.DECISAO),
    }

    print("\n\nSIMPLIFYING\n\n")

    simplified = {
        "relatorio": simplify(summaries["relatorio"]),
    #     "voto": simplify(summaries["voto"]),
    #     "decisao": simplify(summaries["decisao"]),
    }
    # result = "\n\n".join([f"{k.capitalize()}\n\n{v[0]}" for k, v in simplified.items()])
    return simplified['relatorio'][0]

app = Flask(__name__)

@app.post("/test/simplify")
def test_simplify():
    if request.json is None:
        raise ValueError("No JSON provided")
    text = request.json['text']
    if text is None:
        return "No text provided"
    result = simplify(text)
    return jsonify(result[0])

@app.post("/simplify")
def process():
    doc = request.files['doc']
    sections = [int(i) for i in request.form.getlist('sections')]
    print(sections)
    return jsonify(summarize(doc, sections))

if __name__ == "__main__":
    app.run()
