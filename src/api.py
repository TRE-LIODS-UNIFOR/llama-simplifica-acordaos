from io import BytesIO
from flask import Flask, request
from werkzeug.datastructures import FileStorage

import preprocess
from simplify import simplify
from summarize import Prompts, summarize_section

def summarize(doc: FileStorage, sections: list[int] = None) -> dict[str]:
    pdf = BytesIO(doc.read())
    cabecalho = preprocess.partition(pdf, sections[0], sections[1])
    relatorio = preprocess.partition(pdf, sections[1], sections[2])
    voto = preprocess.partition(pdf, sections[2], sections[3])
    decisao = preprocess.partition(pdf, sections[3], sections[4])
    # document = {
    #     "cabecalho": cabecalho,
    #     "relatorio": relatorio,
    #     "voto": voto,
    #     "decisao": decisao
    # }
    summaries = {
        "relatorio": summarize_section(relatorio, prompt=Prompts.RELATORIO, verbose=True),
        # "voto": summarize_section(voto, prompt=Prompts.VOTO),
        # "decisao": summarize_section(decisao, prompt=Prompts.DECISAO),
    }
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
    text = request.json['text']
    result = simplify(text)
    return result[0]

@app.post("/simplify")
def process():
    doc = request.files['doc']
    sections = [int(i) for i in request.form.getlist('sections')]
    print(sections)
    return summarize(doc, sections)

if __name__ == "__main__":
    app.run()
