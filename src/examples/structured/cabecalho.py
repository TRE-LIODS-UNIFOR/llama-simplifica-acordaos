from pprint import pprint
from typing import Callable, Dict, Iterable, List
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from langchain_ollama import ChatOllama
from pydantic import BaseModel, Field

class Recorrente(BaseModel):
    recorrente: str = Field(description="Nome do recorrente")
    advogados: List[str] = Field(description="Nome e OAB do advogado ou dos advogados do recorrente")

class Recorrido(BaseModel):
    recorrido: str = Field(description="Nome do recorrido")
    advogados: List[str] | None = Field(description="Nome e OAB do advogado ou dos advogados do recorrido")

class Cabecalho(BaseModel):
    data_do_julgamento: str | None = Field(description="Data do julgamento")
    numero_do_processo: str = Field(description="Numero do processo ou numero do recurso eleitoral")
    origem: str = Field(description="Origem do processo")
    relator: str = Field(description="Relator do processo")
    recorrentes: List[Recorrente] = Field(description="Lista de recorrentes do processo e seus respectivos advogados")
    recorridos: List[Recorrido] = Field(description="Lista de recorridos do processo e seus respectivos advogados, se houver algum")

llm = ChatOllama(
    base_url='http://10.10.0.99:11434',
    model='llama3.2',
    context_size=10240,
    temperature=0,
    top_k=10,
    # format="json",
)

llm_cabecalho = llm.with_structured_output(Cabecalho)

prompt_cabecalho = PromptTemplate.from_template(
    """### Missão
    Extraia as informações contidas no contexto, de acordo com as instruções. Responda apenas com as informações extraídas, sem incluir as instruções.
    O contexto contém o cabeçalho de um acórdão, onde há um ou mais recorrentes (cada um com um ou mais advogados), sendo cada um demarcado no texto por 'RECORRENTE: ', e um recorrido (que pode ou não ser representado por um ou mais advogados), demarcado no texto por 'RECORRIDO: '. Um recorrente nunca pode ser um recorrido, e vice-versa.

    ### Contexto
    {context}

    ### Formato
    {format_instructions}

    ### Informações estruturadas em JSON"""
)
chain_cabecalho = prompt_cabecalho | llm#_cabecalho

context = """
RECURSO ELEITORAL (11548) nº 0600085-38.2020.6.06.0008.
ORIGEM: ARACATI/CE.
Relator(a): JUIZ ROBERTO SOARES BULCÃO COUTINHO.
RECORRENTE: IVANILDO BARROS FALCÃO.
Advogado: ERIK GOMES SILVEIRA - OABCE41381-A.
RECORRIDO: DIRETÓRIO MUNICIPAL DO PARTIDO DO MOVIMENTO DEMOCRÁTICO BRASILEIRO.
Advogadas(os): DANIELLI GONDIM CAMPELO - OABCE18218-A, ALFREDO NARCISO DA COSTA NETO - OABCE19102-A, LUCAS DA COSTA GUEDES - OABCE42496-A, FRANCISCO RAFAEL FREIRE RAMOS - OABCE25715-A, JULIANNY AMARAL DA COSTA OLIVEIRA - OABCE22747-A, MARCELO DE OLIVEIRA MONTEIRO - OABCE39864-A, CARLOS VICTOR DA COSTA GUEDES - OABCE39870-A, IGOR REBOUCAS PAULA - OABCE33060-A.
"""

context_2 = """
RECURSO ELEITORAL N. 0600012-49.2024.6.06.0033
ORIGEM: CANINDÉ/CE
RELATOR: DESEMBARGADOR ELEITORAL DANIEL CARVALHO CARNEIRO
RECORRENTE: ANTÔNIO ILOMAR VASCONCELOS CRUZ
ADVOGADOS(AS): FRANCISCO JARDEL RODRIGUES DE SOUSA - OAB CE32787-A,
LIDENIRA CAVALCANTE MENDONÇA VIEIRA - OAB CE0016731
RECORRIDO: MINISTÉRIO PÚBLICO ELEITORAL"""

parser = JsonOutputParser(pydantic_object=Cabecalho)
result = chain_cabecalho.invoke({'context': context_2, 'format_instructions': parser.get_format_instructions()})
content = result.content

parsed = parser.parse(content)
formatted = dict()

def unstructured_str_op(x: str | Iterable, op: Callable):
    if isinstance(x, str):
        return op(x)
    elif isinstance(x, list):
        _x = []
        for i in x:
            _x.append(op(i))
        return _x

for k, v in parsed.items():
    _k, _v = k.lower(), unstructured_str_op(v, lambda x: x.title())
    formatted[_k] = _v

pprint(formatted)
