from pathlib import Path
from pprint import pprint
import sys
from call_llms import call_llms
from config import Config
from llms import get_llama
from mapreduce import mapreduce, parallel_mapreduce
from ner import NER
import preprocess
from prompts.prompts import SimplePrompt
import rank_responses
from semantic_similarity import get_similarity_score
from stuff import most_similar, n_stuff
from split_documents import split_text
from langchain.prompts import PromptTemplate
from langchain_core.documents import Document
from postprocessing import postprocess


class Prompts:
    CABECALHO = """
    Extraia do texto encontrado no contexto somente as seguintes informações:

    -   Número do processo (ou número do recurso eleitoral)
    -   Origem
    -   Relator
    -   Recorrente
    -   Advogados(as)
    -   Recorrido
    -   Assunto

    Formate sua resposta como uma lista, assim:

    -   Número do processo: Número do processo
    -   Origem: Origem
    -   Relator: Relator
    -   Recorrente: Recorrente
    -   Advogados: Advogados(as)
    -   Recorrido: Recorrido
    -   Assunto: Assunto, em 2 ou 3 palavras

    ### Exemplo de entrada:

    ```
    PODER JUDICIÁRIO
    TRIBUNAL REGIONAL ELEITORAL DO CEARÁ

    RECURSO ELEITORAL N. 0600012-49.2024.6.06.0033
    ORIGEM: CANINDÉ/CE
    RELATOR: DESEMBARGADOR ELEITORAL DANIEL CARVALHO CARNEIRO
    RECORRENTE: ANTÔNIO ILOMAR VASCONCELOS CRUZ
    ADVOGADOS(AS): FRANCISCO JARDEL RODRIGUES DE SOUSA - OAB CE32787-A,
    LIDENIRA CAVALCANTE MENDONÇA VIEIRA - OAB CE0016731
    RECORRIDO: MINISTÉRIO PÚBLICO ELEITORAL
    ```

    ### Exemplo de saída:

    -   Número do processo: 0600012-49.2024.6.06.0033
    -   ORIGEM: CANINDÉ/CE
    -   RELATOR: DESEMBARGADOR ELEITORAL DANIEL CARVALHO CARNEIRO
    -   RECORRENTE: ANTÔNIO ILOMAR VASCONCELOS CRUZ
    -   ADVOGADOS(AS): FRANCISCO JARDEL RODRIGUES DE SOUSA - OAB CE32787-A, LIDENIRA CAVALCANTE MENDONÇA VIEIRA - OAB CE0016731
    -   RECORRIDO: MINISTÉRIO PÚBLICO ELEITORAL

    ### Contexto

    ```
    {context}
    ```

    ### Fim do contexto

    ### Resposta
    """
    RELATORIO = """
    Você é um especialista jurídico com foco em simplificação de textos legais. Sua tarefa é analisar acórdãos judiciais e gerar um resumo simplificado, acessível ao público geral, sem perder a precisão jurídica. O formato do documento simplificado deve ser claro, objetivo e seguir uma estrutura pré-definida.
    Preencha o acórdão simplificado utilizando as informações fornecidas em cada bloco. Siga o formato abaixo para garantir que todos os elementos necessários sejam cobertos e organizados conforme a estrutura do acórdão:
    Responda com o texto simplificado do acórdão, e nada mais.

    ### FORMATO
    ```
    **Relatório (O Caso)**: Resuma de forma objetiva os fatos apresentados no acórdão, destacando em 3 parágrafos, com no máximo 3 linhas cada um, e, quando for o caso, caixa de texto explicativa, tudo em texto contínuo.

    Informações iniciais do processo analisado pelo juiz: indique o que o autor do recurso pediu e o que o réu alegou para se defender.
    Decisão do juiz no processo inicial: Apresente a decisão do juiz no processo inicial, descrevendo o que o juiz decidiu e as justificativas legais usadas.
    Artigos de lei e fundamentos jurídicos relevantes: cite artigos de lei e fundamentos jurídicos relevantes..
    Quem recorreu e o que alegou: indique quem recorreu à decisão e o que alegou para recorrer.

    Caixa de texto explicativa com termos jurídicos relevantes para a compreensão do assunto principal: forneça definições e explicações simples de termos, expressões ou assuntos jurídicos relevantes para a compreensão do assunto principal. Exemplo: “Propaganda antecipada negativa: A propaganda eleitoral antecipada negativa acontece quando, antes de 16 de agosto do ano eleitoral (art. 36 da Lei nº 9.504/1997), alguém faz críticas para prejudicar adversários políticos e influenciar eleitores. Essa prática é proibida e pode resultar em multa”.
    ```
    ### FIM DO FORMATO

    ### CONTEXTO
    ```
    {context}
    ```
    ### FIM DO CONTEXTO
    """
    VOTO = """
    Você é um especialista jurídico com foco em simplificação de textos legais. Sua tarefa é analisar acórdãos judiciais e gerar um resumo simplificado, acessível ao público geral, sem perder a precisão jurídica. O formato do documento simplificado deve ser claro, objetivo e seguir uma estrutura pré-definida.
    Preencha o acórdão simplificado utilizando as informações fornecidas em cada bloco. Siga o formato abaixo para garantir que todos os elementos necessários sejam cobertos e organizados conforme a estrutura do acórdão:
    Responda com o texto simplificado do acórdão, e nada mais.

    ### FORMATO
    ```
    **Voto (Argumentação/ Motivação/ Fundamentação)**: Apresente as razões jurídicas que embasaram a decisão do Tribunal, destacando, em 2 parágrafos, com no máximo 3 linhas, de texto contínuo.

    Princípios legais aplicados: quais foram os princípios legais aplicados.
    Interpretação do Tribunal: A interpretação dada à legislação.
    Argumentos relevantes: Argumentos relevantes usados na motivação.**

    (Use uma linguagem simples e evite jargões técnicos sempre que possível)

    Fundamentação de Fato e de Direito: Explique os fatos relevantes para a decisão e apresente a fundamentação jurídica, citando artigos de lei, precedentes jurisprudenciais que embasam a decisão do Tribunal.
    Conclusão e Decisão: Conclusão do Voto.
    ```
    ### FIM DO FORMATO

    ### CONTEXTO
    ```
    {context}
    ```
    ### FIM DO CONTEXTO
    """
    DECISAO = """
    Você é um especialista jurídico com foco em simplificação de textos legais. Sua tarefa é analisar acórdãos judiciais e gerar um resumo simplificado, acessível ao público geral, sem perder a precisão jurídica. O formato do documento simplificado deve ser claro, objetivo e seguir uma estrutura pré-definida.
    Preencha o acórdão simplificado utilizando as informações fornecidas em cada bloco. Siga o formato abaixo para garantir que todos os elementos necessários sejam cobertos e organizados conforme a estrutura do acórdão:

    Contexto:
    ```
    {context}
    ```
    Fim do contexto

    Formato:
    ```
    ### **Resultado do Julgamento (Dispositivo):**

    Resultado final da decisão: Informe de maneira direta e clara o resultado final da decisão, incluindo em 2 parágrafos, com no máximo 3 linhas em cada um, em texto contínuo.
    Resultado do Julgamento: Aqui, deverá ser informado o resultado final da decisão, se a decisão foi unânime ou não, se manteve a decisão original ou a modificou. Exemplo: “O Tribunal Regional Eleitoral do Ceará, por unanimidade, decidiu manter a decisão original, entendendo que...".

    Decisão: de acordo com o resultado da decisão, uma entre: "Recurso Negado", "Recurso Aceito" ou "Recurso Parcialmente Aceito".
    ```
    Fim do formato

    Resumo simplificado:
    """

    TOPICOS_OMITIDOS = """
    Com base no seguinte resumo de um acórdão do TRE e os trechos do contexto, refine o resumo, incluindo as seguintes informações omitidas no resumo original, mantendo sua estrutura original:

    Resumo original:
    {original}

    Informações omitidas:
    {missing}

    Trechos do contexto:
    {context}

    Resumo refinado:
    """

    SENTENCAS_REDUNDANTES = """
Encontre sentenças redundantes no texto refinado e liste-as.

Texto refinado:
{refined}

Lista de sentenças redundantes:
"""

    REFINAR_REDUNDANCIAS = """
Com base na lista de redundâncias, remova as sentenças redundantes do texto refinado. Responda com o texto refinado, e nada mais.

Lista de sentenças redundantes:
{redundancies}

Texto refinado:
{refined}

Texto refinado sem redundâncias:"""


def summarize_section(document_contents: str, prompt: str | None = None, verbose: bool = False, n_factor: int = 3, skip_postprocess: bool = False) -> str: # tuple[str, float]:
    if prompt is None:
        raise ValueError("prompt must be provided")

    # Dividindo o texto em chunks
    docs: list[Document] = split_text(document_contents)
    # Produzindo o primeiro resumo, com MapReduce
    resumos: list[str] = [res['response'] for res in parallel_mapreduce(docs)]
    resumo: str = rank_responses.by_similarity(resumos, document_contents, 'bertscore')[0][0]
    # resumo = postprocess(resumo, document_contents)
    print("Tamaho do resumo em B:", sys.getsizeof(resumo))
    docs: list[Document] = split_text(resumo)

    if verbose:
        print("Processamento inicial")
    # Processando o resumo inicial n vezes para obter a resposta mais similar
    res, score = n_stuff(n=n_factor, docs=docs, prompt=prompt) # type: ignore
    if res is None:
        raise ValueError("Nenhuma resposta obtida")
    if score is None:
        raise ValueError("Nenhuma pontuação obtida")

    if verbose: print("Escolhendo a melhor resposta")
    pprint(res)
    pprint(score)
    # res, score = rank_responses.by_similarity(process, document_contents)[0]
    response_contents = [response for response in res] # type: ignore
    pprint(response_contents)
    best, best_score = most_similar(response_contents, score)
    if verbose:
        print("Resposta mais similar:")
        print(best_score)

    if skip_postprocess:
        return best

    if verbose:
        print("Pós-processamento")
        print("Detectando omissões")
    # Pós-processamento: detectando omissões, refinando o resumo
    post: str = postprocess(best, document_contents)
    if verbose:
        print(post)

    if verbose:
        print("Removendo redundâncias")
    redundant_sentences_res: list[tuple[dict[str, str], int | None]] = call_llms([
        {
            "model": Config.OLLAMA_MODEL,
            "prompt": SimplePrompt(Prompts.SENTENCAS_REDUNDANTES.format(refined=post)),
            "options": {
                'temperature': 0.25,
                'top_k': 5,
                'top_p': 0.25,
            }
        }
    ])
    res: list[str] = [res[0]['response'] for res in redundant_sentences_res]
    # best: str = rank_responses.by_similarity(res, post, 'bertscore')[0][0]
    best: str = res[0]
    if verbose:
        print(best)

    if verbose:
        print("Refinando texto")
    refine_res: list[tuple[dict[str, str], int | None]] = call_llms([
        {
            "model": Config.OLLAMA_MODEL,
            "prompt": SimplePrompt(Prompts.REFINAR_REDUNDANCIAS.format(redundancies=best, refined=post)),
            "options": {
                'temperature': 0.25,
                'top_k': 5,
                'top_p': 0.25,
            }
        } for _ in range(n_factor)
    ])
    res: list[str] = [res[0]['response'] for res in refine_res]
    best: str = rank_responses.by_similarity(res, post, 'bertscore')[0][0]
    if verbose:
        print(best)

    return best #, get_similarity_score(best, document_contents)

def alternate_prompt(prompt):
    escaped = prompt.replace('{', '{{').replace('}', '}}')

    llm = get_llama()
    alternate = PromptTemplate.from_template("""Gere 3-5 versões do seguinte prompt, que contém instruções para resumir um acórdão, e um modelo de resposta. Ajuste as instruções, mas mantenha o modelo intacto.

    ```
    {prompt}
    ```

    Versão alternativa:""")
    chain = alternate | llm
    res = chain.invoke({'prompt': escaped})
    return res.content


if __name__ == "__main__":
    # print(alternate_prompt(Prompts.RELATORIO))

    with open("documentos/acordaos/0600012-49_REl_28052024_1.pdf", "rb") as f:
        doc = f.read()
    document_contents = preprocess.extract_text_from_pdf(doc)
    relatorio_contents = preprocess.partition(doc, 2, 5)

    print('Relatório')
    print('Map Reduce')
    doc = split_text(relatorio_contents)
    resumo = mapreduce(doc)
    resumo = postprocess(resumo, relatorio_contents)

    prompt = Prompts.RELATORIO
    relatorio = summarize_section(
        doc,
        prompt,
        verbose=True,
    )
    print(relatorio, '\n\n\n')
