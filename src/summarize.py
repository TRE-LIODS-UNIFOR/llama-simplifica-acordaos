from pprint import pprint
from llms import get_llama
from ner import NER
from semantic_similarity import get_similarity_score
from stuff import most_similar, n_stuff, stuff
from split_documents import split_documents
from langchain.prompts import PromptTemplate
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


def summarize_section(doc_path, start_page, end_page, prompt, base_url=None, verbose=False):
    doc = split_documents(doc_path, start_page, end_page)
    prompt_template = PromptTemplate.from_template(prompt)

    if verbose:
        print("Processamento inicial")
    process = n_stuff(n=3, docs=doc, results={'result': {}}, key='result', prompt=prompt_template, base_url=base_url, verbose=verbose)
    res, score = most_similar(*process)
    if verbose:
        print("Resposta mais similar:")
        print(score)
        pprint(res)
    pass

    if verbose:
        print("Pós-processamento")
        print("Detectando omissões")
    post = postprocess(res, doc_path, start_page, end_page, base_url=base_url)
    if verbose:
        print(post)

    llm = get_llama(base_url=base_url)

    if verbose:
        print("Removendo redundâncias")
    redundant_sentences_prompt = Prompts.SENTENCAS_REDUNDANTES
    redundant_sentences_chain = PromptTemplate.from_template(redundant_sentences_prompt) | llm
    redundant_sentences_res = redundant_sentences_chain.invoke({'refined': post})

    if verbose:
        print("Refinando texto")
    refine_prompt = PromptTemplate.from_template(Prompts.REFINAR_REDUNDANCIAS)
    refine_chain = refine_prompt | llm
    refine_res = refine_chain.invoke({'refined': post, 'redundancies': redundant_sentences_res.content})

    original_content = "\n".join([page.page_content for page in doc])

    return refine_res.content, get_similarity_score(refine_res.content, original_content, method='bertscore')


if __name__ == "__main__":
    doc_path = "documentos/acordaos/0600012-49_REl_28052024_1.txt"

    print('Relatório')
    start_page = 2
    end_page = 5
    prompt = Prompts.RELATORIO
    relatorio = summarize_section(doc_path, start_page, end_page, prompt, verbose=True)
    print(relatorio, '\n\n\n')

    # print('Voto')
    # start_page = 5
    # end_page = 11
    # prompt = Prompts.VOTO
    # voto = summarize_section(doc_path, start_page, end_page, prompt, verbose=True)
    # print(voto, '\n\n\n')

    # print('Decisão')
    # start_page = 11
    # end_page = 12
    # prompt = Prompts.DECISAO
    # decisao = summarize_section(doc_path, start_page, end_page, prompt, verbose=True)
    # print(decisao)
