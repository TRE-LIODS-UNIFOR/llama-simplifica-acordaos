from langchain.prompts import PromptTemplate
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_community.vectorstores import FAISS
from langchain_ollama import OllamaEmbeddings

from llms import get_llama
from ner import NER
from segment_sentences import segment
from split_documents import split_documents, split_text
from stuff import stuff

def postprocess(processed_result: str, original: str, model_configurations: dict | None = None) -> str:
    """
    Encontra as entidades omitidas na versão resumida, e produz uma nova versão mais completa.
    """
    ner = NER()

    pros_chunks = split_text(processed_result, split_by='character', chunk_size=512, chunk_overlap=32)
    orig_chunks = split_text(original, split_by='character', chunk_size=512, chunk_overlap=32)

    pros_topics = ner.get_topics(pros_chunks)
    orig_topics = ner.get_topics(orig_chunks)

    missing_topics = [topic for topic in orig_topics if topic not in pros_topics]

    refine_prompt = PromptTemplate.from_template(
        """
        Com base no seguinte resumo de um acórdão do TRE e os trechos do contexto, refine o resumo, incluindo as seguintes informações omitidas no resumo original, mantendo sua estrutura original:

        Resumo original:
        {original}

        Informações omitidas:
        {missing}

        Trechos do contexto:
        {context}

        Resumo refinado:
        """
    )

    embeddings_model = OllamaEmbeddings(
        model='nomic-embed-text',
        base_url='http://10.10.0.99:11434',
    )
    original_document = split_text(original)
    vectorstore = FAISS.from_documents(documents=original_document, embedding=embeddings_model)
    retriever = vectorstore.as_retriever()

    llm = get_llama(model_configuration=model_configurations)

    stuff_chain = create_stuff_documents_chain(llm, refine_prompt)
    retrieval_chain = create_retrieval_chain(retriever, stuff_chain)

    return retrieval_chain.invoke({
        'original': processed_result,
        'missing': missing_topics,
        'input': 'Comece.',
    })['answer']

def fact_check(processed_result: str | None = None, original_document_path: str | None = None, page_start: int | None = None, page_end: int | None = None, host: int = 0, base_url: int | None = None, model_configurations: dict | None = None) -> str:
    """
    Verifica a veracidade de um resumo em relação ao documento original.
    """
    if processed_result is None:
        raise ValueError("processed_result must be provided")
    if original_document_path is None:
        raise ValueError("original_document_path must be provided")
    if page_start is None:
        raise ValueError("page_start must be provided")
    if page_end is None:
        raise ValueError("page_end must be provided")
    if base_url is None:
        raise ValueError("base_url must be provided")
    if model_configurations is None:
        raise ValueError("model_configurations must be provided")
    filtered_result = processed_result.replace('**', '')
    sentences = segment(filtered_result)

    doc = split_documents(original_document_path, page_start, page_end)
    fact_check_prompt = """Com base no seguinte resumo de um acórdão, verifique a veracidade da seguinte afirmação, respondendo com 'Sim' se for verídica, e 'Não' se for falsa.

Resumo:
{resumo}

Afirmação:
{afirmacao}

Contexto:
{context}

Resposta:
"""

    checks = []
    for sentence in sentences:
        print(sentence)
        res = stuff(
            doc,
            prompt=fact_check_prompt,
            template_kvs={'afirmacao': sentence, 'resumo': processed_result},
            model_configuration=model_configurations
        )
        print(res)
        checks.append(res)

    correcoes_prompt = """A seguinte afirmação está presente no resumo. Verifique se a afirmação é verdadeira comparando com o texto no contexto. Apresente o raciocínio passo a passo e apresente uma versão corrigida da afirmação.

Afirmação:
{afirmacao}

Contexto:
```
{context}
```

Versão corrigida:
"""

    resumo_corrigido = processed_result
    for afirmacao, check in zip(sentences, checks):
        if check == 'Sim.':
            continue
        else:
            resumo_corrigido = stuff(doc, prompt=correcoes_prompt, template_kvs={'afirmacao': afirmacao})
            print(resumo_corrigido, '\n')

    return resumo_corrigido

if __name__ == "__main__":
    processed_result = """**Relatório (O Caso)**: O caso envolve um recurso eleitoral apresentado pelo candidato Ilomar, que alega que sua pré-candidatura foi prejudicada pela realização de propaganda eleitoral antecipada. O juiz decidiu que o ato do candidato Ilomar foi caracterizado como propaganda eleitoral antecipada e imposta a penalidade do artigo 36, § 3º, da Lei 9.504/97.

**Informações iniciais do processo**: O autor do recurso pediu a anulação da decisão do juiz que considerou o ato do candidato Ilomar como propaganda eleitoral antecipada e imposta a penalidade do artigo 36, § 3º, da Lei 9.504/97.

**Decisão do juiz no processo inicial**: O juiz decidiu que o ato do candidato Ilomar foi caracterizado como propaganda eleitoral antecipada e imposta a penalidade do artigo 36, § 3º, da Lei 9.504/97.

**Artigos de lei e fundamentos jurídicos relevantes**: O artigo 36 da Lei 9.504/97 estabelece que a propaganda eleitoral antecipada é proibida e pode resultar em multa. Além disso, o artigo 3º do mesmo diploma legal define a propaganda eleitoral como qualquer ato que tenha por objetivo influenciar o voto dos eleitores.

**Quem recorreu e o que alegou**: O candidato Ilomar recorreu à decisão do juiz, alegando que sua pré-candidatura foi prejudicada pela propaganda eleitoral antecipada e que o ato dele não foi caracterizado como tal.

**Caixa de texto explicativa com termos jurídicos relevantes**: A propaganda eleitoral antecipada é uma prática proibida pela lei, que envolve a realização de atos que tenham por objetivo influenciar o voto dos eleitores antes do período autorizado pela legislação. Além disso, o princípio da
igualdade de oportunidades entre os candidatos foi vulnerado, pois o ato do candidato Ilomar foi caracterizado como propaganda eleitoral antecipada e imposta a penalidade do artigo 36, § 3º, da Lei 9.504/97
"""[24:]
    paragraphs = processed_result.split("\n\n")

    print(fact_check(processed_result=paragraphs[0], original_document_path='documentos/acordaos/0600012-49_REl_28052024_1.txt', page_start=2, page_end=5))
