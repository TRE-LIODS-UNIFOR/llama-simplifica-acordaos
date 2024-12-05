from call_llms import call_llms
from config import Config
from langchain_core.documents import Document
from ner import NER
from prompts.rag_prompt import RAGPrompt
from segment_sentences import segment
from split_documents import split_documents, split_text
from stuff import stuff
from token_count import get_token_count

def postprocess(processed_result: str, original: str) -> str:
    """
    Encontra as entidades omitidas na versão resumida, e produz uma nova versão mais completa.
    """
    ner: NER = NER()

    pros_chunks: list[Document] = split_text(processed_result, split_by='character', chunk_size=512, chunk_overlap=32)
    orig_chunks: list[Document] = split_text(original, split_by='character', chunk_size=512, chunk_overlap=32)

    pros_topics = ner.get_topics(pros_chunks)
    orig_topics = ner.get_topics(orig_chunks)

    missing_topics: list[str] = [topic for topic in orig_topics if topic not in pros_topics]
    missing_topics_text: str = '\n* '.join(missing_topics)

    n_tokens: int = get_token_count(processed_result + missing_topics_text)

    if n_tokens > Config.OLLAMA_CONTEXT_SIZE:
        n_missing_topics = len(missing_topics)
        mid = n_missing_topics // 2
        missing_tokens_list = ['\n* '.join(missing_topics[:mid]), '\n* '.join(missing_topics[mid:])]
    else:
        missing_tokens_list = ['\n* '.join(missing_topics)]

    result: list[tuple[dict[str, str], int | None]] = call_llms([
        {
            'prompt': RAGPrompt(prompt="Com base no seguinte resumo de um acórdão do TRE e os trechos do contexto, refine o resumo, incluindo as seguintes informações omitidas no resumo original, mantendo sua estrutura original:\n\nResumo original:\n{original}\n\nInformações omitidas:\n{missing}\n\nTrechos do contexto:\n{context}\n\nResumo refinado:"),
            'options': {
                'input_dict': {
                    'original': processed_result,
                    'missing': missing_tokens_list[i]
                },
                'documents': orig_chunks,
            }
        } for i in range(len(missing_tokens_list))
    ])

    return result[0][0]['response']

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
