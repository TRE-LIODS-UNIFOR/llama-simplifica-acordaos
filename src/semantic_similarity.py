from bert_score import score
from llms import get_embeddings_model
from langchain_community.utils.math import cosine_similarity

from split_documents import split_documents


def get_similarity_score(summarized: str, original: str, method: str = 'cosine') -> float:
    if method == 'cosine':
        embeddings_model = get_embeddings_model()
        summarized_embeddings = [embeddings_model.embed_query(summarized)]
        original_embeddings = [embeddings_model.embed_query(original)]

        similarity = cosine_similarity(summarized_embeddings, original_embeddings)
        return similarity[0][0]
    elif method == 'bertscore':
        # model_name = 'rufimelo/Legal-BERTimbau-base'
        return score([summarized], [original], lang='pt')[2].item()


if __name__ == "__main__":
    resumo_impreciso = """
**Relatório**: O Tribunal Regional Eleitoral do Ceará (TRE-CE) decidiu que a pré-candidatura de Antônio Ilomar Vasconcelos Cruz foi prejudicada pela propaganda eleitoral antecipada realizada por ele mesmo. A decisão foi baseada na interpretação da lei e nos fatos apresentados no processo.

**Informações iniciais do processo**: O autor do recurso, Antônio Ilomar Vasconcelos Cruz, alegou que não havia pedido explícito de votos ou conduta que prejudicasse a igualdade entre os candidatos. Ele também argumentou que a propaganda eleitoral antecipada foi realizada antes do período autorizado pela legislação.

**Decisão do juiz no processo inicial**: O juiz decidiu que o ato de Antônio Ilomar Vasconcelos Cruz foi caracterizado como propaganda eleitoral antecipada e imposta a penalidade do artigo 36, § 3º, da Lei 9.504/97.

**Artigos de lei e fundamentos jurídicos relevantes**: O artigo 36 da Lei 9.504/97 proíbe a propaganda eleitoral antecipada, que é definida como a realização de atos de propaganda antes do período autorizado pela legislação.

**Quem recorreu e o que alegou**: Antônio Ilomar Vasconcelos Cruz recorreu à decisão do juiz, alegando que sua pré-candidatura foi prejudicada pela propaganda eleitoral antecipada.
    """

    resumo_bom = """# Tribunal Regional Eleitoral do Ceará

- Número do processo: 0600012-49.2024.6.06.0033
- Origem: CANINDÉ/CE
- Relator: DESEMBARGADOR ELEITORAL DANIEL CARVALHO CARNEIRO
- Recorrente: ANTÔNIO ILOMAR VASCONCELOS CRUZ
- Advogados(as): FRANCISCO JARDEL RODRIGUES DE SOUSA - OAB CE32787-A, LIDENIRA CAVALCANTE MENDONÇA VIEIRA - OAB CE0016731
- Recorrido: MINISTÉRIO PÚBLICO ELEITORAL
- Assunto: ELEIÇÕES 2024

# Relatório (O caso)

O Tribunal Regional Eleitoral do Ceará decidiu que o candidato Antônio Ilomar Vasconcelos Cruz realizou propaganda eleitoral antecipada em sua pré-candidatura ao cargo de prefeito do município de Canindé. A decisão se baseia na análise de documentos que mostram a utilização de um jingle de campanha com alusão explícita à candidatura do representado e pedido de votos.

A proteção do princípio da igualdade de oportunidades entre os candidatos foi vulnerada pela realização de propaganda eleitoral antecipada. A infraestrutura do evento e o local onde foi realizado são indicativos do objetivo de ampliar o alcance da mensagem de campanha, buscando influenciar a opinião dos eleitores e promover a imagem política antes do período oficial de campanha.

A sentença condenatória manteve a representação eleitoral por propaganda antecipada e condenou o recorrente ao pagamento de multa no valor de R$ 15.000 (quinze mil reais).

# Voto (Argumentação)

O recurso foi baseado na argumentação de que a pré-candidatura ocorreu dentro dos limites previstos no art. 36-A da Lei 9.504/97, aduzindo que o evento foi só e tão somente para discutir acerca da viabilidade da sua pré-candidatura.

O Tribunal Regional Eleitoral do Ceará não aceitou essa argumentação e manteve a sentença condenatória, concluindo que o evento em alusão foi realizado em ambiente aberto com a presença de um número expressivo de pessoas, houve pedido explícito de votos, bem como o uso de meio proscrito (outdoor), com ampla divulgação do evento nas redes sociais dos representados.

A decisão reforçou a importância do princípio da igualdade de oportunidades entre os candidatos e a necessidade de respeitar as normas eleitorais.

# Decisão (Resultado)

RECURSO NEGADO."""

    doc_path = "documentos/acordaos/0600012-49_REl_28052024_1.txt"

    texto_original = "\n".join([page.page_content for page in split_documents(doc_path, 0, 12)])

    embeddings_model = get_embeddings_model()
    resumo_ruim_embeddings = [embeddings_model.embed_query(resumo_impreciso)]
    resumo_bom_embeddings = [embeddings_model.embed_query(resumo_bom)]
    texto_original_embeddings = [embeddings_model.embed_query(texto_original)]


    print('Bom')
    print('Cosine similarity:', get_similarity_score(resumo_bom, texto_original, method='cosine'))
    # print('BERTScore:', score([resumo_bom], [texto_original], lang='pt'), '\n')
    print('BERTScore:', get_similarity_score(resumo_bom, texto_original, method='bertscore'), '\n')

    print('Ruim')
    print('Cosine similarity:', get_similarity_score(resumo_impreciso, texto_original, method='cosine'))
    # print('BERTScore:', score([resumo_impreciso], [texto_original], lang='pt'), '\n')
    print('BERTScore:', get_similarity_score(resumo_impreciso, texto_original, method='bertscore'))
