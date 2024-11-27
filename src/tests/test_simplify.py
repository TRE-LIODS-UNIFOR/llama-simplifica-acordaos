from pprint import pprint
import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from segment_sentences import segment
from simplify import bert_sinonimos, build_from_simplified, collapse, long_period, personalize, simplify, simplify_segments
from semantic_similarity import get_similarity_score

relatorio = """O recorrente Antônio Ilomar Vasconcelos Cruz impugna a sentença proferida pelo Juízo Eleitoral da 33ª Zona - Canindé/CE que julgou parcialmente procedente os pedidos da Representação Eleitoral por propaganda antecipada e condenou o recorrente ao pagamento de multa no valor de R$ 15.000,00.

O juízo a quo concluiu que o evento realizado pelo recorrente foi uma propaganda eleitoral antecipada, pois houve pedido explícito de votos, uso de meio proscrito (outdoor) e ampla divulgação do evento nas redes sociais. Além disso, o princípio da igualdade de oportunidades entre os candidatos foi vulnerado.

O recorrente alega que as condutas no lançamento de sua pré-candidatura ocorreram dentro dos limites previstos no art. 36-A da Lei 9.504/97 e que o evento foi realizado em ambiente fechado, sem pedido explícito de votos ou uso de meio proscrito.

O prefeito de Canindé, Ilomar, realizou um evento de campanha que contou com estrutura profissional e divulgação nas redes sociais, o que sugere uma prática de propaganda eleitoral antecipada. O evento foi realizado antes do período autorizado pela legislação e teve como objetivo influenciar o eleitorado para que Ilomar fosse eleito prefeito municipal.

A decisão do prefeito foi impugnada por um dos candidatos, que interpôs um recurso contra a decisão. O recurso questiona a igualdade de oportunidade e a paridade de armas entre os candidatos, argumentando que a prática de propaganda eleitoral antecipada feriu a igualdade de chances.

O autor do recurso pediu a anulação da decisão do juízo a quo e a condenação do recorrente à multa. O réu se defendeu afirmando que as condutas do recorrente ocorreram dentro dos limites previstos no art. 36-A da Lei 9.504/97.

O artigo 36-A da Lei 9.504/97 estabelece que a propaganda eleitoral antecipada é proibida, sendo punida com multa. O princípio da igualdade de oportunidades entre os candidatos foi vulnerado pela prática de propaganda eleitoral antecipada.

O recorrente Antônio Ilomar Vasconcelos Cruz impugna a sentença proferida pelo Juízo Eleitoral da 33ª Zona - Canindé/CE. O réu se defendeu afirmandoque as condutas do recorrente ocorreram dentro dos limites previstos no art. 36-A da Lei 9.504/97.

A Procuradoria Regional Eleitoral apresentou parecer, manifestando-se pela manutenção da sentença e imposição da penalidade do art. 36, § 3º, da Lei 9.504/97, por entender que foi praticado ato típico decampanha eleitoral não abrangido no rol do art. 36-A da Lei 9.504/97.

O recurso foi apresentado pelo recorrente Antônio Ilomar Vasconcelos Cruz, que reitera os argumentos apresentados em sede de contestação e cita decisão do TSE na Representação Eleitoral Nº 0600012 - 49.2024.6.06.0033.

O recurso foi impugnado pela Promotoria Eleitoral, que argumentou que o recorrente realizou propaganda irregular,com pedido explícito de votos, e que a infraestrutura do evento e o local onde foi realizado seriam indicativos do objetivo de ampliar o alcance da mensagem de sua pré-candidatura.

A Procuradoria Regional Eleitoral apresentou parecer, manifestando-se pela manutenção da sentença e imposição da penalidade do art. 36, § 3º, da Lei 9.504/97, por entender que foi praticado ato típico de campanha eleitoral não abrangido no rol do art. 36-A da Lei 9.504/97.

O recurso foi apresentado pelo recorrente Antônio Ilomar Vasconcelos Cruz, que reitera os argumentos apresentados em sede de contestação e cita decisão do TSE na Representação Eleitoral Nº 0600012 - 49.2024.6.06.0033.

O recurso foi impugnado pela Promotoria Eleitoral, que argumentou que o recorrente realizou propaganda irregular, com pedido explícito de votos, e que a infraestrutura do evento e o local onde foi realizado seriam indicativos do objetivo de ampliar o alcance da mensagem de sua pré-candidatura.

A Procuradoria Regional Eleitoral apresentou parecer, manifestando-se pela manutenção da sentença e imposição da penalidade do art. 36, § 3º,da Lei 9.504/97, por entender que foi praticado ato típico de campanha eleitoral não abrangido no roldo art. 36-A da Lei 9.504/97.

O recurso foi apresentado pelo recorrente Antônio Ilomar VasconcelosCruz, que reitera os argumentos apresentados em sede de contestação e cita decisão do TSE na Representação Eleitoral Nº 0600012 - 49.2024.6.06.0033.

O recurso foi impugnado pela Promotoria Eleitoral, que argumentou que o recorrente realizou propaganda irregular, com pedido explícito de votos, e que a infraestrutura do evento e o local onde foi realizado seriam indicativos do objetivo de ampliar o alcance da mensagem de sua pré-candidatura.

A Procuradoria Regional Eleitoral apresentou parecer, manifestando-se pela manutenção da sentença e imposição da penalidade do art. 36, § 3º, da Lei 9.504/97, por entender que foi praticado ato típico de campanha eleitoral não abrangido no rol do art. 36-A da Lei9.504/97.

O recurso foi apresentado pelo recorrente Antônio Ilomar Vasconcelos Cruz, que reitera os argumentos apresentados em sede de contestação e cita decisão do TSE na Representação Eleitoral Nº 0600012 - 49
"""

def test_simplify():
    text = relatorio
    # segments = segment(text)
    # _, scores = simplify_segments(segments)
    simplified, score, ratio = simplify(text)

    print("Score:", score, "Ratio:", ratio)
    print("Result:", simplified)

    assert score >= 0.8

def test_simplify_small():
    text = """
O recorrente Antônio Ilomar Vasconcelos Cruz impugna a sentença proferida pelo Juízo Eleitoral da 33ª Zona - Canindé/CE que julgou parcialmente procedente os pedidos da Representação Eleitoral por propaganda antecipada e condenou o recorrente ao pagamento de multa no valor de R$ 15.000,00.

O juízo a quo concluiu que o evento realizado pelo recorrente foi uma propaganda eleitoral antecipada, pois houve pedido explícito de votos, uso de meio proscrito (outdoor) e ampla divulgação do evento nas redes sociais. Além disso, o princípio da igualdade de oportunidades entre os candidatos foi vulnerado.

O recorrente alega que as condutas no lançamento de sua pré-candidatura ocorreram dentro dos limites previstos no art. 36-A da Lei 9.504/97 e que o evento foi realizado em ambiente fechado, sem pedido explícito de votos ou uso de meio proscrito.

A Procuradoria Regional Eleitoral apresentou parecer, manifestando-se pela manutenção da sentença e imposição da penalidade do art. 36, § 3º, da Lei 9.504/97, por entender que foi praticado ato típico de campanha eleitoral não abrangido no rol do art. 36-A da Lei9.504/97.

O recurso foi apresentado pelo recorrente Antônio Ilomar Vasconcelos Cruz, que reitera os argumentos apresentados em sede de contestação e cita decisão do TSE na Representação Eleitoral Nº 0600012 - 49
"""
    segments = segment(text)
    sentences, scores = simplify_segments(segments)

    pprint(sentences)

    avg = sum(scores) / len(scores)
    assert avg >= 0.8

def test_simplify_ratio():
    text = relatorio
    segments = segment(text)
    print("Number of segments:", len(segments))
    simplified_segments, scores = simplify_segments(segments)
    result, score, ratio = build_from_simplified(simplified_segments, scores, segments, 0.75)

    simplified_text = '\n\n'.join(result)

    print(ratio, score)
    print(simplified_text)

    assert ratio >= 0.5
    assert score >= 0.8
    assert len(result) == len(segments)

def test_simplify_ratio_small():
    text = """
O recorrente Antônio Ilomar Vasconcelos Cruz impugna a sentença proferida pelo Juízo Eleitoral da 33ª Zona - Canindé/CE que julgou parcialmente procedente os pedidos da Representação Eleitoral por propaganda antecipada e condenou o recorrente ao pagamento de multa no valor de R$ 15.000,00.

O juízo a quo concluiu que o evento realizado pelo recorrente foi uma propaganda eleitoral antecipada, pois houve pedido explícito de votos, uso de meio proscrito (outdoor) e ampla divulgação do evento nas redes sociais. Além disso, o princípio da igualdade de oportunidades entre os candidatos foi vulnerado.

O recorrente alega que as condutas no lançamento de sua pré-candidatura ocorreram dentro dos limites previstos no art. 36-A da Lei 9.504/97 e que o evento foi realizado em ambiente fechado, sem pedido explícito de votos ou uso de meio proscrito.

A Procuradoria Regional Eleitoral apresentou parecer, manifestando-se pela manutenção da sentença e imposição da penalidade do art. 36, § 3º, da Lei 9.504/97, por entender que foi praticado ato típico de campanha eleitoral não abrangido no rol do art. 36-A da Lei9.504/97.

O recurso foi apresentado pelo recorrente Antônio Ilomar Vasconcelos Cruz, que reitera os argumentos apresentados em sede de contestação e cita decisão do TSE na Representação Eleitoral Nº 0600012 - 49
"""
    segments = segment(text)
    simplified_segments, scores = simplify_segments(segments)
    result, score, ratio, overall_score, simplified_text = build_from_simplified(simplified_segments, scores, segments, 0.75)

    print("Ratio:", ratio, "Average segment score:", score, "Overall score:", overall_score)
    print(simplified_text)

    assert ratio >= 0.5
    assert score >= 0.8
    assert overall_score >= 0.8
    assert len(result) == len(segments)

def test_collapsed():
    original = relatorio
    simplified = """
O recorrente Antônio Ilomar Vasconcelos Cruz contesta a decisão do Juízo Eleitoral da 33ª Zona - Canindé/CE que julgou parcialmente procedente os pedidos da Representação Eleitoral por propaganda antecipada e condenou o recorrente ao pagamento de multa no valor de R$ 15.000,00. O juízo a quo entendeu que o evento realizado pelo recorrente foi uma propaganda eleitoral antecipada, pois houve pedido explícito de votos, uso de meio proibido (outdoor) e ampla divulgação do evento nas redes sociais. O princípio da igualdade de oportunidades entre os candidatos não foi respeitado. O recorrente afirma que suas condutas durante a pré-candidatura ocorreram dentro dos parâmetros legais e que o evento foi realizado em um ambiente fechado, sem solicitar votos explícitos ou usar meios proibidos. A Procuradoria Regional Eleitoral apresentou parecer, defendendo a manutenção da sentença e a imposição de penalidade por entender que houve um ato típico de campanha eleitoral não abrangido pela lei. Antônio Ilomar Vasconcelos Cruz, recorrente, reapresenta seus argumentos iniciais e faz referência a uma decisão específica do Tribunal Superior Eleitoral (TSE) na Representação Eleitoral. Nº 0600012 - 49.
"""
    collapsed = collapse(simplified)
    scores = [get_similarity_score(c, original, 'bertscore') for c in collapsed]

    print(scores)
    pprint(collapsed)

    assert any([score >= 0.8 for score in scores])

def test_personal():
    text = "O juízo a quo entendeu que o evento realizado pelo recorrente foi uma propaganda eleitoral antecipada, pois houve pedido explícito de votos, uso de meio proibido (outdoor) e ampla divulgação do evento nas redes sociais."
    personalize(text)
    text = "O recurso foi impugnado pela Promotoria Eleitoral, que argumentou que o recorrente realizou propaganda irregular, com pedido explícito de votos, e que a infraestrutura do evento e o local onde foi realizado seriam indicativos do objetivo de ampliar o alcance da mensagem de sua pré-candidatura."
    personalize(text)
    # text = "O juízo a quo entendeu que o evento realizado pelo recorrente foi uma propaganda eleitoral antecipada, pois houve pedido explícito de votos, uso de meio proibido (outdoor) e ampla divulgação do evento nas redes sociais."
    # personalize(text)

def test_sinonimos_bert():
    text = "O recurso foi impugnado pela Promotoria Eleitoral, que argumentou que o recorrente realizou propaganda irregular, com pedido explícito de votos, e que a infraestrutura do evento e o local onde foi realizado seriam indicativos do objetivo de ampliar o alcance da mensagem de sua pré-candidatura."
    sinonimos = bert_sinonimos('impugnado', text)
    pprint(list(sinonimos))

def test_long_period():
    text = "O recurso foi impugnado pela Promotoria Eleitoral, que argumentou que o recorrente realizou propaganda irregular, com pedido explícito de votos, e que a infraestrutura do evento e o local onde foi realizado seriam indicativos do objetivo de ampliar o alcance da mensagem de sua pré-candidatura."
    versions, scores = long_period(text)
    pprint(versions)
    pprint(scores)
