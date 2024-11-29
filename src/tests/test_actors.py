# import sys
# import os

# sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

# from ner import NER
# from segment_sentences import segment


# def test_actors():
#     text = """
# **Relatório**

# O recorrente Antônio Ilomar Vasconcelos Cruz impugna a sentença proferida pelo Juízo Eleitoral da 33ª Zona - Canindé/CE que julgou parcialmente procedente os pedidos da Representação Eleitoral por propaganda antecipada e condenou o recorrente ao pagamento de multa no valor de R$ 15.000,00.

# O juízo a quo concluiu que o evento realizado pelo recorrente foi uma propaganda eleitoral antecipada, pois houve pedido explícito de votos, uso de meio proscrito (outdoor) e ampla divulgação do evento nas redes sociais. Além disso, o princípio da igualdade de oportunidades entre os candidatos foi vulnerado.

# O recorrente alega que as condutas no lançamento de sua pré-candidatura ocorreram dentro dos limites previstos no art. 36-A da Lei 9.504/97 e que o evento foi realizado em ambiente fechado, sem pedido explícito de votos ou uso de meio proscrito.

# O prefeito de Canindé, Ilomar, realizou um evento de campanha que contou com estrutura profissional e divulgação nas redes sociais, o que sugere uma prática de propaganda eleitoral antecipada. O evento foi realizado antes do período autorizado pela legislação e teve como objetivo influenciar o eleitorado para que Ilomar fosse eleito prefeito municipal.

# A decisão do prefeito foi impugnada por um dos candidatos, que interpôs um recurso contra a decisão. O recurso questiona a igualdade de oportunidade e a paridade de armas entre os candidatos, argumentando que a prática de propaganda eleitoral antecipada feriu a igualdade de chances.

# O autor do recurso pediu a anulação da decisão do juízo a quo e a condenação do recorrente à multa. O réu se defendeu afirmando que as condutas do recorrente ocorreram dentro dos limites previstos no art. 36-A da Lei 9.504/97.

# O artigo 36-A da Lei 9.504/97 estabelece que a propaganda eleitoral antecipada é proib
# ida, sendo punida com multa. O princípio da igualdade de oportunidades entre os candidatos foi vulnerado pela prática de propaganda eleitoral antecipada.

# O recorrente Antônio Ilomar Vasconcelos Cruz impugna a sentença proferida pelo Juízo Eleitoral da 33ª Zona - Canindé/CE. O réu se defendeu afirmandoque as condutas do recorrente ocorreram dentro dos limites previstos no art. 36-A da Lei 9.504/97.

# A Procuradoria Regional Eleitoral apresentou parecer, manifestando-se pela manutenção da sentença e imposição da penalidade do art. 36, § 3º, da Lei 9.504/97, por entender que foi praticado ato típico decampanha eleitoral não abrangido no rol do art. 36-A da Lei 9.504/97.

# O recurso foi apresentado pelo recorrente Antônio Ilomar Vasconcelos Cruz, que reitera os argumentos apresentados em sede de contestação e cita decisão do TSE na Representação Eleitoral Nº 0600012 - 49.2024.6.06.0033.

# O recurso foi impugnado pela Promotoria Eleitoral, que argumentou que o recorrente realizou propaganda irregular,com pedido explícito de votos, e que a infraestrutura do evento e o local onde foi realizado seriam indicativos do objetivo de ampliar o alcance da mensagem de sua pré-candidatura.

# A Procuradoria Regional Eleitoral apresentou parecer, manifestando-se pela manutenção da sentença e imposição da penalidade do art. 36, § 3º, da Lei 9.504/97, por entender que foi praticado ato típico de campanha eleitoral não abrangido no rol do art. 36-A da Lei 9.504/97.

# O recurso foi apresentado pelo recorrente Antônio Ilomar Vasconcelos Cruz, que reitera os argumentos apresentados em sede de contestação e cita decisão do TSE na Representação Eleitoral Nº 0600012 - 49.2024.6.06.0033.

# O recurso foi impugnado pela Promotoria Eleitoral, que argumentou que o recorrente realizou propaganda irregular, com pedido explícito de votos, e que a infraestrutura do evento e o local onde foi realizado seriam indicativos do objetivo de ampliar o alcance da mensagem de sua pré-candidatura.

# A Procuradoria Regional Eleitoral apresentou parecer, manifestando-se pela manutenção da sentença e imposição da penalidade do art. 36, § 3º,da Lei 9.504/97, por entender que foi praticado ato típico de campanha eleitoral não abrangido no roldo art. 36-A da Lei 9.504/97.

# O recurso foi apresentado pelo recorrente Antônio Ilomar VasconcelosCruz, que reitera os argumentos apresentados em sede de contestação e cita decisão do TSE na Representação Eleitoral Nº 0600012 - 49.2024.6.06.0033.

# O recurso foi impugnado pela Promotoria Eleitoral, que argumentou que o recorrente realizou propaganda irregular, com pedido explícito de votos, e que a infraestrutura do evento e o local onde foi realizado seriam indicativos do objetivo de ampliar o alcance da mensagem de sua pré-candidatura.

# A Procuradoria Regional Eleitoral apresentou parecer, manifestando-se pela manutenção da sentença e imposição da penalidade do art. 36, § 3º, da Lei 9.504/97, por entender que foi praticado ato típico de campanha eleitoral não abrangido no rol do art. 36-A da Lei9.504/97.

# O recurso foi apresentado pelo recorrente Antônio Ilomar Vasconcelos Cruz, que reitera os argumentos apresentados em sede de contestação e cita decisão do TSE na Representação Eleitoral Nº 0600012 - 49
# """

#     sentences = segment(text)
#     ner = NER()
#     entities = ner.get_topics(text)

#     for entity in entities:
#         pass
