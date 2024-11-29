import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from cwi import cwi

def test_cwi():
    text = "O recurso foi impugnado pela Promotoria Eleitoral, que argumentou que o recorrente realizou propaganda irregular, com pedido explícito de votos, e que a infraestrutura do evento e o local onde foi realizado seriam indicativos do objetivo de ampliar o alcance da mensagem de sua pré-candidatura."
    cw = cwi(text, sorted=True)
    print(cw)

    text = "O juízo a quo entendeu que o evento realizado pelo recorrente foi uma propaganda eleitoral antecipada, pois houve pedido explícito de votos, uso de meio proibido (outdoor) e ampla divulgação do evento nas redes sociais."
    cw = cwi(text, sorted=True, threshold=10.0)
    print(cw)
