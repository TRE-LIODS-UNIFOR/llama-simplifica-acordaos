import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import bert_score
from semantic_similarity import get_similarity_score, pair_similar_chunks
from split_documents import split_text

from config import Config

def test_load_model():
    bert_score.BERTScorer(lang='pt', model_type=Config.BERTSCORE_MODEL, num_layers=Config.BERTSCORE_MODEL_N_LAYERS)

def test_chunks():
    chunks = bert_score.score(['a', 'b', 'c'], ['d'] * 3, lang='pt', model_type=Config.BERTSCORE_MODEL, num_layers=Config.BERTSCORE_MODEL_N_LAYERS)
    assert len(chunks) == 3
    assert len(chunks[0]) == 3

def test_similarity():
    A = """O caso envolveu Antônio Ilomar Vasconcelos Cruz, que realizou propaganda eleitoral antecipada ao lançar sua pré-candidatura ao cargo de prefeito.Ele foi condenado a pagar multa por violar as regras eleitorais.

Antônio Ilomar Vasconcelos Cruz interpôs recurso contra a sentença, argumentando que suas condutas estavam dentro dos limites previstos na Lei 9.504/97. O recorrente alegou que o evento em alusão foi realizado em ambiente fechado e que não houve pedido explícito de votos.

A Promotoria Eleitoral pugnou pelo não provimento do recurso, aduzindo que o recorrente realizou propaganda irregular com pedido explícito de votos. A Procuradoria Regional Eleitoral apresentou parecer manifestando-se pela manutenção da sentença e imposição da penalidade do art. 36, § 3º, da Lei 9.504/97.

O juiz julgou parcialmente procedente a Representação Eleitoral."""
    B = """Antônio Ilomar Vasconcelos Cruz interpôs recurso contra a sentença, argumentando que suas condutas estavam dentro dos limites previstos na Lei 9.504/97. O recorrente alegou que o evento em alusão foi realizado em ambiente fechado e que não houve pedido explícito de votos.

A Promotoria Eleitoral pugnou pelo não provimento do recurso, aduzindo que o recorrente realizou propaganda irregular com pedido explícito de votos. A Procuradoria Regional Eleitoral apresentou parecer manifestando-se pela manutenção da sentença e imposição da penalidade do art. 36, § 3º, da Lei 9.504/97.

O juiz julgou parcialmente procedente a Representação Eleitoral.

O caso envolveu Antônio Ilomar Vasconcelos Cruz, que realizou propaganda eleitoral antecipada ao lançar sua pré-candidatura ao cargo de prefeito.Ele foi condenado a pagar multa por violar as regras eleitorais."""
    score = get_similarity_score(A, B, method='bertscore')

    print(score)

def test_self_similarity():
    text = """O caso envolveu Antônio Ilomar Vasconcelos Cruz, que realizou propaganda eleitoral antecipada ao lançar sua pré-candidatura ao cargo de prefeito.Ele foi condenado a pagar multa por violar as regras eleitorais.

Antônio Ilomar Vasconcelos Cruz interpôs recurso contra a sentença, argumentando que suas condutas estavam dentro dos limites previstos na Lei 9.504/97. O recorrente alegou que o evento em alusão foi realizado em ambiente fechado e que não houve pedido explícito de votos.

A Promotoria Eleitoral pugnou pelo não provimento do recurso, aduzindo que o recorrente realizou propaganda irregular com pedido explícito de votos. A Procuradoria Regional Eleitoral apresentou parecer manifestando-se pela manutenção da sentença e imposição da penalidade do art. 36, § 3º, da Lei 9.504/97.

O juiz julgou parcialmente procedente a Representação Eleitoral."""
    score = get_similarity_score(text, text, method='bertscore')

    print(score)

    assert score

def test_long_input():
    text = """O caso envolveu Antônio Ilomar Vasconcelos Cruz, que realizou propaganda eleitoral antecipada ao lançar sua pré-candidatura ao cargo de prefeito.Ele foi condenado a pagar multa por violar as regras eleitorais.

Antônio Ilomar Vasconcelos Cruz interpôs recurso contra a sentença, argumentando que suas condutas estavam dentro dos limites previstos na Lei 9.504/97. O recorrente alegou que o evento em alusão foi realizado em ambiente fechado e que não houve pedido explícito de votos.

A Promotoria Eleitoral pugnou pelo não provimento do recurso, aduzindo que o recorrente realizou propaganda irregular com pedido explícito de votos. A Procuradoria Regional Eleitoral apresentou parecer manifestando-se pela manutenção da sentença e imposição da penalidade do art. 36, § 3º, da Lei 9.504/97.

O juiz julgou parcialmente procedente a Representação Eleitoral."""

    score = get_similarity_score(text, text, method='bertscore')

    print(score)

    assert score

def test_pair_similar_chunks():
    text = """Sofia encontrou um relógio de areia no sótão, mas algo estava errado. Quando o virou, o tempo parou. Pessoas congelaram, pássaros flutuaram imóveis. Testando o artefato, percebeu que podia moldar a realidade enquanto a areia caía. Curiosa, corrigiu erros da vida: desculpou-se com um amigo, salvou um vaso quebrado. Porém, ao virar o relógio novamente, tudo mudou. As pessoas não a reconheciam mais; o mundo não era o mesmo. Sofia entendeu: o tempo cobra caro por ser manipulado.

    João adorava a chuva. Certo dia, ao abrigar-se numa velha cabana, encontrou um estranho molhado. "Você pediu por mim", disse o homem, estendendo um mapa misterioso. Antes que João perguntasse algo, ele desapareceu. No mapa, um "X" marcava o quintal de sua infância. Cavou lá e encontrou um baú com cartas escritas por sua avó, cheia de segredos e conselhos sobre a vida. Percebeu que o visitante era mais que um estranho; era a resposta às perguntas que nunca ousou fazer."""
    mixed_text = """Sofia encontrou um relógio de areia no sótão, mas algo estava errado. Quando o virou, o tempo parou. Pessoas congelaram, pássaros flutuaram imóveis. Testando o artefato, percebeu que podia moldar a realidade enquanto a areia caía.

    João adorava a chuva. Certo dia, ao abrigar-se numa velha cabana, encontrou um estranho molhado. "Você pediu por mim", disse o homem, estendendo um mapa misterioso. Antes que João perguntasse algo, ele desapareceu. No mapa, um "X" marcava o quintal de sua infância.

    Curiosa, corrigiu erros da vida: desculpou-se com um amigo, salvou um vaso quebrado. Porém, ao virar o relógio novamente, tudo mudou. As pessoas não a reconheciam mais; o mundo não era o mesmo. Sofia entendeu: o tempo cobra caro por ser manipulado.

    Cavou lá e encontrou um baú com cartas escritas por sua avó, cheia de segredos e conselhos sobre a vida. Percebeu que o visitante era mais que um estranho; era a resposta às perguntas que nunca ousou fazer."""

    candidate_chunks = [page.page_content for page in split_text(text, chunk_size=512, chunk_overlap=32)]
    reference_chunks = [page.page_content for page in split_text(mixed_text, chunk_size=512, chunk_overlap=32)]

    alignment = pair_similar_chunks(candidate_chunks, reference_chunks)

    print(alignment)

    assert alignment
