import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from postprocessing import postprocess
from stuff import most_similar
from summarize import summarize_section
from semantic_similarity import get_similarity_score

def test_summarize_section():
    document = """Alice, uma jovem curiosa e imaginativa, está sentada sob uma árvore quando avista um coelho branco muito peculiar com um relógio de bolso. Fascinada, ela decide segui-lo e acaba caindo em uma toca que a leva a um mundo mágico e surreal. Nesse novo lugar, ela se depara com portas mágicas, poções que a fazem encolher e crescer, e um ambiente completamente diferente do que ela conhece. A jornada de Alice começa com sua curiosidade, mas rapidamente se transforma em uma aventura cheia de desafios e encontros inesperados.

Durante sua exploração, Alice encontra personagens icônicos e excêntricos. O primeiro deles é a Lagarta Azul, que a ajuda a entender como controlar seu tamanho comendo pedaços de um cogumelo mágico. Em seguida, ela participa de um chá muito estranho com o Chapeleiro Maluco e a Lebre de Março, onde tudo parece desprovido de lógica. Nesse momento, Alice percebe que as regras do País das Maravilhas são completamente absurdas, mas ainda assim, ela tenta se adaptar e aprender com os eventos que encontra.

Mais adiante, Alice encontra a Rainha de Copas, uma figura autoritária e temperamental que vive ordenando a execução de qualquer um que a desagrade com a frase "Cortem-lhe a cabeça!" Apesar da aparência ameaçadora, Alice percebe que a Rainha é mais teatral do que perigosa. Durante um jogo de croqué caótico, a jovem é novamente confrontada com a loucura do lugar e tenta manter a calma em meio a tantas situações absurdas.

A jornada de Alice culmina em um julgamento surreal, onde ela testemunha contra o Valete de Copas, acusado de roubar tortas. Nesse momento, o caos atinge seu ápice, e Alice, finalmente, começa a questionar a lógica do País das Maravilhas e a autoridade das figuras que encontra. Ela cresce em tamanho, literalmente e metaforicamente, ao confrontar a Rainha de Copas e declarar que eles não passam de cartas de baralho.

No final, Alice acorda e descobre que tudo não passou de um sonho. De volta ao mundo real, ela reflete sobre as lições aprendidas e a estranheza de sua experiência no País das Maravilhas. A história termina com uma nota de curiosidade e imaginação, destacando o poder dos sonhos e da fantasia na vida de uma criança."""
    result = summarize_section(document, "Resuma a seguinte história em até três parágrafos.\n\n{context}", verbose=True)

    print(result)

    assert len(result) == 2
    assert get_similarity_score(result[0], document) >= 0.75

def test_summarize_section_pass():
    """
    Apenas precisa completar sem erros. O sentido do resultado aqui não importa.
    """
    document = """ABCDEFG"""
    result = summarize_section(document, "Resuma o seguinte texto em até três caracteres.\n\n{context}", verbose=True)

    print(result)

    assert len(result) == 2

def test_postprocess_most_similar():
    document = """Alice, uma jovem curiosa e imaginativa, está sentada sob uma árvore quando avista um coelho branco muito peculiar com um relógio de bolso. Fascinada, ela decide segui-lo e acaba caindo em uma toca que a leva a um mundo mágico e surreal. Nesse novo lugar, ela se depara com portas mágicas, poções que a fazem encolher e crescer, e um ambiente completamente diferente do que ela conhece. A jornada de Alice começa com sua curiosidade, mas rapidamente se transforma em uma aventura cheia de desafios e encontros inesperados.

Durante sua exploração, Alice encontra personagens icônicos e excêntricos. O primeiro deles é a Lagarta Azul, que a ajuda a entender como controlar seu tamanho comendo pedaços de um cogumelo mágico. Em seguida, ela participa de um chá muito estranho com o Chapeleiro Maluco e a Lebre de Março, onde tudo parece desprovido de lógica. Nesse momento, Alice percebe que as regras do País das Maravilhas são completamente absurdas, mas ainda assim, ela tenta se adaptar e aprender com os eventos que encontra.

Mais adiante, Alice encontra a Rainha de Copas, uma figura autoritária e temperamental que vive ordenando a execução de qualquer um que a desagrade com a frase "Cortem-lhe a cabeça!" Apesar da aparência ameaçadora, Alice percebe que a Rainha é mais teatral do que perigosa. Durante um jogo de croqué caótico, a jovem é novamente confrontada com a loucura do lugar e tenta manter a calma em meio a tantas situações absurdas.

A jornada de Alice culmina em um julgamento surreal, onde ela testemunha contra o Valete de Copas, acusado de roubar tortas. Nesse momento, o caos atinge seu ápice, e Alice, finalmente, começa a questionar a lógica do País das Maravilhas e a autoridade das figuras que encontra. Ela cresce em tamanho, literalmente e metaforicamente, ao confrontar a Rainha de Copas e declarar que eles não passam de cartas de baralho.

No final, Alice acorda e descobre que tudo não passou de um sonho. De volta ao mundo real, ela reflete sobre as lições aprendidas e a estranheza de sua experiência no País das Maravilhas. A história termina com uma nota de curiosidade e imaginação, destacando o poder dos sonhos e da fantasia na vida de uma criança."""
    responses = [document, document, document]
    scores = [0.75, 0.8, 0.9]
    best, score = most_similar(responses, scores)

    print(best, score)

    assert best == document
    assert score == 0.9

    try:
        post = postprocess(best, best)
        print(post)
        assert True
    except Exception as e:
        assert False, e

