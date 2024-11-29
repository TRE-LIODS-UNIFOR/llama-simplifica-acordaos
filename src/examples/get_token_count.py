from langchain_ollama import ChatOllama

llm = ChatOllama(
    base_url='http://10.10.0.99:11434/',
    model='llama3.2',
    context_size=1024
)

print(llm.get_num_tokens("""Na peça contestatória (Id 19574900), o Representado, ora recorrente, alegou que as condutas no lançamento
de sua pré-candidatura ocorreram dentro dos limites previstos no art. 36-A da Lei 9.504/97, aduzindo que o
ocorrera “foi só e tão somente só, que o Representado, no genuíno exercício do direito previsto em lei para a
realização de atos de pré-campanha, respeitados os limites impostos pelo art. 36-A da Lei nº 9.504/97, sem
qualquer pedido de votos, participou de evento no qual os correligionários de sua agremiação partidária
puderam, de forma legítima, em ambiente fechado, discutir acerca da viabilidade da sua pré-candidatura, bem
como divulgar ideias, objetos e propostas partidárias.”"""))
