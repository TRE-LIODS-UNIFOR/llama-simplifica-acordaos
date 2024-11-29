import csv
from itertools import product
import time

from langchain.prompts import PromptTemplate
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain
from langchain_community.vectorstores import FAISS
from langchain_ollama import ChatOllama, OllamaEmbeddings

from semantic_similarity import get_similarity_score
from split_documents import split_documents


def collect(temp, top_p, top_k, chunk_size, chunk_overlap, context_size, prompt, doc_path, page_start=2, page_end=5):
    llm = ChatOllama(
        base_url='http://10.10.0.99:11434',
        model='llama3.2',
        temperature=temp,
        top_p=top_p,
        top_k=top_k,
        num_ctx=context_size,
    )
    embeddings_model = OllamaEmbeddings(
        base_url='http://10.10.0.99:11434',
        model='nomic-embed-text',
    )

    doc = split_documents(doc_path, page_start, page_end, chunk_size=chunk_size, chunk_overlap=chunk_overlap)

    vectorstore = FAISS.from_documents(documents=doc, embedding=embeddings_model,)
    retriever = vectorstore.as_retriever()

    document_chain = create_stuff_documents_chain(llm, prompt)
    retrieval_chain = create_retrieval_chain(retriever, document_chain)

    input_dict = {
        'input': 'Comece.'
    }
    res = retrieval_chain.invoke(input_dict)

    del(vectorstore)
    return res['answer']

def save_results(results, path):
    print(results[:5])
    with open(path, 'w') as f:
        w = csv.DictWriter(f, results[0].keys())
        w.writeheader()
        for result in results:
            w.writerow(result)

def new_result(temp, top_p, top_k, chunk_size, chunk_overlap, context_size, response, bertscore, cosine):
    return {
        'temperature': temp,
        'top_p': top_p,
        'top_k': top_k,
        'context_size': context_size,
        'chunk_size': chunk_size,
        'chunk_overlap': chunk_overlap,
        'response': response,
        'bertscore': bertscore,
        'cosine': cosine,
    }

if __name__ == '__main__':
    temperatures = [i for i in range(0, 101, 20)]
    top_p = [i/10 for i in range(0, 31, 10)]
    top_k = [i for i in range(1, 4, 2)]
    context_size = [i for i in range(2048, 8193, 1024)]
    chunk_size = [i for i in range(0, 2561, 256)]
    chunk_overlap = [i for i in range(0, 256, 64)]

    chunk_size[0] = 32

    results = []

    prompt = PromptTemplate.from_template(
        """
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
    )

    comb = list(product(temperatures, top_p, top_k, context_size, chunk_size, chunk_overlap))

    print('Número de testes: ', len(comb))

    original_doc = "\n".join(open('documentos/acordaos/0600012-49_REl_28052024_1.txt', 'r').read().split("\n-----\n"))

    i = 0

    for c in comb:
        i += 1
        print(f'Teste {i}/{len(comb)}')
        try:
            print(*c)
            response = collect(*c, prompt=prompt, doc_path='documentos/acordaos/0600012-49_REl_28052024_1.txt', page_start=2, page_end=5)
            results.append(new_result(*c, response, get_similarity_score(response, original_doc, method='bertscore'), get_similarity_score(response, original_doc, method='cosine')))
            time.sleep(60)
        except:
            save_results(results, 'data_collection_results.csv')

    save_results(results, 'data_collection_results.csv')
    print(results)
