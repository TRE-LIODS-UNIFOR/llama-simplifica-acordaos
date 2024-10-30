import os
from pathlib import Path
import threading
from langchain.prompts import PromptTemplate
from postprocessing import postprocess
from stuff import stuff
from split_documents import split_documents


def single_pass(doc_path, start_page, end_page, prompt, base_url, lock, results, i, verbose=False):
    start_page = 2
    end_page = 5
    doc = split_documents(doc_path, start_page, end_page)

    prompt_template = PromptTemplate.from_template(prompt)

    process = stuff(doc, 0, {'relatorio': {}}, 'relatorio', prompt_template, base_url, verbose=verbose)
    post = postprocess(process, doc_path, start_page, end_page, base_url=base_url)

    print(f"{i} finished")

    with lock:
        results[i] = post

    return post

def main():
    doc_path = "documentos/acordaos/0600012-49_REl_28052024_1.txt"
    start_page = 2
    end_page = 5
    prompt = """
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

    print(doc_path)

    # out_dir = Path(doc_path).parent / Path(doc_path).stem
    out_dir = Path("documentos/acordaos/0600012-49_REl_28052024_1")
    # os.makedirs(out_dir, exist_ok=True)

    threads = set()
    lock = threading.Lock()

    results = {}

    hosts = ["http://10.10.0.99:11434", "http://10.10.0.98:11434", "http://10.10.0.95:11434"]

    for i in range(10):
        threads.add(threading.Thread(target=single_pass, args=(doc_path, start_page, end_page, prompt, hosts[i % 3], lock, results, i)))

    for thread in threads:
        thread.start()

    for thread in threads:
        thread.join()

    for i in results:
        res = results[i]
        with open(out_dir / f"resumo_{i}.txt", "w") as f:
            f.write(res)


if __name__ == '__main__':
    main()
