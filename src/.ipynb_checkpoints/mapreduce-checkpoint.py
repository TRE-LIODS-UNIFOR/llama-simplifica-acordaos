from pprint import pprint
from threading import Lock
from langchain.chains import (
    LLMChain,
    StuffDocumentsChain,
    ReduceDocumentsChain,
    MapReduceDocumentsChain
)
from langchain.prompts import ChatPromptTemplate, PromptTemplate
from langchain_ollama import ChatOllama
from langchain_community.document_loaders import TextLoader, PyMuPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

from config import Config
from database import PromptDB
import history
from llms import get_llama

def mapreduce(docs, host, results, key, lock):
    llm: ChatOllama = get_llama(host=host)

    # Map
    map_template = """
        ### SISTEMA
        Responda com um resumo fiel de até três parágrafos dos documentos no contexto.
        Responda apenas com o resumo, e mais nada.

        ### CONTEXTO
        {docs}.

        ### RESPOSTA:
    """
    map_prompt = ChatPromptTemplate([("human", map_template)])
    map_chain = LLMChain(llm=llm, prompt=map_prompt)


    # Reduce
    reduce_template = """
        ### SISTEMA
        Você receberá uma sequência de resumos, combine-os mantendo a ordem e a coerência, produzindo um novo resumo.
        Responda apenas com o novo resumo, e mais nada.

        ### CONTEXTO
        {docs}

        ### Resposta:
    """
    reduce_prompt = ChatPromptTemplate([("human", reduce_template)])
    reduce_chain = LLMChain(llm=llm, prompt=reduce_prompt)


    # Takes a list of documents, combines them into a single string, and passes this to an LLMChain
    combine_documents_chain = StuffDocumentsChain(
        llm_chain=reduce_chain, document_variable_name="docs"
    )

    collapse_prompt = PromptTemplate.from_template(
        """
        Colapse o seguinte texto: {context}
        """
    )
    collapse_documents_chain = StuffDocumentsChain(
        llm_chain=LLMChain(llm=llm, prompt=collapse_prompt)
    )

    # Combines and iteratively reduces the mapped documents
    reduce_documents_chain = ReduceDocumentsChain(
        # This is final chain that is called.
        combine_documents_chain=combine_documents_chain,
        # If documents exceed context for `StuffDocumentsChain`
        collapse_documents_chain=collapse_documents_chain,
    )

    # Combining documents by mapping a chain over them, then combining results
    map_reduce_chain = MapReduceDocumentsChain(
        # Map chain
        llm_chain=map_chain,
        # Reduce chain
        reduce_documents_chain=reduce_documents_chain,
        # The variable name in the llm_chain to put the documents in
        document_variable_name="docs",
        # Return the results of the map steps in the output
        return_intermediate_steps=True,
    )

    format_template = PromptTemplate.from_template(
        '''
        Separe o texto em dois a três parágrafos, mantendo a ordem, o tom e a coerência:

        ```
        {context}
        ```

        (responda apenas com o texto separado, e nada mais)
        '''
    )
    format_chain = LLMChain(llm=llm, prompt=format_template)


    print(f"calling {host}")

    result = map_reduce_chain.invoke(docs)
    pprint(result)
    result = format_chain.invoke({'context': result['output_text']})

    print('\a')
    print(f"\nHOST {host}:\n")
    pprint(result)

    # simplificar_prompt = PromptTemplate.from_template(
    #     '''
    #     Simplifique a linguagem usada no texto. Para isso, siga as seguintes diretrizes
    #     '''
    # )

    with lock:
        results[key]['response'] = result['text']
        results[key]['prompt'] = [
            str(map_prompt),
            str(reduce_prompt),
            str(format_template),
        ]

    # history.save('mapreduce', '', result['text'])
    return result

def split_documents(file_path, page_start=None, page_end=None):
    loader = PyMuPDFLoader(file_path=file_path)
    doc = loader.load()[page_start:page_end]

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=Config.SPLITTER_CHUNK_SIZE,
        chunk_overlap=Config.SPLITTER_CHUNK_OVERLAP
    )
    docs = text_splitter.split_documents(doc)
    return docs

def main():
    import sys
    from threading import Thread

    file_path = sys.argv[1]
    relatorio_start = int(sys.argv[2])
    relatorio_end = int(sys.argv[3])
    voto_start = int(sys.argv[4])
    voto_end = int(sys.argv[5])

    doc_relatorio, doc_voto = None, None

    if relatorio_start != -1 and relatorio_end != -1:
        doc_relatorio = split_documents(file_path, relatorio_start, relatorio_end)
    if voto_start != -1 and voto_end != -1:
        doc_voto = split_documents(file_path, voto_start, voto_end)

    results = {
        'relatorio': {},
        'voto': {},
    }

    lock = Lock()

    threads = set([
        Thread(target=mapreduce, args=(doc_relatorio, 0, results, 'relatorio', lock)) if doc_relatorio else Thread(target=lambda x: x, args=(0,)),
        Thread(target=mapreduce, args=(doc_voto, 0, results, 'voto', lock)) if doc_voto else Thread(target=lambda x: x, args=(0,)),
    ])

    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    pprint(results)
    history.save('acordao_resumido', "\n\n".join([results[i]['prompt'] for i in results]), "\n\n".join([results[i]['response'] for i in results]))


if __name__ == '__main__':
    main()
