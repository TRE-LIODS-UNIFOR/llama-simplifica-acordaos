from pprint import pprint
from langchain.chains import (
    LLMChain,
    StuffDocumentsChain,
    ReduceDocumentsChain,
    MapReduceDocumentsChain
)
from langchain.prompts import ChatPromptTemplate, PromptTemplate
from langchain_ollama import ChatOllama
from langchain_core.documents import Document

from call_llms import call_llms
from config import Config
from llms import get_llama
from prompts.prompts import SimplePrompt

class Prompts:
    MAP = """
Responda com um resumo fiel de até três parágrafos do seguinte texto (Responda apenas com o resumo, e mais nada):

{context}
"""
    REDUCE = """
Combine os seguintes resumos de diferentes partes do mesmo texto, mantendo a ordem dos fatos e a coerência, produzindo um novo resumo (Responda apenas com o novo resumo, e mais nada):

{context}
"""

def mapreduce(docs, host=0, results=None, key=None, lock=None):
    llm: ChatOllama = get_llama(host=host, log_callbacks=True,)

    # Map
    map_template = """
### SISTEMA
Responda com um resumo fiel de até três parágrafos dos documentos no contexto.
Responda apenas com o resumo, e mais nada.

### CONTEXTO
{docs}

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
    format_chain = format_template | llm

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

    if results and key and lock:
        with lock:
            results[key]['response'] = result['text']
            results[key]['prompt'] = [
                str(map_prompt),
                str(reduce_prompt),
                str(format_template),
            ]

    # history.save('mapreduce', '', result['text'])
    return result.content

def parallel_mapreduce(docs: list[Document]):
    map_results = call_llms(
        [
            {
                "model": Config.OLLAMA_MODEL,
                "prompt": SimplePrompt(Prompts.MAP.format(context=doc.page_content)),
                "options": {
                    'temperature': 0.25,
                    'top_k': 5,
                    'top_p': 0.25,
                },
                "key": i
            } for i, doc in enumerate(docs)
        ],
        n_workers=len(Config.OLLAMA_BASE_URL_POOL),
        sort=True
    )
    summary = "\n".join([result[0]['response'] for result in map_results])

    reduce_results = call_llms(
        [
            {
                "model": Config.OLLAMA_MODEL,
                "prompt": SimplePrompt(Prompts.REDUCE.format(context=summary)),
                "options": {
                    'temperature': 0.25 + 0.1 * i,
                    'top_k': 5 + i,
                    'top_p': 0.25 + 0.1 * i,
                },
                "key": i
            } for i in range(Config.N_FACTOR)
        ],
        n_workers=len(Config.OLLAMA_BASE_URL_POOL),
        sort=True
    )
    return [result[0] for result in reduce_results]
