from langchain_ollama import ChatOllama
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_core.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain_chroma import Chroma
from langchain_core.output_parsers import JsonOutputParser
from pydantic import BaseModel, Field
from pprint import pprint
from langchain_ollama import OllamaEmbeddings
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain

from langchain_community.embeddings import HuggingFaceEmbeddings

llm = ChatOllama(
    base_url='http://10.10.0.99:11434',
    model='llama3.2',
    context_size=1024,
    temperature=0,
    top_k=10,
)

cabecalho = {
    'data_do_julgamento': '04/06/2024',
    'numero_do_recurso_eleitoral': '0600012-49.2024.6.06.0033',
    'origem': 'Canindé/CE',
    'recorrentes': [
        {
            'advogados': [
                'Francisco Jardel Rodrigues de Souza (OABCE32787-A)',
                'Lidienira Cavalcante Mendonça Vieira (OAB CE0016731)'
            ],
            'recorrente': 'Antônio Ilomar Vasconcelos Cruz'
        }
    ],
    'recorridos': [
        {
            'advogados': None,
            'recorrido': 'Ministério Público Eleitoral'
        }
    ],
    'relator': 'Desembargador Eleitoral Daniel Carvalho Carneiro'
}

relatorio_prompt = PromptTemplate.from_template(
    f"""
    ### Missão
    Responda de forma resumida, em até três frases, à seguinte pergunta:

    * Por quê o autor do recurso eleitoral ({cabecalho['recorrentes'][0]['recorrente']}) interpôs o recurso?
    
    ### Contexto

    {{context}}

    ### Resposta em lista pontuada
    """
)

#    * O que o réu alegou para se defender?
#    * O que o juiz decidiu?
#    * Quais foram as justificativas legais da decisão?
#    * Quais artigos de lei e fundamentos jurídicos foram citados?
#    * Quem recorreu?
#    * O que alegou para recorrer?

loader = PyMuPDFLoader(file_path='../../documentos/acordaos/0600012-49_REl_28052024_1.pdf')
document = loader.load()[2:4]

splitter = RecursiveCharacterTextSplitter(
    chunk_size=512,
    chunk_overlap=64
)

docs = splitter.split_documents(document)

embeddings_model = HuggingFaceEmbeddings(
    model_name='neuralmind/bert-base-portuguese-cased'
)
vectorstore = Chroma(
    collection_name='acordaos',
    embedding_function=embeddings_model,
)
vectorstore.add_documents(documents=docs)

retriever = vectorstore.as_retriever()

stuff_chain = create_stuff_documents_chain(llm, relatorio_prompt)
retrieval_chain = create_retrieval_chain(retriever, stuff_chain)

result = retrieval_chain.invoke({'input': 'Responda.'})

pprint(result)
pprint(result['answer'])

