import argparse
import sys
from chain import stuff_documents_chain, retrieval_chain
from dotenv import load_dotenv
from embedding import embedding
from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_ollama import ChatOllama
from load_document import load_text, split_documents
from pathlib import Path
from prompts import from_template
from typing import List
from vectorstore import to_vectorstore, to_retriever
import os

if __name__ == "__main__":
    BASE_DIR: Path = Path(__file__).resolve().parent

    load_dotenv()
    parser = argparse.ArgumentParser()
    parser.add_argument(
        'sources',
        action='store',
        nargs='+',
        help='Caminhos dos documentos fonte.',
    )
    parser.add_argument(
        'prompts',
        action='store',
        nargs='+',
        help='Caminhos dos prompts, na ordem desejada de execução.'
    )
    parser.add_argument(
        'destination',
        action='store',
        help='Caminho de destino, para salvar os prompts e o resultado.',
    )
    args = parser.parse_args()

    sources: List[Path] = [Path(source) for source in args.sources]
    prompts: List[Path] = [Path(prompt) for prompt in args.prompts]
    destination: Path = Path(args.destination)
    print(sources, destination)

    print("Carregando documentos")
    docs: List[Document] = load_text(
        sources
    )
    print("Dividindo documentos")
    chunks: List[Document] = split_documents(
        docs[:1]
    )
    print("Criando modelo embarcado")
    embedding = embedding()
    print("Criando vectorstores")
    vectorstore = to_vectorstore(chunks, embedding)
    print("Criando retriever")
    retriever = to_retriever(vectorstore)

    print("Carregando prompts")
    prompts = [from_template(prompt) for prompt in prompts]

    print("Criando conexão com modelo")
    model = ChatOllama(
        base_url=os.environ.get('OLLAMA_BASE_URL') or 'http://localhost:11434',
        temperature=0,
        model='llama3.1',
        streaming=True,
        top_k=10,   # A higher value (100) will give more diverse answers, while a lower value (10) will be more conservative.
        top_p=0.3,  # Higher value (0.95) will lead to more diverse text, while a lower value (0.5) will generate more focused text.
        num_ctx=4096,  # Sets the size of the context window used to generate the next token.
        verbose=False
    )

    print("Criando chain de documentos")
    chain = stuff_documents_chain(model, prompts[0])
    print("Criando chain de retrieval")
    qa_chain = retrieval_chain(retriever, chain)

    print("Invocando chain")
    result = qa_chain.invoke({'input': 'Inicie.'})
    print(type(result))
    print("Resultado:\n")
    print(result['answer'])
