import sys
from langchain.prompts import PromptTemplate
from langchain_community.vectorstores import FAISS
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain

from langchain_ollama import ChatOllama, OllamaEmbeddings
from config import Config
from llms import MyLoggingCallbackHandler
from prompts.prompts import Prompt


class RAGPrompt(Prompt):
    def __init__(self, prompt: str | None = None):
        if prompt is None:
            raise ValueError("Prompt must be provided")
        super().__init__(prompt=prompt)

    def execute(self, model: str, host=None, options=None):
        if host is None:
            host = Config.OLLAMA_BASE_URL
        if options is None:
            options = {}

        client = ChatOllama(
            base_url=host,
            model=model,
            temperature=options.get('temperature', Config.OLLAMA_TEMPERATURE),
            top_k=options.get('top_k', Config.OLLAMA_TOP_K),
            top_p=options.get('top_p', Config.OLLAMA_TOP_P),
            num_ctx=options.get('num_ctx', Config.OLLAMA_CONTEXT_SIZE),
            repeat_penalty=options.get('repeat_penalty', 1.1),
            repeat_last_n=options.get('repeat_last_n', 64),
            callbacks=[MyLoggingCallbackHandler()] if options.get('verbose', False) else None
        )
        embeddings = OllamaEmbeddings(
            base_url=host,
            model=options.get('embeddings_model', 'nomic-embed-text')
        )
        vectorstore = FAISS.from_documents(documents=options['documents'], embedding=embeddings)
        retriever = vectorstore.as_retriever()

        print("Size of vectorstore in B:", sys.getsizeof(vectorstore))
        print("Size of documents in B:", sys.getsizeof(options['documents']))

        prompt_template = PromptTemplate.from_template(self.prompt)

        print("Prompt:")
        print(self.prompt)
        print()

        document_chain = create_stuff_documents_chain(client, prompt_template)
        retrieval_chain = create_retrieval_chain(retriever, document_chain)

        input_dict = {
            'input': 'Comece.'
        }
        input_dict.update(options.get('input_dict', {}))
        result = retrieval_chain.invoke(input_dict)
        answer = result['answer']

        print(answer)

        del(vectorstore)
        return { 'response': answer }
