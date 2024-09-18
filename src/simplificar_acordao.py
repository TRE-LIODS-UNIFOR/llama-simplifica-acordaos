from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.prompts import PromptTemplate
from langchain_community.document_loaders import TextLoader
from langchain_ollama import ChatOllama
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_core.output_parsers import StrOutputParser

from langchain_huggingface import HuggingFaceEmbeddings

from history import save

llm = ChatOllama(
    temperature=0,
    base_url='http://10.10.0.95:11434',
    model='llama3.1',
    streaming=True,
    top_k=10,   # A higher value (100) will give more diverse answers, while a lower value (10) will be more conservative.
    top_p=0.3,  # Higher value (0.95) will lead to more diverse text, while a lower value (0.5) will generate more
    # focused text.
    num_ctx=4096,  # Sets the size of the context window used to generate the next token.
    verbose=False
)
# loader = PyMuPDFLoader(file_path='out/output.txt')
loader = TextLoader('out/ilomar.txt', 'utf8')
doc = loader.load()

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=20
)
chunks = text_splitter.split_documents(documents=doc)

embeddings_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")
vectorstore = FAISS.from_documents(chunks, embeddings_model)
retriever = vectorstore.as_retriever()

glossario = TextLoader('documentos/glossarios/teste.csv')
glossario = glossario.load()

chunks = text_splitter.split_documents(documents=glossario)
vectorstore_2 = FAISS.from_documents(chunks, embeddings_model)
retriever_2 = vectorstore_2.as_retriever()

prompts = []

with open("prompts/simplificar_acordao/1 - listar_pontos_principais.txt", "r") as f:
    prompts.append(PromptTemplate.from_template(f.read()))
with open("prompts/simplificar_acordao/3 - listar_complexos.txt", "r") as f:
    prompts.append(PromptTemplate.from_template(f.read()))

chain = create_stuff_documents_chain(llm, prompts[0])
qa_chain = create_retrieval_chain(retriever, chain)

lista_complexos = {'acordao': qa_chain} | prompts[1] | llm | StrOutputParser()
result = lista_complexos.invoke({'input': 'Inicie.'})

save('simplificar_acordao', '', result)

print("".join(result))
