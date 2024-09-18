from langchain.chains import RetrievalQA
# from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.prompts import PromptTemplate
from langchain_community.document_loaders import PyMuPDFLoader, TextLoader
from langchain_ollama import ChatOllama
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS

from langchain_huggingface import HuggingFaceEmbeddings

from history import save

llm = ChatOllama(
    temperature=0,
    base_url='http://10.10.0.95:11434',
    model='llama3.1',
    streaming=True,
    # seed=2,
    top_k=10,
    # A higher value (100) will give more diverse answers, while a lower value (10) will be more conservative.
    top_p=0.3,
    # Higher value (0.95) will lead to more diverse text, while a lower value (0.5) will generate more
    # focused text.
    num_ctx=4096,  # Sets the size of the context window used to generate the next token.
    verbose=False
)
# loader = PyMuPDFLoader(file_path='out/output.txt')
loader = TextLoader('out/ilomar.txt', 'utf8')
doc = loader.load()
loader = TextLoader('documentos/glossarios/teste.csv')
doc = doc + loader.load()

print(type(doc), doc)

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=20
)
chunks = text_splitter.split_documents(documents=doc)

embeddings_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")
vectorstore = FAISS.from_documents(chunks, embeddings_model)
retriever = vectorstore.as_retriever()

with open("prompts/templates/acordao_completo.txt", "r") as f:
    template = f.read()
prompt = PromptTemplate.from_template(template)
# chain = create_stuff_documents_chain(llm, prompt)

qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    retriever=retriever,  # here we are using the vectorstore as a retriever
    chain_type="stuff",
    return_source_documents=True,  # including source documents in output
    chain_type_kwargs={'prompt': prompt}  # customizing the prompt
)
response = qa_chain.invoke({'query': 'Produza o resumo simplificado.' })
result = response['result']

save("steps", template, result)

print("".join(result))
