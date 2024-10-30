import sys
from uuid import uuid4
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.prompts import PromptTemplate
from langchain_chroma import Chroma
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_ollama import ChatOllama
from langchain_community.embeddings.ollama import OllamaEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

from config import Config
from database import PromptDB
from models import Document, Prompt, Response

db = PromptDB()

llm = ChatOllama(
    temperature=Config.OLLAMA_TEMPERATURE,
    base_url=Config.OLLAMA_BASE_URL,
    model=Config.OLLAMA_MODEL,
    streaming=True,
    top_k=Config.OLLAMA_TOP_K,   # A higher value (100) will give more diverse answers, while a lower value (10) will be more conservative.
    top_p=Config.OLLAMA_TOP_P,  # Higher value (0.95) will lead to more diverse text, while a lower value (0.5) will generate more focused text.
    num_ctx=Config.OLLAMA_CONTEXT_SIZE,  # Sets the size of the context window used to generate the next token.
    verbose=False,
    keep_alive=Config.OLLAMA_KEEP_ALIVE
)

document_id = sys.argv[1]
# print(f"document_id: {document_id}")

loader = PyMuPDFLoader(file_path=f'documentos/acordaos/{document_id}')
doc = loader.load()

document = Document(
    id=document_id
)
document_id = db.insert_document(document)

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=Config.SPLITTER_CHUNK_SIZE,
    chunk_overlap=Config.SPLITTER_CHUNK_OVERLAP
)
chunks = text_splitter.split_documents(documents=[doc[0]])

# print(f"amount of chunks: {len(chunks)}")
# pprint(chunks)

embeddings_model = OllamaEmbeddings(
    base_url=Config.OLLAMA_EMBEDDINGS_BASE_URL,
    model=Config.OLLAMA_EMBEDDINGS_MODEL
)
vectorstore = Chroma(
    collection_name='acordaos',
    embedding_function=embeddings_model,
    persist_directory=f'./chroma_db/{document_id.split('.')[0]}'
)
vectorstore.add_documents(documents=chunks)
retriever = vectorstore.as_retriever()

with open('prompts/simplificar_acordao/cabecalho.txt', 'r') as f:
    prompt_text = f.read()
    prompt = PromptTemplate.from_template(prompt_text)

# prompt_id = str(uuid4())
# prompt_id = db.insert_prompt(
#     Prompt(
#         id=prompt_id,
#         temperature=Config.OLLAMA_TEMPERATURE,
#         prompt=prompt_text,
#         chunk_overlap=Config.SPLITTER_CHUNK_OVERLAP,
#         chunk_size=Config.SPLITTER_CHUNK_SIZE,
#         context_size=Config.OLLAMA_CONTEXT_SIZE,
#         embeddings_model=Config.OLLAMA_EMBEDDINGS_MODEL,
#         model=Config.OLLAMA_MODEL,
#         top_k=Config.OLLAMA_TOP_K,
#         top_p=Config.OLLAMA_TOP_P,
#         created_at=None
#     )
# )

document_chain = create_stuff_documents_chain(llm, prompt)
retrieval_chain = create_retrieval_chain(retriever, document_chain)

def write_chunk(chunk):
    with open('chunks.txt', 'a') as f:
        f.write(chunk)

chunks = []

for chunk in retrieval_chain.stream(
    {'input':
        'Liste os tópicos especificados anteriormente no prompt de sistema.'}
):
    answer = chunk.get('answer', '')
    chunks.append(answer)
    write_chunk(answer)
    print(answer, end='', flush=True)
print()

#answer = result['answer']
# print(answer)
# try:
#     quality = int(input('quality (0-10): '))
# except KeyboardInterrupt:
#     quality = -1

# response_id = str(uuid4())
# db.insert_response(
#     Response(
#         id=response_id,
#         prompt_id=prompt_id,
#         quality=quality,
#         response=answer,
#         created_at=None,
#         document_id=document_id
#     )
# )

# db.commit()
# db.close()

# corpo = "# TRIBUNAL REGIONAL ELEITORAL DO CEARÁ\n\n" + corpo

# to_save = corpo
# save('simplificar_acordao', '', to_save)

# print("".join(result))
