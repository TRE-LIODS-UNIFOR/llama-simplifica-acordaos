from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.prompts import PromptTemplate
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_ollama import ChatOllama
from langchain_text_splitters import RecursiveCharacterTextSplitter

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
    num_ctx=8_000,  # Sets the size of the context window used to generate the next token.
    verbose=False
)
loader = PyMuPDFLoader(file_path='acordao.pdf')
doc = loader.load()

print(doc)

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=20
)
chunks = text_splitter.split_documents(documents=doc)

template = """
### System:
Você é um explicador de acórdãos honesto.
Você receberá arquivos em PDF e produzirá resumos do documento.
Você deverá usar linguagem simples na produção dos resumos.
Use apenas informações contidas no arquivo na produção dos resumos. Nunca retire informações de outras fontes.

### Context:
{context}

### Response:
"""
prompt = PromptTemplate.from_template(template)
chain = create_stuff_documents_chain(llm, prompt)

# result = chain.invoke({"context": chunks})
# print(result)

# with open('out_large.txt', 'w') as f:
#     f.write(result)
