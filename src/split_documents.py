from langchain_community.document_loaders import PyMuPDFLoader, TextLoader
from langchain_core.documents import Document
from langchain_text_splitters import CharacterTextSplitter, RecursiveCharacterTextSplitter
from config import Config
from transformers import GPT2TokenizerFast

tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")

def split_documents(file_path, page_start=None, page_end=None, chunk_size=Config.SPLITTER_CHUNK_SIZE, chunk_overlap=Config.SPLITTER_CHUNK_OVERLAP, split_by='character'):
    if file_path.endswith('.pdf'):
        loader = PyMuPDFLoader(file_path=file_path)
        doc = loader.load()[page_start:page_end]
    else:
        loader = TextLoader(file_path)
        docs = loader.load()
        doc = [Document(page) for page in docs[0].page_content.split('\n-----\n')[page_start:page_end]]

    if page_start is None:
        page_start = 0
    if page_end is None:
        page_end = len(doc)

    if split_by == 'character':
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
        )
    elif split_by == 'token':
        text_splitter = CharacterTextSplitter.from_huggingface_tokenizer(
            tokenizer=tokenizer,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
        )
    docs = text_splitter.split_documents(doc)
    return docs

def split_text(text, chunk_size=Config.SPLITTER_CHUNK_SIZE, chunk_overlap=Config.SPLITTER_CHUNK_OVERLAP, split_by='character') -> list[Document]:
    doc = Document(page_content=text)
    if split_by == 'character':
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
        )
    elif split_by == 'token':
        text_splitter = CharacterTextSplitter.from_huggingface_tokenizer(
            tokenizer=tokenizer,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
        )
    docs = text_splitter.split_documents([doc])
    return docs
