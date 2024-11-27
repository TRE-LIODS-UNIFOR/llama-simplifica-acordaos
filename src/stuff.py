from pathlib import Path
from langchain_core.documents import Document
from langchain_ollama import ChatOllama, OllamaEmbeddings
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.prompts import PromptTemplate
from langchain_community.vectorstores import FAISS

from call_llms import call_llms
from config import Config
from llms import get_llama
from prompts.rag_prompt import RAGPrompt
from semantic_similarity import get_similarity_score


def stuff(docs: list[Document] | None = None, prompt: str | None = None, template_kvs: dict[str, str] | None = None, verbose=False, model_configuration=None):
    if docs is None:
        raise ValueError("docs must be provided")
    if prompt is None:
        raise ValueError("prompt must be provided")

    llm: ChatOllama = get_llama(log_callbacks=verbose, model_configuration=model_configuration)

    embeddings_model = OllamaEmbeddings(
        base_url=Config.OLLAMA_EMBEDDINGS_BASE_URL,
        model=Config.OLLAMA_EMBEDDINGS_MODEL
    )
    vectorstore = FAISS.from_documents(documents=docs, embedding=embeddings_model)
    retriever = vectorstore.as_retriever()

    prompt_template = PromptTemplate.from_template(prompt)
    document_chain = create_stuff_documents_chain(llm, prompt_template)
    retrieval_chain = create_retrieval_chain(retriever, document_chain)

    chunks = []

    input_dict = {
        'input': 'Comece.'
    }
    if template_kvs:
        input_dict.update(template_kvs)
    if verbose: print("Starting retrieval chain. Input:", input_dict)
    for chunk in retrieval_chain.stream(
        input_dict
    ):
        answer = chunk.get('answer', '')
        chunks.append(answer)
        if verbose: print(answer, end='', flush=True)
    if verbose: print()
    result = "".join(chunks)

    del(vectorstore)
    return result

def n_stuff(n: int = 1, docs: list[Document] | None = None, prompt: str | None = None, similarity_threshold: float = 0.33, ground_truth: str | None = None):
    if docs is None:
        raise ValueError("docs must be provided")
    responses: list[tuple[dict[str, str], int | None]] = call_llms([
        {
            "model": Config.OLLAMA_MODEL,
            "prompt": RAGPrompt(prompt),
            "options": {
                'temperature': 0.0,
                'top_k': 2,
                'top_p': 0.05,
                'documents': docs,
            },
        } for _ in range(n)
    ])

    # print(responses)

    response_contents = [response[0]['response'] for response in responses]
    print(response_contents)

    if ground_truth is None:
        ground_truth = "\n".join([page.page_content for page in docs])

    scores = []
    for response in response_contents:
        print(response)
        print(ground_truth)
        similarity = get_similarity_score(response, ground_truth, method='bertscore', model_name=Config.BERTSCORE_MODEL)
        print(similarity)
        scores.append(similarity)
        print("\n\n")
    return response_contents, scores

def most_similar(responses: list[str], scores: list[float]) -> tuple[str, float]:
    max_score = max(scores)
    max_index = scores.index(max_score)
    return responses[max_index], max_score
