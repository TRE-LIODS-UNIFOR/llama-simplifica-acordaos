from langchain_ollama import ChatOllama, OllamaEmbeddings
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_community.vectorstores import FAISS

from config import Config
from llms import get_llama
from semantic_similarity import get_similarity_score


def stuff(docs=None, host=0, results=None, key=None, prompt=None, template_kvs: dict[str, str] | None = None, base_url=None, verbose=False):
    llm: ChatOllama = get_llama(host=host, model=0, log_callbacks=verbose, base_url=base_url)

    embeddings_model = OllamaEmbeddings(
        base_url=Config.OLLAMA_EMBEDDINGS_BASE_URL,
        model=Config.OLLAMA_EMBEDDINGS_MODEL
    )
    vectorstore = FAISS.from_documents(documents=docs, embedding=embeddings_model)
    retriever = vectorstore.as_retriever()

    document_chain = create_stuff_documents_chain(llm, prompt)
    retrieval_chain = create_retrieval_chain(retriever, document_chain)

    chunks = []

    input_dict = {
        'input': 'Comece.'
    }
    if template_kvs:
        input_dict.update(template_kvs)
    for chunk in retrieval_chain.stream(
        input_dict
    ):
        answer = chunk.get('answer', '')
        chunks.append(answer)
        if verbose: print(answer, end='', flush=True)
    print()
    result = "".join(chunks)

    if results is not None and key is not None:
        results[key]['response'] = result
        results[key]['prompt'] = [
            str(prompt),
        ]

    del(vectorstore)
    return result

def n_stuff(n=1, docs=None, host=0, results=None, key=None, prompt=None, template_kvs: dict[str, str] | None = None, base_url=None, verbose=False, similarity_threshold=0.33):
    responses = []
    for _ in range(n):
        response = stuff(docs, host, results, key, prompt, template_kvs, base_url, verbose)
        responses.append(response)
    scores = []
    original = "\n".join([page.page_content for page in docs])
    for response in responses:
        similarity = get_similarity_score(response, original, method='bertscore')
        scores.append(similarity)
    return responses, scores

def most_similar(responses, scores):
    max_score = max(scores)
    max_index = scores.index(max_score)
    return responses[max_index], max_score
