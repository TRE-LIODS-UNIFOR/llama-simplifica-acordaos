from pprint import pprint
import sys
import os
from threading import Lock, Thread

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import summarize
from utils.thread_with_result import thread_with_result

from summarize import summarize_section


if __name__ == "__main__":
    models = ["llama3.2:3b-instruct-q4_K_M", "llama3.1:8b"]
    urls = ["http://10.10.0.99:11434", "http://10.10.0.95:11434"]
    doc_path = "documentos/acordaos/0600012-49_REl_28052024_1.txt"
    page_start = 2
    page_end = 5
    prompt = summarize.Prompts.RELATORIO

    models_urls = zip(models, urls)
    results = {}

    threads = []
    lock = Lock()
    for model, url in models_urls:
        print(f"Starting thread for model {model} with url {url}")
        target = thread_with_result(func=summarize_section, results=results, key=model, lock=lock, args=(doc_path, page_start, page_end, prompt, url, False, True, None, None, None, {"model": model, "base_url": url}))
        threads.append(Thread(target=target))

    for thread in threads:
        thread.start()

    for thread in threads:
        thread.join()

    pprint(results)
