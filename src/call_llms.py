from queue import Queue
from threading import Lock, Thread

from config import Config
from prompts.prompts import Prompt


class LLMPool:
    def __init__(self, n_workers, hosts):
        self.n_workers = n_workers
        self.lock = Lock()
        self.queue = Queue()
        self.hosts = hosts
        self.workers = [LLMWorker(self.queue, self, self.lock) for _ in range(n_workers)]
        self.results = []

    def run(self, data):

        for worker in self.workers:
            worker.start()

        for i in range(len(data)):
            if self.hosts:
                data[i].update({'host': self.hosts[i % len(self.hosts)], 'verbose': True})
            self.queue.put(data[i])

        for _ in range(self.n_workers):
            self.queue.put(None)

        for worker in self.workers:
            worker.join()

        return self.results

    def collect(self, result):
        self.results.append(result)
        print(f"{len(self.results)} results collected")

class LLMWorker(Thread):
    def __init__(self, queue, pool, lock):
        self.queue = queue
        self.pool = pool
        self.lock = lock
        super(LLMWorker, self).__init__()

    def run(self):
        while True:
            with self.lock:
                data = self.queue.get()
            if data is None:
                break
            try:
                result = self.process(data)
                with self.lock:
                    self.pool.collect(result)
                print('Finished processing data:', data)
            except Exception as e:
                print('Error processing data:', data, "Error:", e)
                with self.lock:
                    self.queue.put(data)

    def process(self, data):
        print('Processing data:', data)
        key = data.get('key', None)
        # client = ollama.Client(host=data['host'])
        prompt: Prompt = data['prompt']
        # response = client.generate(model=data['model'], prompt=prompt, options=data['options'])
        response = prompt.execute(host=data['host'], model=data['model'], options=data['options'])
        return response, key

# TODO: n_workers should come from .env
# TODO: amount of prompts should come from .env
def call_llms(data, n_workers=4, hosts=Config.OLLAMA_BASE_URL_POOL, sort=False):
    pool = LLMPool(n_workers=n_workers, hosts=hosts)
    results = pool.run(data)
    if sort:
        results = sorted(results, key=lambda x: x[1])
    return results
