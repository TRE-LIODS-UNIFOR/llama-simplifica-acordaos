from queue import Queue
from threading import Lock, Thread

from ollama import ResponseError

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

    def run(self, data) -> list[tuple[dict[str, str], int | None]]:
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

    def run(self) -> None:
        while True:
            with self.lock:
                data: dict = self.queue.get()
            if data is None:
                break
            data['retry_count'] = data.get('retry_count', 0)
            try:
                result: tuple[dict[str, str], float | None] = self.process(data)
                with self.lock:
                    self.pool.collect(result)
                print('Finished processing data:', data)
            except ResponseError as e:
                print('Error processing data:', data, "Error:", e.error)
                data['retry_count'] = data.get('retry_count', 0) + 1
                if data['retry_count'] < len(self.pool.hosts):
                    data['host'] = self.pool.hosts[data['retry_count']]
                    print('Retrying data:', data)
                with self.lock:
                    self.queue.put(data)

    def process(self, data) -> tuple[dict[str, str], float | None]:
        print('Processing data:', data)
        key: int | None = data.get('key', None)
        prompt: Prompt = data['prompt']
        if data.get('options') is None:
            data['options'] = {}
        response = prompt.execute(host=data['host'], model=data.get('model', Config.OLLAMA_MODEL), options=data['options'])
        return response, key

# TODO: amount of prompts should come from .env
def call_llms(data, n_workers=4, hosts=Config.OLLAMA_BASE_URL_POOL, sort=False) -> list[tuple[dict[str, str], int | None]]:
    n_workers = min(n_workers, len(hosts))
    pool: LLMPool = LLMPool(n_workers=n_workers, hosts=hosts)
    results: list[tuple[dict[str, str], int | None]] = pool.run(data)
    if sort:
        results = sorted(results, key=lambda x: x[1]) # type: ignore
    return results
