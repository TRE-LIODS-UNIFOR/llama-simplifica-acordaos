import ollama

from config import Config


class Prompt:
    def __init__(self, prompt: str | None =None):
        if prompt is None:
            raise ValueError("Prompt must be provided")
        self.prompt: str = prompt

    def execute(self, model=None, host=None, options=None):
        raise NotImplementedError()

    def __repr__(self):
        return self.prompt

class SimplePrompt(Prompt):
    def __init__(self, prompt=None):
        super().__init__(prompt=prompt)

    def execute(self, model: str | None = None, host: str | None = None, options: dict | None = None):
        if host is None:
            raise ValueError("Host must be provided")
        if model is None:
            raise ValueError("Model must be provided")
        if options is None:
            raise ValueError("Options must be provided")
        client = ollama.Client(host=host)
        options.update(
            {
                'temperature': options.get('temperature', Config.OLLAMA_TEMPERATURE),
                'top_k': options.get('top_k', Config.OLLAMA_TOP_K),
                'top_p': options.get('top_p', Config.OLLAMA_TOP_P),
                'num_ctx': options.get('num_ctx', Config.OLLAMA_CONTEXT_SIZE),
                'repeat_penalty': options.get('repeat_penalty', 1.1),
                'repeat_last_n': options.get('repeat_last_n', 64),
            }
        )
        response = client.generate(model=model, prompt=self.prompt, options=options) # type: ignore
        return response
