import ollama


class Prompt:
    def __init__(self, prompt=None):
        self.prompt = prompt

    def execute(self, model=None, host=None, options=None):
        raise NotImplementedError()

    def __repr__(self):
        return self.prompt

class SimplePrompt(Prompt):
    def __init__(self, prompt=None):
        super().__init__(prompt=prompt)

    def execute(self, model=None, host=None, options=None):
        client = ollama.Client(host=host)
        response = client.generate(model=model, prompt=self.prompt, options=options)
        return response
