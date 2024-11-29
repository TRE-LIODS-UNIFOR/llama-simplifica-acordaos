from langchain_ollama import ChatOllama, OllamaEmbeddings

from config import Config

from langchain_core.callbacks import BaseCallbackHandler

class MyLoggingCallbackHandler(BaseCallbackHandler):
    def on_chain_start(self, chain, inputs, **kwargs):
        print(f"\nStarting chain: {chain.__class__.__name__}")
        print(f"Inputs: {inputs}")

    def on_chain_end(self, outputs, **kwargs):
        print(f"Outputs: {outputs}")

    def on_llm_start(self, llm, prompts, **kwargs):
        for prompt in prompts:
            print(f"\nPrompt: {prompt}\n\nTokens: {get_llama().get_num_tokens(prompt)}")

    # def on_llm_end(self, response, **kwargs):
    #     print(f"LLM Response: {response.generations[0][0].text}")

callback_handler = MyLoggingCallbackHandler()

def get_llama(host=0, model=0, log_callbacks=False, base_url=None, model_configuration=None) -> ChatOllama:
    model_configs = {
        'base_url': Config.OLLAMA_BASE_URL,
        'temperature': Config.OLLAMA_TEMPERATURE,
        'model': Config.OLLAMA_MODEL,
        'stream': True,
        'top_k': Config.OLLAMA_TOP_K,
        'top_p': Config.OLLAMA_TOP_P,
        'num_ctx': Config.OLLAMA_CONTEXT_SIZE,
        'verbose': False,
        'keep_alive': Config.OLLAMA_KEEP_ALIVE,
        'callbacks': [callback_handler] if log_callbacks else None,
        'repeat_last_n': Config.OLLAMA_REPEAT_LAST_N,
    }
    base_url = base_url or (Config.OLLAMA_BASE_URL if host == 0 else Config.OLLAMA_BASE_URL_2)

    if model_configuration:
        print(f"Overriding model configuration with: {model_configuration}")
        model_configs.update(model_configuration)

    print(f"Using model {model} with configuration: {model_configs}")

    return ChatOllama(
        temperature=model_configs['temperature'],
        base_url=model_configs['base_url'] or base_url,
        model=model_configs['model'],
        # stream=model_configs['stream'],
        top_k=model_configs['top_k'],   # A higher value (100) will give more diverse answers, while a lower value (10) will be more conservative.
        top_p=model_configs['top_p'],  # Higher value (0.95) will lead to more diverse text, while a lower value (0.5) will generate more focused text.
        num_ctx=model_configs['num_ctx'],  # Sets the size of the context window used to generate the next token.
        verbose=False,
        keep_alive=model_configs['keep_alive'],
        callbacks=[callback_handler] if log_callbacks else None,
        repeat_last_n=model_configs['repeat_last_n'],
        repeat_penalty=1.1,
        seed=42,
    )

def get_embeddings_model(base_url=None, model_name=None) -> OllamaEmbeddings:
    return OllamaEmbeddings(
        base_url=base_url or Config.OLLAMA_EMBEDDINGS_BASE_URL,
        model=Config.OLLAMA_EMBEDDINGS_MODEL
   )
