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

def get_llama(host=0, model=0, log_callbacks=False, base_url=None) -> ChatOllama:
    model_configs = [
        {
            'temperature': Config.OLLAMA_TEMPERATURE,
            'model': Config.OLLAMA_MODEL,
            'streaming': True,
            'top_k': Config.OLLAMA_TOP_K,
            'top_p': Config.OLLAMA_TOP_P,
            'num_ctx': Config.OLLAMA_CONTEXT_SIZE,
            'verbose': False,
            'keep_alive': Config.OLLAMA_KEEP_ALIVE,
            'callbacks': [callback_handler] if log_callbacks else None,
            'repeat_last_n': Config.OLLAMA_REPEAT_LAST_N,
        },
        {
            'temperature': Config.OLLAMA_TEMPERATURE_2,
            'model': Config.OLLAMA_MODEL_2,
            'streaming': True,
            'top_k': Config.OLLAMA_TOP_K_2,
            'top_p': Config.OLLAMA_TOP_P_2,
            'num_ctx': Config.OLLAMA_CONTEXT_SIZE_2,
            'verbose': False,
            'keep_alive': Config.OLLAMA_KEEP_ALIVE_2,
            'callbacks': [callback_handler] if log_callbacks else None,
            'repeat_last_n': Config.OLLAMA_REPEAT_LAST_N,
        },
    ]

    base_url = base_url or (Config.OLLAMA_BASE_URL if host == 0 else Config.OLLAMA_BASE_URL_2)

    return ChatOllama(
        temperature=model_configs[model]['temperature'],
        base_url=base_url,
        model=model_configs[model]['model'],
        streaming=model_configs[model]['streaming'],
        top_k=model_configs[model]['top_k'],   # A higher value (100) will give more diverse answers, while a lower value (10) will be more conservative.
        top_p=model_configs[model]['top_p'],  # Higher value (0.95) will lead to more diverse text, while a lower value (0.5) will generate more focused text.
        num_ctx=model_configs[model]['num_ctx'],  # Sets the size of the context window used to generate the next token.
        verbose=False,
        keep_alive=model_configs[model]['keep_alive'],
        callbacks=[callback_handler] if log_callbacks else None,
        repeat_last_n=model_configs[model]['repeat_last_n'],
    )

def get_embeddings_model(base_url=None):
    return OllamaEmbeddings(
        base_url=base_url or Config.OLLAMA_EMBEDDINGS_BASE_URL,
        model=Config.OLLAMA_EMBEDDINGS_MODEL
   )
