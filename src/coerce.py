from langchain_ollama import ChatOllama
from langchain_core.output_parsers import StrOutputParser
from langchain.prompts import PromptTemplate

model = ChatOllama(
    temperature=0,
    base_url='http://10.10.0.95:11434',
    model='llama3.1',
    streaming=True,
    # seed=2,
    top_k=10,
    # A higher value (100) will give more diverse answers, while a lower value (10) will be more conservative.
    top_p=0.3,
    # Higher value (0.95) will lead to more diverse text, while a lower value (0.5) will generate more
    # focused text.
    num_ctx=4096,  # Sets the size of the context window used to generate the next token.
    verbose=False
)

prompt_1 = PromptTemplate.from_template("me conte uma piada sobre {topico}")
prompt_2 = PromptTemplate.from_template("essa piada é engraçada? {piada}")

chain = prompt_1 | model | StrOutputParser()
composed_chain = {"piada": chain} | prompt_2 | model | StrOutputParser()

result = composed_chain.invoke({"topico": "martelos"})
print(result)
