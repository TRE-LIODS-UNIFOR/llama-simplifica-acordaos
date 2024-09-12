from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama.llms import OllamaLLM

template = """Question: {question}

Answer: Let's think step by step."""

prompt = ChatPromptTemplate.from_template(template)

model = OllamaLLM(model="llama3.1", base_url='http://10.10.0.95:11434', keep_alive='60m',)

chain = prompt | model

res = chain.invoke({"question": "Can i use a LLaMa3.1 embedding model for reading PDFs? Explain in one paragraph."})

print(res)
