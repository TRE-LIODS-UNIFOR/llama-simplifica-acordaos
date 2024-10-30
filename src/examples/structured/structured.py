from pprint import pprint
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from langchain_ollama import ChatOllama
from pydantic import BaseModel, Field

class Joke(BaseModel):
    setup: str = Field(description='the preparation for the joke.')
    punchline: str = Field(description='the resolution of the joke.')

class Story(BaseModel):
    beggining: str = Field(description='the start of the story')
    development: str = Field(description='the middle of the story')
    end: str = Field(description='the end of the story')


llm = ChatOllama(
    base_url='http://10.10.0.99:11434',
    model='llama3.2',
    context_size=10240,
    temperature=0,
)

llm_jokes = llm.with_structured_output(Joke)
llm_stories = llm.with_structured_output(Story)

prompt_joke = PromptTemplate.from_template('tell me a joke about {topic}')
prompt_story = PromptTemplate.from_template('tell me a story about {topic}')

chain_joke = prompt_joke | llm_jokes | (lambda res: JsonOutputParser().parse(res.model_dump_json()))
chain_story = prompt_story | llm_stories

result = chain_joke.invoke({'topic': 'bricks'})
pprint(result)
# pprint(result.model_dump())

# result = chain_story.invoke({'topic': 'bricks'})
# pprint(result.model_dump())
