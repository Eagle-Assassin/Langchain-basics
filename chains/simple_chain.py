from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

load_dotenv()

prompts =PromptTemplate(
    template='Generate 5 interesting facts about {topics}',
    input_variables=['topics']
)

model= ChatOpenAI()

parser =StrOutputParser()

chain=prompts|model|parser

result= chain.invoke({'topics':'cricket'})

print(result)

#visualise chain
chain.get_graph().print_ascii()