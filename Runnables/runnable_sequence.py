from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv
from langchain_core.runnables import RunnableSequence

load_dotenv()

prompt=PromptTemplate(
    template='write a joke about{topic}',
    input_variables=['topic']
)

promt2=PromptTemplate(
    template='explain the following joke -\n{text}',
    input_variables=['text']
)

model= ChatOpenAI()

parser=StrOutputParser()

chain= RunnableSequence(prompt,model,parser,promt2,model,parser)

print(chain.invoke({'topic':'AI'}))