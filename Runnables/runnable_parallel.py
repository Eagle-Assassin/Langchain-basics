from langchain_openai import ChatOpenAI
from langchain_huggingface import ChatHuggingFace , HuggingFaceEndpoint
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableParallel, RunnableSequence

load_dotenv()

model1=ChatOpenAI()
parser= StrOutputParser()

llm = HuggingFaceEndpoint(
    repo_id="mistralai/Mixtral-8x7B-Instruct-v0.1",
    task="text-generation",
)

model2= ChatHuggingFace(llm=llm)

prompt1 = PromptTemplate(
    template="generate a tweet based on the topic {input}",
    input_variables=['input']
)

prompt2 = PromptTemplate(
    template="generate a linked in post based on the topic {input}",
    input_variables=['input']
)

parallel_chain=RunnableParallel({'tweet':RunnableSequence(prompt1,model1,parser),
                                 'linkedin':RunnableSequence(prompt2,model2,parser)})


result=parallel_chain.invoke({'input':'AI'})


print(result)

parallel_chain.get_graph().print_ascii()