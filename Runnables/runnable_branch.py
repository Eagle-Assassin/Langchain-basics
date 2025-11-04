from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv
from langchain_core.runnables import RunnableSequence,RunnableLambda, RunnablePassthrough,RunnableParallel,RunnableBranch


load_dotenv()

prompt1 = PromptTemplate(
    template="Write a detailed report on {topic}",
    input_variables=['topic']
)

prompt2=  PromptTemplate(
    template="Summarize the following \n{text}",
    input_variables=['text']
)

model = ChatOpenAI()

parser = StrOutputParser()

report_generation_chain= RunnableSequence(prompt1,model,parser)
branch_chain= RunnableBranch(
    # (condition, runnabele),
    (lambda x:len(x.split())>500,RunnableSequence(prompt2,model,parser)),
    RunnablePassthrough()
)

final_chain= RunnableSequence(report_generation_chain,branch_chain)

result=final_chain.invoke({'topic':'russia v/s Ukraine'})

print(result)