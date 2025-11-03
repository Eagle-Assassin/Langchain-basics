from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv
from langchain_core.runnables import RunnableSequence,RunnableParallel, RunnablePassthrough

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



passthrough= RunnablePassthrough()



chain=RunnableSequence(RunnableSequence(prompt,model,parser,) ,RunnableParallel({'explaination':RunnableSequence(promt2,model,parser),
                                                                                 'joke':RunnablePassthrough()}))

print(chain.invoke({'topic':'AI'}))

print(chain.get_graph().print_ascii())


