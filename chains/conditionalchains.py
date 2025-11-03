from langchain_openai import ChatOpenAI
from langchain_huggingface import ChatHuggingFace , HuggingFaceEndpoint
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field
from typing import Literal
from langchain_core.runnables import RunnableBranch, RunnableLambda


load_dotenv()

parser =StrOutputParser()

class Feedback(BaseModel):
    sentiment:Literal['positive', 'negative']=Field(description='Give the sentiment of the feedback')

parser2 = PydanticOutputParser(pydantic_object=Feedback)

model= ChatOpenAI()


prompt1=PromptTemplate(template="Classify the sentiment of the following text in to positive or negative \n {feedback} \n{format_instructions}",
                       input_variables=['feedback'],
                       partial_variables={'format_instructions':parser2.get_format_instructions()})


classifierchain= prompt1 | model  | parser2

promt2= PromptTemplate(template="Write an approximate response to this  feedback \n {feedback}",
                       input_variables=["feedback"])

promt3= PromptTemplate(template="Write an approximate response to this  feedback \n {feedback}",
                       input_variables=["feedback"])


branch_chain = RunnableBranch(
    (lambda x:x.sentiment == 'positive', promt2 | model | parser),
    (lambda x:x.sentiment== 'negative', promt3 | model | parser),
    RunnableLambda(lambda x: "could not find sentiment")
)


chain = classifierchain | branch_chain

result = chain.invoke({"feedback":"this is a terrible phone"})

# result=classifierchain.invoke({"feedback":"this is a terrible phone"})

print((result))