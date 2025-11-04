from langchain_community.document_loaders import TextLoader
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from dotenv import load_dotenv

load_dotenv()

loader = TextLoader("blackhole.txt", encoding='utf-8')
docs=loader.load() #text file loaded as docs
parser=StrOutputParser()


model= ChatOpenAI()

print(type(docs))

print(len(docs))

print((docs[0].page_content))  #Page content and meta data
print((docs[0].metadata)) 

prompt = PromptTemplate(
    template= 'Write a summary for the following poem -\n{text}',
    input_variables=['text']
)

chain= prompt|model|parser

print(chain.invoke({'text':docs[0].page_content}))