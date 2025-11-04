from langchain_community.document_loaders import WebBaseLoader

from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

load_dotenv()

prompt1=PromptTemplate(
    template="Answer the following questions \n{questions} from the following text \n{text}",
    input_variables=['questions','text']
)

parser= StrOutputParser()

model=ChatOpenAI()

url ="https://www.amazon.in/ASUS-Dual-NVIDIA-GeForce-GDDR7/dp/B0F4DWKRBQ/?_encoding=UTF8&pd_rd_w=jCVIb&content-id=amzn1.sym.9edad6de-f85d-41dd-a893-96a2a0223a93%3Aamzn1.symc.b1464ab7-6d6a-4fc8-be8f-f2e9bcc64228&pf_rd_p=9edad6de-f85d-41dd-a893-96a2a0223a93&pf_rd_r=WVFR09V2YNH7RYB8VAGH&pd_rd_wg=M3BkA&pd_rd_r=b4096049-c731-4936-80df-58ef04679831&ref_=pd_hp_d_btf_ci_mcx_mr_ca_id_hp_d"

loader= WebBaseLoader(url)  # We can use multiple urls

docs=loader.load()

# print(len(docs))

# print(docs[0].page_content)

chain= prompt1|model|parser

result=chain.invoke({"questions":"What is the peak performace of this product","text":docs[0].page_content})

print(result)