from langchain_openai import ChatOpenAI
from langchain_huggingface import ChatHuggingFace , HuggingFaceEndpoint
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableParallel

load_dotenv()

model1=model=ChatOpenAI()


llm = HuggingFaceEndpoint(
    repo_id="mistralai/Mixtral-8x7B-Instruct-v0.1",
    task="text-generation",
)

model2=ChatHuggingFace(llm=llm)

prompt1= PromptTemplate(
    template="Generate short and simple notes from the following text \n{text}"
,
    input_variables=['text']
)
prompt2=PromptTemplate(template="generate 5 Question and answers from the text \n {text}",
    input_variables=['text']
)

prompt3=PromptTemplate(template="Merge the provided notes and quiz into a single document \n notes ->{notes} and quiz -> {quiz}",
    input_variables=['notes','quiz']
)

parser= StrOutputParser()

parallel_chain=RunnableParallel({
    'notes': prompt1|model1|parser,
    'quiz':prompt2|model2|parser
})

merge_chain =prompt3|model1|parser

chain = parallel_chain|merge_chain

text="""Google Preferred Source

When the Taliban recaptured Kabul in August 2021, after two decades of insurgency, Pakistan viewed it as a tactical victory. Imran Khan, then Prime Minister, famously said the Afghans had “broken the shackles of slavery”. But the Taliban’s return emboldened the Tehrik-e Taliban Pakistan (TTP), which is organisationally distinct from, yet ideologically aligned with, the Afghan Taliban. Over four years, Pakistan, particularly the Khyber Pakhtunkhwa province bordering Afghanistan, has seen a dramatic surge in militant attacks. This year alone, at least 2,414 people were killed in militancy-related violence in Pakistan, according to an Islamabad-based think tank. Pakistan blames the Afghan Taliban for sheltering the TTP, better known as the Pakistani Taliban, and tensions escalated into full-scale cross-border clashes. On October 9, 2025, while Taliban Foreign Minister Amir Khan Muttaqi was visiting India, Pakistan carried out air strikes in Kabul, apparently targeting the TTP. These strikes triggered a week of cross-border attacks that left dozens dead, before a fragile Qatar-brokered ceasefire took effect.

When the Afghan Taliban were waging an insurgency, Pakistan offered them refuge and support. Pakistan’s military establishment expected the Taliban to remain loyal once in power. But when the Taliban took over the reins in Afghanistan, the old patron-client relationship was replaced by state-to-state ties, with deep structural contradictions. The Pakistani Taliban, which oppose the merger of Pakistan’s tribal areas with Khyber Pakhtunkhwa, want their version of a strict Islamic code to be enforced and demand the release of TTP prisoners, drew inspiration from the Afghan Taliban’s triumph. The Durand Line, the colonial-era border, re-emerged as another flashpoint as Kabul has never recognised it. Besides, Pakistan’s decision to deport thousands of Afghan refugees strained ties further. What likely provoked the Pakistani establishment even more was India’s diplomatic outreach to the Taliban. By carrying out air strikes on Kabul in response to militant attacks, Pakistan appears to be setting a new precedent — holding Kabul directly accountable for cross-border militant attacks — in a move reminiscent of India’s doctrine of responding to terrorism with overwhelming force against Pakistan. Yet, the security crisis in Pakistan now engulfing its tribal areas is largely of its own making. For decades, it has followed a contradictory policy — fighting terrorism while harbouring terrorist/militant groups that were fighting its neighbouring countries. It backed the Taliban for over two decades, hoping that a Taliban-controlled Kabul would offer it strategic depth. That strategy has now backfired. If Pakistan believes that it can restore internal security by bombing Afghanistan, it is mistaken. Prolonged conflict and chaos will only deepen instability and strengthen insurgency."""




result=chain.invoke({'text':text})

print(result)


chain.get_graph().print_ascii()