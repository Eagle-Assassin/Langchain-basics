from langchain_openai import OpenAI
from langchain_core.prompts import PromptTemplate
from dotenv import load_dotenv

load_dotenv()

#Initialise the LLms
llm= OpenAI(model_name='gpt-3.5-turbo',temperature=0.7)

#Create a prompt template
prompt=PromptTemplate(template="Suggest a catchy blog title about {topic}",
                      input_variables=['topic'])

#Define the input
topic = input("Enter a topic")


#Format the prompt manully using Prompt template
formatted_prompt=prompt.format(topic=topic)

blog_title = llm.invoke(formatted_prompt)

print("Generated Blog tiles:", blog_title)