from langchain.llms import OpenAI, HuggingFaceHub
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate

# Define prompt template
prompt = PromptTemplate.from_template("Translate to French: {input}")

# LLM 1
llm_openai = OpenAI(model="text-davinci-003")

# LLM 2
llm_huggingface = HuggingFaceHub(repo_id="google/flan-t5-base")

# Chain examples
chain1 = LLMChain(prompt=prompt, llm=llm_openai)
chain2 = LLMChain(prompt=prompt, llm=llm_huggingface)

response1 = chain1.run("Good morning!")
response2 = chain2.run("Good morning!")

print("OpenAI:", response1)
print("HuggingFace:", response2)
