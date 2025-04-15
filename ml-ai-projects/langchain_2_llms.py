from langchain.llms import OpenAI
from langchain.agents import load_tools, initialize_agent
from langchain.agents.agent_types import AgentType

llm = OpenAI(temperature=0)
tools = load_tools(["serpapi", "llm-math"], llm=llm)
agent = initialize_agent(tools, llm, agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION)

response = agent.run("What is 24 * 7 plus the current year?")
print("Response:", response)
