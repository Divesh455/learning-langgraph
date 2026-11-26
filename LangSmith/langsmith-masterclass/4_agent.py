import os
import requests
from dotenv import load_dotenv
from langchain_core.tools import tool
from langchain_core.prompts import PromptTemplate
from langchain_community.tools import DuckDuckGoSearchRun
from langchain_google_genai import ChatGoogleGenerativeAI


# Legacy imports from langchain_classic
from langchain_classic.agents import create_react_agent, AgentExecutor

load_dotenv()
os.environ['LANGCHAIN_PROJECT'] = 'ReAct Agent'

search_tool = DuckDuckGoSearchRun()

@tool
def get_weather_data(city: str) -> str:
    """ This function fetches the current weather data for a given city """
    # FIX: Restored the proper Weatherstack API endpoint structure
    url = f'https://api.apilayer.net/weatherstack/current?access_key=55c0cf5e818322c054ded7903088fd1e&query={city}'
    response = requests.get(url)
    return response.json()

llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash"
)

# Step 2: Define the ReAct prompt locally to bypass the broken online Hub pulling entirely
template = """Answer the following questions as best you can. You have access to the following tools:

{tools}

Use the following format:

Question: the input question you must answer
Thought: you should always think about what to do
Action: the action to take, should be one of [{tool_names}]
Action Input: the input to the action
Observation: the result of the action
... (this Thought/Action/Action Input/Observation can repeat N times)
Thought: I now know the final answer
Final Answer: the final answer to the original input question

Begin!

Question: {input}
Thought:{agent_scratchpad}"""

prompt = PromptTemplate.from_template(template)

# Step 3: Create the ReAct agent manually with our hardcoded local prompt
agent = create_react_agent(
    llm=llm,
    tools=[search_tool, get_weather_data],
    prompt=prompt
)

# Step 4: Wrap it with AgentExecutor
agent_executor = AgentExecutor(
    agent=agent,
    tools=[search_tool, get_weather_data],
    verbose=True,
    max_iterations=5
)

# Step 5: Invoke
response = agent_executor.invoke({"input": "What is the current temperature in Kalpana Chawla's birth city?"})
print(response)
print(response['output'])
