from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage,SystemMessage,AIMessage
from langgraph.prebuilt import create_react_agent
from langgraph.checkpoint.memory import MemorySaver
from langchain_core.tools import tool, Tool
from langchain_community.tools import DuckDuckGoSearchRun, WikipediaQueryRun
from langchain_community.utilities import WikipediaAPIWrapper

load_dotenv()

llm = ChatGroq(model="llama-3.1-8b-instant",temperature=0.7)

system_message = "I AM NOVA AI HOW CAN I HELP YOU:"

sreach_tool = DuckDuckGoSearchRun()
wiki_tool = WikipediaQueryRun(api_wrapper=WikipediaAPIWrapper())

tools = [sreach_tool, wiki_tool]

memory = MemorySaver()

agent = create_react_agent(
    model=llm,
    tools=tools,
    checkpointer=memory,
    prompt=system_message
)

config = {"configurable": {"thread_id": "1"}}

print("NOVA Agent is ready! Type 'quit' to exit.\n")

if __name__ == "__main__":
    response = agent.invoke({"messages": [("user", "Hello! Who are you?")]}, config=config)
    print(response['messages'][-1].content)
