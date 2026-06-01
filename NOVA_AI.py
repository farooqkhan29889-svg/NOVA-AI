# from dotenv import load_dotenv
# from langchain_groq import ChatGroq
# from langchain.agents import create_react_agent, AgentExecutor
# from langchain.tools import Tool
# from langchain_community.tools import DuckDuckGoSearchRun, WikipediaQueryRun
# from langchain_community.utilities import WikipediaAPIWrapper
# from langchain.memory import ConversationBufferMemory
# from langchain.prompts import PromptTemplate

# load_dotenv()

# llm = ChatGroq(model="llama-3.1-8b-instant", temperature=0.7)

# sreach_tool = DuckDuckGoSearchRun()

# wiki_tool = WikipediaAPIWrapper(api_wrapper=WikipediaAPIWrapper())

# tools = [
#     Tool(
#         name="web sreach",
#         func=sreach_tool,
#         discription="Use this for current news, weather, scores, prices, recent events."
# ),
#     Tool(
#         name="wikipidia sreach",
#         func=wiki_tool,
#         discription="Use this for current news, weather, scores, prices, recent events."
             
#     )
# ]

# template = """You are NOVA, a powerful AI assistant made by Farooq.
# You have access to the following tools:

# {tools}

# Use this format:

# Question: the input question you must answer
# Thought: think about what to do
# Action: the action to take, must be one of [{tool_names}]
# Action Input: the input to the action
# Observation: the result of the action
# ... (repeat Thought/Action/Action Input/Observation as needed)
# Thought: I now know the final answer
# Final Answer: the final answer to the question 

# Chat History:
# {chat_history}

# Question: {input}
# Thought: {agent_scratchpad}"""

# prompt = PromptTemplate(
#     input_variables=["tools", "tool_names", "chat_history", "input", "agent_scratchpad"],
#     template=template
# )

# memory = ConversationBufferMemory(
#     memory_key="chetHistry",
#     return_messege=False
# )

# create_agent = create_react_agent(llm=llm,tools=tools,prompt=prompt)

# agent_executor = AgentExecutor(
#     create_agent=create_agent,
#     tools=tools,
#     memory=memory,
#     verbose=True,
#     handle_parsing_errors=True,
#     max_iterations=5
    
# )

# print("NOVA Agent ready! Type 'quit' to exit.\n")

# # ── Step 6: Chat Loop ──
# while True:
#     user_input = input("You: ")
#     if user_input.lower() == "quit":
#         break
#     response = agent_executor.invoke({"input": user_input})
#     print(f"\nNOVA: {response['output']}\n")

from dotenv import load_dotenv
from datetime import datetime
from langchain_groq import ChatGroq
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from langchain_community.tools import DuckDuckGoSearchRun, WikipediaQueryRun
from langchain_community.utilities import WikipediaAPIWrapper

load_dotenv()

# ── Step 1: Tools ──
search_tool = DuckDuckGoSearchRun()
wiki_tool   = WikipediaQueryRun(api_wrapper=WikipediaAPIWrapper())
tools       = [search_tool, wiki_tool]


# ── Step 2: LLM with tools bound ──
llm = ChatGroq(model="llama-3.1-8b-instant", temperature=0.7)
llm_with_tools = llm.bind_tools(tools)

today = datetime.now().strftime("%A, %d %B %Y")

# ── Step 3: Memory ──
chat_history = [
    SystemMessage(content="""You are NOVA, a powerful AI assistant made by Farooq.
You have access to these tools{today}:
- DuckDuckGoSearch: for current news, weather, scores, prices
- WikipediaQueryRun: for facts about people, history, science

Always use tools when you need current or factual information.
Be friendly, smart and helpful.""")
]

print("NOVA Agent ready! Type 'quit' to exit.\n")

# ── Step 4: Chat Loop ──
while True:
    user_input = input("You: ")
    if user_input.lower() == "quit":
        break

    # Add user message
    chat_history.append(HumanMessage(content=user_input))

    # Get response from LLM
    response = llm_with_tools.invoke(chat_history)

    # ── Step 5: Handle Tool Calls ──
    if response.tool_calls:
        # LLM wants to use a tool
        chat_history.append(response)

        for tool_call in response.tool_calls:
            tool_name  = tool_call["name"]
            tool_input = tool_call["args"].get("query", "")

            print(f"🔧 Using tool: {tool_name}")
            print(f"🔍 Searching: {tool_input}")

            # Run the correct tool
            if "duckduckgo" in tool_name.lower() or "search" in tool_name.lower():
                result = search_tool.run(tool_input)
            else:
                result = wiki_tool.run(tool_input)

            # Add tool result to history
            from langchain_core.messages import ToolMessage
            chat_history.append(
                ToolMessage(content=result, tool_call_id=tool_call["id"])
            )

        # Get final answer after tool use
        final_response = llm_with_tools.invoke(chat_history)
        ai_reply = final_response.content
        chat_history.append(AIMessage(content=ai_reply))

    else:
        # No tool needed — direct answer
        ai_reply = response.content
        chat_history.append(AIMessage(content=ai_reply))

    print(f"\nNOVA: {ai_reply}\n")