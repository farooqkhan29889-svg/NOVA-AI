

# ── STEP 1: IMPORTS ──────────────────────────────
# Same as before PLUS streamlit
import streamlit as st
from dotenv import load_dotenv
from datetime import datetime
from langchain_groq import ChatGroq
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage, ToolMessage
from langchain_community.tools import DuckDuckGoSearchRun, WikipediaQueryRun
from langchain_community.utilities import WikipediaAPIWrapper

load_dotenv()

# ── STEP 2: PAGE CONFIG ──────────────────────────
# This must be the FIRST streamlit command
# It sets the browser tab title and icon
st.set_page_config(
    page_title="NOVA AI",
    page_icon="🤖",
    layout="wide"
)

# ── STEP 3: BEAUTIFUL CSS STYLING ────────────────
# This makes NOVA look amazing with dark neon theme
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Orbitron:wght@700&family=Rajdhani:wght@400;500&display=swap');

/* Dark background for whole app */
.stApp {
    background: #080810;
    color: #e0e0e0;
}

/* Sidebar dark style */
[data-testid="stSidebar"] {
    background: #0d0d1a;
    border-right: 1px solid #1a1a3a;
}

/* NOVA title style */
h1 {
    font-family: 'Orbitron', monospace !important;
    color: #00ffcc !important;
    text-align: center;
    letter-spacing: 4px;
    text-shadow: 0 0 30px #00ffcc66;
    padding: 20px 0;
}

/* Chat message style */
[data-testid="stChatMessage"] {
    background: #0f0f20 !important;
    border: 1px solid #1a1a3a;
    border-radius: 15px;
    margin: 5px 0;
    padding: 5px;
}

/* Chat input box */
[data-testid="stChatInput"] textarea {
    background: #0f0f20 !important;
    border: 1px solid #00ffcc44 !important;
    color: #e0e0e0 !important;
    border-radius: 15px !important;
    font-family: 'Rajdhani', sans-serif !important;
}

/* Buttons */
.stButton > button {
    background: linear-gradient(135deg, #00ffcc22, #0066ff22);
    border: 1px solid #00ffcc55;
    color: #00ffcc;
    border-radius: 10px;
    width: 100%;
    transition: all 0.3s;
}
.stButton > button:hover {
    background: linear-gradient(135deg, #00ffcc44, #0066ff44);
    box-shadow: 0 0 20px #00ffcc44;
}

/* Selectbox */
[data-testid="stSelectbox"] > div > div {
    background: #0f0f20 !important;
    border: 1px solid #1a1a3a !important;
    color: #e0e0e0 !important;
}

/* Text input */
.stTextInput > div > div > input {
    background: #0f0f20 !important;
    border: 1px solid #1a1a3a !important;
    color: #e0e0e0 !important;
}

/* Subtitle text */
.subtitle {
    text-align: center;
    color: #666699;
    font-family: 'Rajdhani', sans-serif;
    font-size: 16px;
    margin-top: -15px;
    margin-bottom: 20px;
}
</style>
""", unsafe_allow_html=True)

# ── STEP 4: SESSION STATE ─────────────────────────
# Session state = memory box that survives page reruns
# Without this, everything resets on every click!

# chat_history = what gets sent to LLM (has SystemMessage)
# messages     = what gets shown on screen (user + assistant only)

if "chat_history" not in st.session_state:
    today = datetime.now().strftime("%A, %d %B %Y")
    st.session_state.chat_history = [
        SystemMessage(content=f"""You are NOVA, a powerful AI assistant made by Farooq.
Today's date is {today}.
Use DuckDuckGo for current news, weather, scores, prices.
Use Wikipedia for facts about people, history, science.
Be friendly, smart and helpful. Keep answers clear.""")
    ]

if "messages" not in st.session_state:
    # messages is a list of dicts: {"role": "user/assistant", "content": "..."}
    st.session_state.messages = []

# ── STEP 5: CREATE TOOLS AND LLM ─────────────────
# @st.cache_resource means — create this ONCE and reuse
# Without this, tools get recreated on every message!
@st.cache_resource
def load_tools_and_llm():
    search_tool    = DuckDuckGoSearchRun()
    wiki_tool      = WikipediaQueryRun(api_wrapper=WikipediaAPIWrapper())
    tools          = [search_tool, wiki_tool]
    llm            = ChatGroq(model="llama-3.1-8b-instant",temperature=0.7,api_key=st.secrets["GROQ_API_KEY"])
    llm_with_tools = llm.bind_tools(tools)
    return search_tool, wiki_tool, llm_with_tools

search_tool, wiki_tool, llm_with_tools = load_tools_and_llm()

# ── STEP 6: SIDEBAR ──────────────────────────────
# Sidebar = left panel with settings
with st.sidebar:
    st.markdown("## ⚙️ NOVA Settings")
    st.divider()

    # Model selector
    model_choice = st.selectbox(
        "🧠 AI Model",
        ["llama-3.1-8b-instant", "llama-3.3-70b-versatile", "mixtral-8x7b-32768"]
    )

    # Temperature slider
    temp = st.slider("🌡️ Temperature", 0.0, 1.0, 0.7, 0.05,
                     help="Low = focused answers, High = creative answers")

    st.divider()

    # Show stats
    total = len(st.session_state.messages)
    st.markdown(f"**💬 Total Messages:** {total}")
    st.markdown(f"**📅 Date:** {datetime.now().strftime('%d %b %Y')}")
    st.markdown(f"**⏰ Time:** {datetime.now().strftime('%I:%M %p')}")

    st.divider()

    # Clear chat button
    if st.button("🗑️ Clear Chat"):
        today = datetime.now().strftime("%A, %d %B %Y")
        st.session_state.chat_history = [
            SystemMessage(content=f"""You are NOVA, a powerful AI assistant made by Farooq.
Today's date is {today}.
Use DuckDuckGo for current news, weather, scores, prices.
Use Wikipedia for facts about people, history, science.""")
        ]
        st.session_state.messages = []
        st.rerun()  # refresh the page

    st.divider()
    st.markdown("**Built by Farooq 🚀**")

# ── STEP 7: MAIN UI ───────────────────────────────
# Title and subtitle
st.markdown("# NOVA 🤖")
st.markdown('<p class="subtitle">Next-Gen AI Assistant — Built by Farooq</p>',
            unsafe_allow_html=True)
st.divider()

# ── STEP 8: SHOW CHAT MESSAGES ────────────────────
# Loop through all saved messages and show them
# This runs every time page refreshes
for msg in st.session_state.messages:
    if msg["role"] == "user":
        # Show user message on right side
        st.chat_message("user", avatar="👤").write(msg["content"])
    else:
        # Show NOVA message on left side
        st.chat_message("assistant", avatar="🤖").write(msg["content"])

# ── STEP 9: CHAT INPUT ────────────────────────────
# st.chat_input = the input box at bottom of screen
# It returns None if user hasn't typed anything
user_input = st.chat_input("Ask NOVA anything... 💬")

# Only runs when user actually sends a message
if user_input:

    # Show user message immediately on screen
    st.chat_message("user", avatar="👤").write(user_input)

    # Save to messages list for display
    st.session_state.messages.append({
        "role": "user",
        "content": user_input
    })

    # Add to chat_history for LLM memory
    st.session_state.chat_history.append(HumanMessage(content=user_input))

    # ── STEP 10: GET NOVA REPLY ───────────────────
    with st.spinner("🤖 NOVA is thinking..."):

        # Send full history to LLM
        response = llm_with_tools.invoke(st.session_state.chat_history)

        # Did LLM want to use a tool?
        if response.tool_calls:
            st.session_state.chat_history.append(response)

            for tool_call in response.tool_calls:
                tool_name  = tool_call["name"]
                tool_input = tool_call["args"].get("query", "")

                # Show which tool is being used
                st.info(f"🔧 Using: {tool_name} → searching: {tool_input}")

                # Run the correct tool
                if "duck" in tool_name.lower():
                    result = search_tool.run(tool_input)
                else:
                    result = wiki_tool.run(tool_input)

                # Add tool result to history
                st.session_state.chat_history.append(
                    ToolMessage(content=result, tool_call_id=tool_call["id"])
                )

            # Get final answer after tool use
            final    = llm_with_tools.invoke(st.session_state.chat_history)
            ai_reply = final.content

        else:
            # No tool needed — direct answer
            ai_reply = response.content

    # ── STEP 11: SHOW AND SAVE NOVA REPLY ────────
    # Show reply on screen
    st.chat_message("assistant", avatar="🤖").write(ai_reply)

    # Save to chat_history (LLM memory)
    st.session_state.chat_history.append(AIMessage(content=ai_reply))

    # Save to messages (screen display)
    st.session_state.messages.append({
        "role": "assistant",
        "content": ai_reply
    })