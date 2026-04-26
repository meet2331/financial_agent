# app.py - Finance Multi-Agent for Streamlit Cloud
import streamlit as st

st.set_page_config(
    page_title="Finance Team Agent",
    page_icon="💹",
    layout="wide"
)

st.title("💹 Finance Team Agent")
st.caption("Ask about any stock ticker - analyst ratings + latest news")

# --- 1. Get API keys from Streamlit Secrets ---
try:
    GROQ_API_KEY = st.secrets["GROQ_API_KEY"]
    PHI_API_KEY = st.secrets["PHI_API_KEY"]  # phidata needs this for tracing
except KeyError as e:
    st.error(f"🚨 Missing secret: {e}")
    st.info("Go to your app → Settings → Secrets and add:\n\nGROQ_API_KEY = \"gsk_...\"\nPHI_API_KEY = \"phi_...\"")
    st.stop()

# --- 2. Import after keys are confirmed ---
try:
    from phi.agent import Agent
    from phi.model.groq import Groq
    from phi.tools.yfinance import YFinanceTools
    from phi.tools.duckduckgo import DuckDuckGo
except Exception as e:
    st.error(f"Import failed: {e}")
    st.stop()

# --- 3. Create agents (cached so they load once) ---
@st.cache_resource(show_spinner="Loading AI models...")
def get_team():
    # Pass keys directly - no os.environ needed
    model = Groq(id="llama-3.3-70b-versatile", api_key=GROQ_API_KEY)
    
    web_search_agent = Agent(
        name="Web Search Agent",
        role="Search the web for current information",
        model=model,
        tools=[DuckDuckGo()],
        instructions=["Always include sources with URLs"],
        show_tool_calls=True,
        markdown=True,
    )

    finance_agent = Agent(
        name="Finance AI Agent",
        model=model,
        tools=[
            YFinanceTools(
                stock_price=True,
                analyst_recommendations=True,
                stock_fundamentals=True,
                company_news=True
            )
        ],
        instructions=["Use tables to display the data"],
        show_tool_calls=True,
        markdown=True,
    )

    team = Agent(
        team=[web_search_agent, finance_agent],
        model=model,
        instructions=[
            "Always include sources",
            "Use tables to display financial data",
            "For stock queries, get both analyst recommendations and latest news"
        ],
        show_tool_calls=True,
        markdown=True,
    )
    return team

try:
    team = get_team()
except Exception as e:
    st.error(f"Failed to initialize agents: {e}")
    st.stop()

# --- 4. UI ---
with st.sidebar:
    st.header("Examples")
    st.markdown("- `NVDA analyst recommendations`")
    st.markdown("- `Latest news for TSLA`")
    st.markdown("- `AAPL stock price and fundamentals`")
    st.markdown("- `Compare MSFT and GOOGL`")

# Initialize chat
if "messages" not in st.session_state:
    st.session_state.messages = []

# Show history
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# Chat input
if prompt := st.chat_input("Enter a stock ticker or question..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Analyzing..."):
            try:
                response = team.run(prompt)
                answer = response.content
                st.markdown(answer)
                
                # Show tool calls
                if hasattr(response, 'messages'):
                    tool_msgs = [m for m in response.messages if getattr(m, 'role', '') == 'tool']
                    if tool_msgs:
                        with st.expander(f"🔧 {len(tool_msgs)} tool calls"):
                            for i, tm in enumerate(tool_msgs, 1):
                                st.code(f"Tool {i}: {tm.content[:500]}...")
                
                st.session_state.messages.append({"role": "assistant", "content": answer})
            except Exception as e:
                err = f"Error: {str(e)}"
                st.error(err)
                st.session_state.messages.append({"role": "assistant", "content": err})
