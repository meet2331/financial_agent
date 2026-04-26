# app.py - Finance Agent (Streamlit Cloud - stable version)
import streamlit as st

st.set_page_config(
    page_title="Finance Agent",
    page_icon="💹",
    layout="wide"
)

st.title("💹 Finance Agent")
st.caption("Real stock data • analyst ratings • news • historical returns")

# --- 1. Load API keys from Streamlit Secrets ---
try:
    GROQ_API_KEY = st.secrets["GROQ_API_KEY"]
    PHI_API_KEY = st.secrets["PHI_API_KEY"]
except KeyError as e:
    st.error(f"🚨 Missing secret: {e}")
    st.info("Add in Settings → Secrets:\nGROQ_API_KEY = \"gsk_...\"\nPHI_API_KEY = \"phi_...\"")
    st.stop()

# --- 2. Imports ---
try:
    from phi.agent import Agent
    from phi.model.groq import Groq
    from phi.tools.yfinance import YFinanceTools
    from phi.tools.duckduckgo import DuckDuckGo
except Exception as e:
    st.error(f"Import error: {e}")
    st.stop()

@st.cache_resource(show_spinner="Initializing AI...")
def get_agent():
    # Switch to 70b - same free tier but handles larger context better
    # AND set max_tokens to limit response size
    model = Groq(
        id="llama-3.3-70b-versatile",  # 12k TPM on free tier vs 6k for 8b
        api_key=GROQ_API_KEY,
        temperature=0.1,
        max_tokens=1500  # cap output
    )
    
    return Agent(
        name="Finance Agent",
        model=model,
        tools=[
            YFinanceTools(
                stock_price=True,
                analyst_recommendations=True,
                stock_fundamentals=True,
                company_news=True,
                historical_prices=True,
                key_financial_ratios=False,  # disable heavy tools
                income_statements=False,
            ),
            DuckDuckGo(
                search=True,
                news=True,
                fixed_max_results=3,  # was 5, now 3
                timeout=10
            )
        ],
        instructions=[
            "CRITICAL: For historical prices, ALWAYS use period='3y' and interval='1mo' - never daily data",
            "For returns: calculate CAGR = (end/start)^(1/years)-1 using monthly data only",
            "Summarize news in 2 sentences max per article",
            "Use tables, keep responses under 300 words",
            "Never return raw JSON or full price history"
        ],
        show_tool_calls=False,  # was True - saves ~2000 tokens
        markdown=True,
        add_datetime_to_instructions=False,  # saves tokens
        num_history_responses=0,  # don't send chat history to model
    )

try:
    agent = get_agent()
except Exception as e:
    st.error(f"Failed to start agent: {e}")
    st.stop()

# --- 4. Sidebar ---
with st.sidebar:
    st.header("Try these")
    st.markdown("- `AAPL average return past 3 years`")
    st.markdown("- `NVDA analyst recommendations`")
    st.markdown("- `TSLA latest news`")
    st.markdown("- `MSFT stock fundamentals`")
    
    if st.button("Clear chat"):
        st.session_state.messages = []
        st.rerun()

# --- 5. Chat history ---
if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# --- 6. Chat input ---
if prompt := st.chat_input("Ask about any stock..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Fetching data..."):
            try:
                response = agent.run(prompt,stream=False,extra_instructions="Use minimal data. For historical prices: period=3y, interval=1mo only.")
                answer = response.content
                
                st.markdown(answer)
                
                # Show tool calls in expander
                if hasattr(response, 'messages'):
                    tool_calls = [m for m in response.messages if getattr(m, 'tool_calls', None)]
                    if tool_calls:
                        with st.expander(f"🔧 Tools used ({len(tool_calls)})"):
                            for tc in tool_calls:
                                for call in tc.tool_calls:
                                    st.code(f"{call['function']['name']}\n{call['function']['arguments']}", language="json")
                
                st.session_state.messages.append({"role": "assistant", "content": answer})
                
            except Exception as e:
                error_msg = f"**Error:** {str(e)}\n\nTry rephrasing or ask for a simpler query."
                st.error(error_msg)
                st.session_state.messages.append({"role": "assistant", "content": error_msg})
