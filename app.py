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

# --- 3. Create single agent (no team = no transfer errors) ---
@st.cache_resource(show_spinner="Initializing AI...")
def get_agent():
    # Use llama-3.1-8b-instant - Groq's most reliable tool-caller
    model = Groq(
        id="llama-3.1-8b-instant",
        api_key=GROQ_API_KEY,
        temperature=0.1
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
                historical_prices=True,  # enables 3-year data
                key_financial_ratios=True,
                income_statements=True,
            ),
            DuckDuckGo(
                search=True,
                news=True,
                fixed_max_results=5,  # prevents max_results hallucination
                timeout=15
            )
        ],
        instructions=[
            "For 'average return over X years': ALWAYS use historical_prices tool first",
            "Calculate CAGR as: (end_price / start_price) ** (1/years) - 1",
            "Never estimate or assume growth rates - calculate from actual data",
            "Present financial data in markdown tables",
            "Always include sources with URLs for news",
            "If a tool fails, explain what data is missing instead of guessing"
        ],
        show_tool_calls=True,
        markdown=True,
        add_datetime_to_instructions=True,
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
                response = agent.run(prompt, stream=False)
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
