import streamlit as st
from dotenv import load_dotenv
import os

load_dotenv()

st.set_page_config(page_title="Finance Agent", page_icon="💹", layout="wide")
st.title("💹 Finance Team Agent")

# --- Get keys safely ---
GROQ_KEY = st.secrets.get("GROQ_API_KEY") or os.getenv("GROQ_API_KEY")
PHI_KEY = st.secrets.get("PHI_API_KEY") or os.getenv("PHI_API_KEY")

if not GROQ_KEY:
    st.error("🚨 GROQ_API_KEY missing. Add it in Streamlit Secrets.")
    st.stop()

os.environ["GROQ_API_KEY"] = GROQ_KEY
os.environ["PHI_API_KEY"] = PHI_KEY or ""

# --- Import AFTER keys are set (so crash shows in UI) ---
try:
    from phi.agent import Agent
    from phi.model.groq import Groq
    from phi.tools.yfinance import YFinanceTools
    from phi.tools.duckduckgo import DuckDuckGo
except Exception as e:
    st.error(f"Import error: {e}")
    st.stop()

@st.cache_resource
def get_team():
    web = Agent(
        name="Web Search",
        model=Groq(id="llama-3.3-70b-versatile"),
        tools=[DuckDuckGo()],
        instructions=["Always include sources"],
        show_tool_calls=True,
        markdown=True,
    )
    finance = Agent(
        name="Finance",
        model=Groq(id="llama-3.3-70b-versatile"),
        tools=[YFinanceTools(stock_price=True, analyst_recommendations=True, 
                            stock_fundamentals=True, company_news=True)],
        instructions=["Use tables"],
        show_tool_calls=True,
        markdown=True,
    )
    return Agent(team=[web, finance], model=Groq(id="llama-3.3-70b-versatile"),
                 instructions=["Use tables", "Always include sources"],
                 show_tool_calls=True, markdown=True)

try:
    team = get_team()
except Exception as e:
    st.error(f"Agent init failed: {e}")
    st.stop()

query = st.text_input("Ask about any stock:", "NVDA analyst recommendations")
if st.button("Run"):
    with st.spinner("Thinking..."):
        resp = team.run(query)
        st.markdown(resp.content)
