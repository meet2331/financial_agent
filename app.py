import streamlit as st

st.set_page_config(page_title="Finance Team Agent", page_icon="💹", layout="wide")
st.title("💹 Finance Team Agent")

try:
    GROQ_API_KEY = st.secrets["GROQ_API_KEY"]
    PHI_API_KEY = st.secrets["PHI_API_KEY"]
except KeyError as e:
    st.error(f"Missing secret: {e}")
    st.stop()

from phi.agent import Agent
from phi.model.groq import Groq
from phi.tools.yfinance import YFinanceTools
from phi.tools.duckduckgo import DuckDuckGo

@st.cache_resource
def get_team():
    model = Groq(id="llama-3.3-70b-versatile", api_key=GROQ_API_KEY)

    # FIX: fixed_max_results prevents Groq from passing max_results
    web = Agent(
        name="Web Search Agent",
        model=model,
        tools=[DuckDuckGo(search=True, news=True, fixed_max_results=5, timeout=15)],
        instructions=[
            "Always include sources",
            "Call only ONE tool at a time",
            "Never pass max_results parameter"
        ],
        show_tool_calls=True,
        markdown=True,
    )

    finance = Agent(
        name="Finance AI Agent",
        model=model,
        tools=[YFinanceTools(stock_price=True, analyst_recommendations=True,
                            stock_fundamentals=True, company_news=True)],
        instructions=["Use tables to display the data"],
        show_tool_calls=True,
        markdown=True,
    )

    return Agent(
        team=[web, finance],
        model=model,
        instructions=["Use tables", "Always include sources"],
        show_tool_calls=True,
        markdown=True,
    )

team = get_team()

if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

if prompt := st.chat_input("Ask about a stock..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Analyzing..."):
            try:
                response = team.run(prompt)
                st.markdown(response.content)
                st.session_state.messages.append({"role": "assistant", "content": response.content})
            except Exception as e:
                st.error(f"Tool error: {e}")
                # Fallback: retry without web search
                if "tool_use_failed" in str(e).lower():
                    st.info("Retrying with finance data only...")
                    fallback = team.run(f"{prompt} (use only yfinance data, no web search)")
                    st.markdown(fallback.content)
