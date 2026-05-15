import requests
from langchain_community.tools import DuckDuckGoSearchRun
from langchain_core.tools import tool

from langchain_openai import ChatOpenAI
from dotenv import load_dotenv

load_dotenv()

llm = ChatOpenAI(model="gpt-4o")

search_tool = DuckDuckGoSearchRun(region="us-en")


@tool
def get_stock_price(symbol: str) -> dict:
    """
    Fetch latest stock price for a given symbol (e.g. 'AAPL', 'TSLA') 
    using Alpha Vantage with API key in the URL.
    """
    url = f"https://www.alphavantage.co/query?function=GLOBAL_QUOTE&symbol={symbol}&apikey=C9PE94QUEW9VWGFM"
    r = requests.get(url)
    return r.json()


@tool
def add_numbers(a: float, b: float) -> float:
    """Add two numbers and return the result."""
    return a + b


tools = [search_tool, get_stock_price, add_numbers]
llm_with_tools = llm.bind_tools(tools)
