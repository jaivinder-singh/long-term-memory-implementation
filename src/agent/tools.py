import requests
from langchain_community.tools import DuckDuckGoSearchRun
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv

load_dotenv()

LLM_PROVIDERS = {
    "openai": {"class": ChatOpenAI, "default_model": "gpt-4o"},
}


def create_llm(provider: str = "openai", model: str | None = None, **kwargs):
    config = LLM_PROVIDERS.get(provider)
    if config is None:
        raise ValueError(f"Unknown LLM provider: {provider}")
    model = model or config["default_model"]
    return config["class"](model=model, **kwargs)


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


TOOL_REGISTRY = {
    "search": lambda: DuckDuckGoSearchRun(region="us-en"),
    "stock_price": lambda: get_stock_price,
    "add_numbers": lambda: add_numbers,
}


def create_tools(tool_names: list[str] | None = None):
    if tool_names is None:
        tool_names = list(TOOL_REGISTRY.keys())
    unknown = set(tool_names) - set(TOOL_REGISTRY.keys())
    if unknown:
        raise ValueError(f"Unknown tools: {unknown}")
    return [TOOL_REGISTRY[name]() for name in tool_names]


llm = create_llm()
tools = create_tools()
llm_with_tools = llm.bind_tools(tools)
