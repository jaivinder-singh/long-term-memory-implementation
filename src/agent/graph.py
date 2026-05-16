from langchain_core.messages import SystemMessage, ToolMessage
from langgraph.graph import END, START, StateGraph
from states import AgentState
from tools import tools, llm_with_tools
from ltm_memory import search_memory, create_memory_store
from prompt_template import SYSTEM_PROMPT_TEMPLATE
from langchain_core.runnables import RunnableConfig

tools_by_name = {tool.name: tool for tool in tools}

ltm_store = create_memory_store()


def call_model(state: AgentState):
    memory = state.get("memory", [])
    system_prompt = SYSTEM_PROMPT_TEMPLATE.format(user_details_content=memory)
    messages = [SystemMessage(content=system_prompt)] + state["messages"]
    response = llm_with_tools.invoke(messages)
    return {"messages": [response]}


def fetch_memory(state: AgentState, config: RunnableConfig):
    user_id = config.get("configurable", {}).get("user_id", "u1")
    last_message = state["messages"][-1]
    query = last_message.content if hasattr(last_message, "content") else str(last_message)
    results = search_memory(user_id, query)
    return {"memory": results}


def call_tools(state: AgentState):
    last_message = state["messages"][-1]
    results = []
    for tool_call in last_message.tool_calls:
        tool = tools_by_name[tool_call["name"]]
        result = tool.invoke(tool_call["args"])
        results.append(
            ToolMessage(content=str(result), tool_call_id=tool_call["id"])
        )
    return {"messages": results}


def should_continue(state: AgentState):
    last_message = state["messages"][-1]
    if last_message.tool_calls:
        return "continue"
    return "end"


workflow = StateGraph[AgentState, None, AgentState, AgentState](AgentState)

workflow.add_node("fetch_memory", fetch_memory)
workflow.add_edge(START, "fetch_memory")
workflow.add_edge("fetch_memory", "agent")

workflow.add_node("agent", call_model)
workflow.add_node("tools", call_tools)

workflow.add_conditional_edges(
    "agent",
    should_continue,
    {
        "continue": "tools",
        "end": END,
    },
)

workflow.add_edge("tools", "agent")

chatbot = workflow.compile(store=ltm_store)
