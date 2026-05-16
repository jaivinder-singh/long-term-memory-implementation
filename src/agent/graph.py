import uuid
from langchain_core.messages import SystemMessage, ToolMessage
from langchain_openai import ChatOpenAI
from langgraph.graph import END, START, StateGraph
from states import AgentState
from tools import tools, llm_with_tools
from ltm_memory import search_memory, create_memory_store, get_namespace
from memory_decision import MemoryDecision
from prompt_template import SYSTEM_PROMPT_TEMPLATE
from langchain_core.runnables import RunnableConfig

tools_by_name = {tool.name: tool for tool in tools}

ltm_store = create_memory_store()
extractor_llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.0)
memory_extractor = extractor_llm.with_structured_output(MemoryDecision)


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


def remember_memory(state: AgentState, config: RunnableConfig):
    user_id = config.get("configurable", {}).get("user_id", "u1")
    namespace = get_namespace(user_id)

    conversation = "\n".join(
        f"{m.type}: {m.content}"
        for m in state["messages"]
        if hasattr(m, "content") and m.content
    )

    response = memory_extractor.invoke([
        SystemMessage(
            content=(
                "Extract LONG-TERM memories from the conversation.\n"
                "Only store stable, user-specific info (identity, preferences, ongoing projects).\n"
                "Do NOT store transient info.\n"
                "Return should_write=false if nothing is worth storing.\n"
                "Each memory should be a short atomic sentence."
            )
        ),
        {"role": "user", "content": conversation},
    ])

    if response.should_write:
        for mem in response.memories:
            ltm_store.put(namespace, str(uuid.uuid4()), {"data": mem})

    return

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
workflow.add_node("agent", call_model)
workflow.add_node("tools", call_tools)
workflow.add_node("remember_memory", remember_memory)

workflow.add_edge(START, "fetch_memory")
workflow.add_edge("fetch_memory", "agent")
workflow.add_edge("tools", "agent")

workflow.add_conditional_edges(
    "agent",
    should_continue,
    {
        "continue": "tools",
        "end": "remember_memory",
    },
)

workflow.add_edge("remember_memory", END)

chatbot = workflow.compile(store=ltm_store)
