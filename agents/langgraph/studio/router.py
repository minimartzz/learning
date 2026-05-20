import os

from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import MessagesState
from langgraph.prebuilt import ToolNode, tools_condition

load_dotenv()


# Define the LLM provider
llm = ChatGroq(model="llama-3.1-8b-instant", api_key=os.getenv("GROQ_API_KEY", ""))


# Define the Tools
def multiply(a: int, b: int) -> int:
    """
    Multiply a and b together

    Args:
        a (int): Integer 1
        b (int): Integer 2

    Returns:
        int: Multiplication of both integers
    """
    return a * b


llm_with_tools = llm.bind_tools([multiply])


# Define the State
class MessagesState(MessagesState):
    pass


# Node
def tool_calling_llm(state: MessagesState):
    return {"messages": [llm_with_tools.invoke(state["messages"])]}


# Extending with tool as a node
builder = StateGraph(MessagesState)
builder.add_node("tool_calling_llm", tool_calling_llm)
builder.add_node("tools", ToolNode([multiply]))

builder.add_edge(START, "tool_calling_llm")
# If the latest message (result) is a tool call -> tools_condition routes to tools
# If the latest message (result) is not a tool call -> tools_condition routes to END
builder.add_conditional_edges("tool_calling_llm", tools_condition)
builder.add_edge("tools", END)
graph = builder.compile()
