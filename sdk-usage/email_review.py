import os
from typing import TypedDict

from langchain_openai import ChatOpenAI
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.types import Command, interrupt

# DeepSeek provides an OpenAI-compatible interface, just change base_url in ChatOpenAI
llm = ChatOpenAI(
    base_url="https://api.deepseek.com/v1",
    model="deepseek-chat",
    api_key=os.environ["API_KEY"],
)

# Shared state for the entire graph
class State(TypedDict):
    topic: str       # email subject
    draft: str       # current draft
    feedback: str    # revision notes from last round
    approved: bool   # whether approval passed

# Node 1: Generate email draft based on topic (and previous feedback)
def write_email(state: State) -> dict:
    if state.get("feedback"):
        # Already have a previous draft and revision notes, rewrite based on them
        prompt = (
            f"Below is the previous email draft:\n{state['draft']}\n\n"
            f"Revision notes: {state['feedback']}\n\n"
            "Please rewrite based on the draft and notes above. Keep it within 3 sentences."
        )
    else:
        # First-time generation
        prompt = f"Write a short email with the subject: {state['topic']}. Keep it within 3 sentences."
    resp = llm.invoke(prompt)
    return {"draft": resp.content}

# Node 2: Pause and wait for human review
# Human either approves or gives revision notes for the model to rewrite
def human_review(state: State) -> dict:
    decision = interrupt({"draft": state["draft"]})
    if decision == "approve":
        return {"approved": True}
    return {"feedback": decision, "approved": False}

# Conditional edge: go to END if approved, otherwise loop back to write_email for revision
def route_after_review(state: State) -> str:
    return END if state["approved"] else "write_email"

builder = StateGraph(State)
# Create graph nodes
# START and END are predefined special nodes in LangGraph, representing the start and end of the graph
# No need to create them manually
builder.add_node("write_email", write_email)
builder.add_node("human_review", human_review)

# Create edges to connect nodes
# From START node, first go to write_email node
builder.add_edge(START, "write_email")
# From write_email node, go to human_review node
builder.add_edge("write_email", "human_review")
# From human_review node, use the return value of route_after_review
# to decide which node to go to next
builder.add_conditional_edges("human_review", route_after_review)

# InMemorySaver stores execution progress so that interrupt can pause and resume
graph = builder.compile(checkpointer=InMemorySaver())
config = {"configurable": {"thread_id": "email-1"}}

# First invocation, runs until interrupt and automatically pauses
# Access the model's output draft field via result["__interrupt__"]
result = graph.invoke({"topic": "Wishing the team a happy weekend"}, config=config)
print("=== First draft ===")
print(result["__interrupt__"][0].value["draft"])

# Simulate human feedback requesting more casual tone
result = graph.invoke(Command(resume="Too formal, make it more casual and relaxed"), config=config)
print("\n=== Second draft (rewritten based on feedback) ===")
print(result["__interrupt__"][0].value["draft"])

# Simulate human approval
result = graph.invoke(Command(resume="approve"), config=config)
print("\n=== Approved, final email ===")
print(result["draft"])