## CREATE AGENT THAT USES TOOLS, DEFAULT, AND CUSTOM TOOLS
from dotenv import load_dotenv
load_dotenv()

## BASIC CHATBOT THAT HAS EVERYTHING (TOOLS, MEMORY, YOUTUBE) IN IT.

from typing import Annotated
from typing_extensions import TypedDict
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langchain.chat_models import init_chat_model

# TOOLS
from langchain_tavily import TavilySearch
from langchain_community.tools import YouTubeSearchTool

search_tool= TavilySearch(max_results=2)
youtube_search_tool = YouTubeSearchTool()


tools= [search_tool, youtube_search_tool]

#1. Create s StateGraph - Structure of our Chatbot
class State(TypedDict):
    # in this case messages gets appended instead of overwriting them
    messages: Annotated[list, add_messages]

# FOUNDATION - A plot
graph_builder = StateGraph(State)

## model - ENGINE or BRAIN
model= init_chat_model(model="gemini-2.0-flash", model_provider="google_genai")
model_with_tools= model.bind_tools(tools)

def chatbot(state: State):
    return {"messages": [model_with_tools.invoke(state["messages"])]}

# Adding NODES, EDGES - WALLS for building
graph_builder.add_node("chatbot", chatbot)

#Entry AND EXIT
graph_builder.add_edge(START, "chatbot")
graph_builder.add_edge("chatbot", END)

# COMPILE EVERYTHING - ASSEMBLE
graph= graph_builder.compile()


# RUN THE CHATBOT
def stream_graph_updates(user_input: str):
    for event in graph.stream({"messages": [{"role": "user", "content": user_input}]}):
        for value in event.values():
            print("ASSISTANT: ", value["messages"][-1].content)


# while True:
#     try:
#         user_input = input("USER: ")
#         if user_input.lower() in ["quit", "exit", "q"]:
#             print("Goodbye!!")
#             break
#         stream_graph_updates(user_input)
#     except:
#         user_input = "Explain the Langgraph, head should say what you are explain"
#         print("User: " + user_input)
#         stream_graph_updates(user_input)
#         break

