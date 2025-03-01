from typing import Annotated
from typing_extensions import TypedDict
from langchain_core.messages import AnyMessage, SystemMessage, HumanMessage, AIMessage 
from langgraph.graph.message import add_messages
from langgraph.checkpoint.memory import MemorySaver
from langchain_groq import ChatGroq
from langgraph.graph import END, StateGraph, START
import os
from dotenv import load_dotenv
load_dotenv()

os.environ["GROQ_API_KEY"]=os.getenv("GROQ_API_KEY")
os.environ["LANGSMITH_API_KEY"]=os.getenv("LANGCHAIN_API_KEY")

model = ChatGroq(model="qwen-2.5-32b")


class State(TypedDict):
    messages: Annotated[list[AnyMessage], add_messages]
    #memory: Annotated[MemorySaver, MemorySaver.save_memory]

def graph_builder():
    graph = StateGraph(State)

    def title_generator(state:State):
        """
        Generate the title for the blog.
        """
        system_prompt = SystemMessage(content= "You are a title generator")
        return{"messages": [model.invoke([system_prompt] + state["messages"])]}

    def generate_content(state:State):
        """
        Generate the content for the blog
        """
        system_prompt = SystemMessage(content= "You are a content generator for the blog")
        return{"messages": [model.invoke([system_prompt] + state["messages"])]}

    graph.add_node("title_generation", title_generator)
    graph.add_node("content_generation", generate_content)
    
    # Define graph edges
    graph.add_edge("title_generation", "content_generation")
    graph.add_edge("content_generation", END)
    graph.add_edge(START, "title_generation")

    # Compile the graph into an executable agent
    agent = graph.compile()
    
    return agent


#if __name__ == "__main__":
"""def main():
    agent = graph_builder()
        # initialized initial state
    initial_state = State(
        messages=[HumanMessage(content="Generative AI")]
    )
    response = agent.invoke(initial_state) 
 
    print(f"'\033[4mTitle\033[0m': {response["messages"][1].content}")
    print(f"'\033[4mContent\033[0m': {response["messages"][2].content}")
#   for message in response["messages"]:
#        print(message.content)"""

agent=graph_builder()