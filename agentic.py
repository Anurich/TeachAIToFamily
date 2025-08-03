from typing import TypedDict, List
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain.prompts import PromptTemplate
from langchain.tools import tool
from dotenv import load_dotenv
from langgraph.graph import StateGraph, END
from langchain_core.messages import AIMessage
import json
from langchain_chroma import Chroma
from langchain_core.messages import ToolMessage, HumanMessage

from langchain.globals import set_debug

set_debug(False)

load_dotenv()
#state
class State(TypedDict):
    messages: List[str]
    

    

class supervisedAgent:
    def __init__(self):
        self.llm = ChatOpenAI(model="gpt-4o-mini")
        
    
    def define_prompt(self):
        self.prompt = """
            Your a supervised agent, and you have access to two other agents, which are `retreiveragent` and `webagent`.
            Your job is to route to the user query to appropriate agent.
            # Input question: {question}
            # Specialised Agent:
                - retreiveragent:  call this agent when question is about the company information. 
                
            # Instruction:
                * If question is about acknowledgement and greeting for example: (hi, thank you, hello) in that case you must respond from your own knowledge.
                * All the general question that is not related to specialised agent you must answer it from your own knowledge.
                * Make sure  not to hallucinate and only made decision based on facts. 
                
            # Output JSON format only:
                * Do not provide any explaination, only provide valid JSON.
                ```json
                {{
                    "answer": "answer in html format if user question is not related to  specialised agent if question is related to specialised agent return name: `retreiveragent` "
                }}
                ```
        """
        return self.prompt
    
    
    def create_supervisor(self):
        prompt = self.define_prompt()
        prompt_template = PromptTemplate.from_template(prompt)
        chain = prompt_template | self.llm
        return chain 


embedding = OpenAIEmbeddings(model="text-embedding-3-large")
retriever = Chroma(
    embedding_function=embedding, persist_directory="chromadb",collection_name="rag"
    )
@tool
def pdf_chatter( query):
    """
        this tool must be called when question is about company or question related to pdf.
    """
    
    print("*"*100)
    docs = retriever.similarity_search(query, k=2)
    return docs
tools = dict()
tools[pdf_chatter.name] = pdf_chatter
class Agnet2:
    def __init__(self):
        # retreiver data 

        self.llm = ChatOpenAI(model="gpt-4o-mini")        
        self.agent2prompt()
    
    def agent2prompt(self):
        self.agent2_prompt = """
            You are an expert analyst answering questions about company information using document retrieval.

            TASK: Answer the user's question using the provided context from the PDF documents.
            * Only call the tool once do not call the tool more than one.
            USER QUESTION: {question}

            INSTRUCTIONS:
            - Use ONLY the information provided in the context
            - Structure your response with clear HTML formatting (headings, lists, emphasis)
            - Include specific data, numbers, and facts when available
            - If context is insufficient, explain what additional information would be needed
            - Be accurate and cite relevant details from the source material

            Provide your response in this JSON format:
            {{
                "answer": "Comprehensive HTML-formatted answer with proper structure and emphasis"
            }}
        """

    
    def finalAgent(self):
        llm_with_tool = self.llm.bind_tools([pdf_chatter])
        chain = PromptTemplate.from_template(self.agent2_prompt) | llm_with_tool    
        return chain 
    
        
        

def node_1(state: State):
    question = state["messages"][-1]
    sp_agent = supervisedAgent()
    sp1 = sp_agent.create_supervisor()
    response = sp1.invoke({"question": question})
    output = response.content.replace("```json", "").replace("```", "")
    output = json.loads(output)
    state["messages"].append(AIMessage(content = output["answer"]))
    return state


def conditional_check(state: State):
    ot = state["messages"][-1].content
    if ot == "retreiveragent":
        return "call_next"
    else:
        return "__end__"
    
#1353594633
def agent_2(state: State):
    agent = Agnet2()
    agent = agent.finalAgent()
    # we focus only on question
    for message in state["messages"][:-1]:
        if "user" in message:
            _, question = message
    output = agent.invoke({"question": question})
    state["messages"].append(output)
    return state
    

def should_continue(state: State):
    if len(state["messages"][-1].tool_calls) > 0:
        return "continue"
    else:
        return "__end__"

def tool_node(state: State):
    for tn in state["messages"][-1].tool_calls:
        tname = tn["name"]
        args = tn["args"]
        result = tools[tname].invoke(args)
        context = ToolMessage(
            content=result, 
            name= tname,
            tool_call_id = tn["id"]
        )
    state["messages"].append(context)
    return state    

def final_answer(state: State):
    question = None 
    context = None 
    for messages in state["messages"][:-1]:
        if "user" in messages:
            question = messages[1]
        
        if type(messages) == ToolMessage:
            context = messages
    
    
    pmpt = """
        User question: {question}
        context: {context}
        
        Provide the answer based on question and context provided.
        JSON Structure format:
        * Do not provide the explaination and always format the data into provided format below:
        ```json{{
            "answer": html format only.
        }}```
    """
    
    chain = PromptTemplate.from_template(pmpt) | ChatOpenAI(model="gpt-4o-mini")
    output = chain.invoke({"question": question, "context": context})
    state["messages"].append(json.loads(output.content.replace("```json", "").replace("```", "")))
    return state


graph = StateGraph(state_schema=State)

graph.add_node("boss", node_1)
graph.add_node("node_2",agent_2)
graph.add_node("tool_node", tool_node)
graph.add_node("final_answer", final_answer)
graph.set_entry_point("boss")
graph.add_conditional_edges(
    "boss", conditional_check, 
    {
        "call_next": "node_2",
        "__end__": END
    }
)
graph.add_conditional_edges("node_2", should_continue, {
    "continue": "tool_node",
    "__end__": END
})
graph.add_edge("tool_node", "final_answer")
graph.add_edge("final_answer", END)

workflow = graph.compile()

import os 
png_graph_bytes = workflow.get_graph().draw_mermaid_png()
output_file_path = "my_langgraph.png"
with open(output_file_path, "wb") as f:
    f.write(png_graph_bytes)

print(f"Graph saved as '{output_file_path}' in {os.getcwd()}")

ot = workflow.invoke({
    "messages": [("user", "can you tell me about company status of tata motors?")]
})

print(ot["messages"][-1])



