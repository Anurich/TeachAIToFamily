from typing import TypedDict, List
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain.prompts import PromptTemplate
from langchain.tools import tool
from dotenv import load_dotenv
import json
from langchain_chroma import Chroma

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
                - webagent: call this agent when the question is about the current information, or recent information.
                
            # Instruction:
                * If question is about acknowledgement and greeting for example: (hi, thank you, hello) in that case you must respond from your own knowledge.
                * All the general question that is not related to specialised agent you must answer it from your own knowledge.
                * Make sure  not to hallucinate and only made decision based on facts. 
                
            # JSON output format:
             {{
                 "answer": html format response.
             }}
        """
        return self.prompt
    
    
    def create_supervisor(self):
        prompt = self.define_prompt()
        prompt_template = PromptTemplate.from_template(prompt)
        chain = prompt_template | self.llm
        return chain 


class Agnet2:
    def __init__(self):
        embedding = OpenAIEmbeddings(model="text-embedding-3-large")
        # retreiver data 
        self.retriever = Chroma(
            embedding_function=embedding, persist_directory="chromadb",collection_name="rag"
        )
        self.llm = ChatOpenAI(model="gpt-4o-mini")
        self.agemt2prompt()
    
    def agent2prompt(self):
        self.agent2_prompt = """
            Given the user question, your job is to answer the user question.  
            You have access to retreiver tool name `pdf_chatter`. 
            You must call that tool only onece and respond to user. 
            * user question: {question}
            # JSON format.
            {{
                "answer": answer in html format
            }}      
        """
    @tool
    def pdf_chatter(self, query):
        """
            this tool must be called when question is about company or question related to pdf.
        """
        print("*"*100)
        docs = self.retriever.similarity_search(query, k=2)
        return docs
    
    def finalAgent(self):
        llm_with_tool = self.llm.bind_tools([self.pdf_chatter])
        chain = PromptTemplate.from_template(self.agent2_prompt) | llm_with_tool    
        return chain 
    
        
        

sup_agent  = supervisedAgent()
Agent_1 = sup_agent.create_supervisor()    

