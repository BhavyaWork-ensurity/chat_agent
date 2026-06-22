from langchain_ollama import ChatOllama
import json
from config import MODEL_NAME
from tools import search_memory, web_search
from langchain_classic.agents import create_tool_calling_agent, AgentExecutor
from langchain_core.prompts import ChatPromptTemplate

llm = ChatOllama(
    model=MODEL_NAME,
    temperature=0.7
)

# Tools available to the agent.
# The model can decide when to use them.
tools = [
    search_memory, web_search
]

# Defines the agent's behavior and provides
# a scratchpad for intermediate reasoning
# and tool-calling steps.
prompt = ChatPromptTemplate.from_messages(
    [
        ("system",
         "You are a helpful AI assistant with memory and web access."),
        ("human", "{input}"),
        ("placeholder", "{agent_scratchpad}")
    ]
)

# Creates a LangChain tool-calling agent
# capable of choosing and using tools.
agent = create_tool_calling_agent(
    llm,
    tools,
    prompt
)


# Executes the agent loop:
# User -> Tool Selection -> Tool Execution
# -> Reasoning -> Final Answer
agent_executor = AgentExecutor(
    agent=agent,
    tools=tools,
    verbose=True
)


# Main entry point for user conversations.
# Runs the LangChain agent and returns
# the final response.
def chat(user_input):

    result = agent_executor.invoke(
        {
            "input": user_input
        }
    )

    return result["output"]


# Uses the LLM to identify durable user facts
# that should be stored in long-term memory.
def extract_memory(
    user_input
):
    # Prompt instructing the model to convert
# user information into a structured memory.
    extraction_prompt = f"""
Extract durable user facts.

Return JSON:

{{
  "remember": true,
  "memory": "..."
}}

User:
{user_input}
"""

    result = llm.invoke(
        extraction_prompt
    )

    try:
        return json.loads(
            result.content
        )
    
    except json.JSONDecodeError:
        return None
    
