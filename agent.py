from langchain_ollama import ChatOllama
from langchain_core.messages import HumanMessage
from memory import retrieve_memories
import json
from config import MODEL_NAME

llm = ChatOllama(
    model=MODEL_NAME,
    temperature=0.7
)

# Retrieves relevant memories and uses them as context
# to generate a response from the language model.
def chat(user_input):

    memories = retrieve_memories(
        user_input
    )

    memory_block = "\n".join(
        memories
    )

    prompt = f"""
Relevant memories:

{memory_block}

User:
{user_input}
"""

    response = llm.invoke(
        [HumanMessage(content=prompt)]
    )

    return response.content

# Extracts long-term user facts from the input and
# returns them as a JSON object for storage.
def extract_memory(
    user_input
):

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
    
    except:
        return None
    
