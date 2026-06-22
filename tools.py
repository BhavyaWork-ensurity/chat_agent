from langchain.tools import tool
from memory import retrieve_memories

from duckduckgo_search import DDGS


@tool
def search_memory(query: str) -> str:
    """
    Search the user's long-term memory.
    Use this when answering questions about
    user preferences, goals, history, and facts.
    """

    memories = retrieve_memories(query)

    return "\n".join(memories)


@tool
def web_search(query: str) -> str:
    """
    Search the web for current information.
    Use this for news, recent events,
    current facts, and internet searches.
    """

    results = []

    with DDGS() as ddgs:

        for r in ddgs.text(
            query,
            max_results=5
        ):

            results.append(
                f"Title: {r['title']}\n"
                f"Body: {r['body']}\n"
                f"URL: {r['href']}"
            )

    return "\n\n".join(results)