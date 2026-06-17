from chromadb import PersistentClient
from sentence_transformers import SentenceTransformer

from config import CHROMA_PATH

client = PersistentClient(path=CHROMA_PATH)

collection = client.get_or_create_collection(name = "user_memory")
embedder = SentenceTransformer("all-MiniLM-L6-v2")

def save_memory(memory_text):

    embedding = embedder.encode(memory_text).tolist()

    memory_id = str(collection.count() + 1)

    collection.add(
        ids=[memory_id],
        documents=[memory_text],
        embeddings=[embedding]
    )

def retrieve_memories(
    query,
    k=5
):

    query_embedding = embedder.encode(query).tolist()

    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=k
    )

    return results["documents"][0]

