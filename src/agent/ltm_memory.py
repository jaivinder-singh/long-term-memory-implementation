from langgraph.store.base import embed
from langgraph.store.memory import InMemoryStore
from langchain_openai import OpenAIEmbeddings
from dotenv import load_dotenv

load_dotenv()


embedding_model = OpenAIEmbeddings(model="text-embedding-3-small")

#creating a memory store
memory = InMemoryStore(index={"embed": embedding_model, "dims": 1536})

#creating a namespace
def get_namespace(user_id: str) -> tuple[str, ...]:
    return ("users", user_id, "details")


#Adding a new item to the memory store
namespace = get_namespace("u1")
memory.put(namespace, "1", {"data": "User likes pizza"})
memory.put(namespace, "1", {"data": "User name is Jaivinder"})
memory.put(namespace, "2", {"data": "User likes python"})
memory.put(namespace, "3", {"data": "User likes to go for a walk"})
memory.put(namespace, "4", {"data": "User likes to play basketball"})
memory.put(namespace, "4", {"data": "User is currently learning machine learning"})



#get memory store
#print(memory.get(namespace, "1"))

#print(memory.get(namespace2, "2"))

#search memory store

def search_memory(user_id: str, query: str):
    items = memory.search(get_namespace(user_id), query=query, limit=1)
    return [item.value for item in items]
