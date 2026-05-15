from langgraph.store.base import embed
from langgraph.store.memory import InMemoryStore
from langchain_openai import OpenAIEmbeddings
from dotenv import load_dotenv

load_dotenv()


embedding_model = OpenAIEmbeddings(model="text-embedding-3-small")

#creating a memory store
memory = InMemoryStore(index={"embed": embedding_model, "dims": 1536})

#creating a namespace
namespace = ("users", "u1")


#Adding a new item to the memory store
memory.put(namespace, "1", {"data": "User likes pizza"})
memory.put(namespace, "2", {"data": "User likes python"})
memory.put(namespace, "3", {"data": "User likes to go for a walk"})
memory.put(namespace, "4", {"data": "User likes to play basketball"})
memory.put(namespace, "4", {"data": "User is currently learning machine learning"})



#get memory store
#print(memory.get(namespace, "1"))

#print(memory.get(namespace2, "2"))

#search memory store
items = memory.search(namespace, query="what does the user currently learning?", limit=1)

for item in items:
    print(item.value)