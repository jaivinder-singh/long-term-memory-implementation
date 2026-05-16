from langgraph.store.memory import InMemoryStore
from langchain_openai import OpenAIEmbeddings
from dotenv import load_dotenv

load_dotenv()

EMBEDDING_CONFIGS = {
    "openai": {"class": OpenAIEmbeddings, "default_model": "text-embedding-3-small"},
}

STORE_DIMS = {
    "text-embedding-3-small": 1536,
    "text-embedding-3-large": 3072,
}


def create_embedding_model(provider: str = "openai", model: str | None = None):
    config = EMBEDDING_CONFIGS.get(provider)
    if config is None:
        raise ValueError(f"Unknown embedding provider: {provider}")
    model = model or config["default_model"]
    return config["class"](model=model)


def create_memory_store(
    store_type: str = "in_memory",
    embedding_provider: str = "openai",
    embedding_model: str | None = None,
    **kwargs,
):
    if store_type == "in_memory":
        emb = create_embedding_model(embedding_provider, embedding_model)
        model_name = embedding_model or EMBEDDING_CONFIGS[embedding_provider]["default_model"]
        dims = STORE_DIMS.get(model_name, 1536)
        return InMemoryStore(index={"embed": emb, "dims": dims})
    elif store_type == "postgres":
        from langgraph.checkpoint.postgres import PostgresSaver
        conn_string = kwargs.get("conn_string")
        if not conn_string:
            raise ValueError("conn_string is required for postgres store")
        saver = PostgresSaver.from_conn_string(conn_string)
        saver.setup()
        return saver
    else:
        raise ValueError(f"Unknown store type: {store_type}")


memory = create_memory_store()


def get_namespace(user_id: str) -> tuple[str, ...]:
    return ("users", user_id, "details")


def seed_memory(store=None):
    store = store or memory
    namespace = get_namespace("u1")
    store.put(namespace, "1", {"data": "User name is Jaivinder"})
    store.put(namespace, "2", {"data": "User likes python"})
    store.put(namespace, "3", {"data": "User likes to go for a walk"})
    store.put(namespace, "4", {"data": "User likes to play basketball"})
    store.put(namespace, "5", {"data": "User is currently learning machine learning"})
    store.put(namespace, "6", {"data": "User likes pizza"})


seed_memory()


def search_memory(user_id: str, query: str, store=None):
    store = store or memory
    items = store.search(get_namespace(user_id), query=query, limit=1)
    return [item.value for item in items]
