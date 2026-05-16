from pydantic import BaseModel, Field


class MemoryDecision(BaseModel):
    should_write: bool = Field(description="whether to store an item in the memory store")
    memories: list[str] = Field(default_factory=list, description="the memories to store")