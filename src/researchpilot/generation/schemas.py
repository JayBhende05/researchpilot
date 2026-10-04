from pydantic import BaseModel, Field


class Citation(BaseModel):
    chunk_id: str
    document_id: str
    text: str
    score: float | None = None
    metadata: dict = Field(default_factory=dict)


class RAGResponse(BaseModel):
    question: str
    answer: str
    citations: list[Citation] = Field(default_factory=list)
from pydantic import BaseModel, Field


class Citation(BaseModel):
    chunk_id: str
    document_id: str
    text: str
    score: float | None = None
    metadata: dict = Field(default_factory=dict)


class GeneratedAnswer(BaseModel):
    answer: str
    citation_ids: list[str] = Field(default_factory=list)


class RAGResponse(BaseModel):
    question: str
    answer: str
    citations: list[Citation] = Field(default_factory=list)
