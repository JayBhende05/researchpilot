from pydantic import BaseModel, Field, field_validator

from researchpilot.generation.schemas import Citation


class ChatRequest(BaseModel):
    question: str = Field(max_length=1000)

    @field_validator("question")
    @classmethod
    def not_blank(cls, value: str) -> str:
        value = value.strip()

        if not value:
            raise ValueError("question must not be blank")

        return value


class LatencyMs(BaseModel):
    retrieval: float
    generation: float
    total: float


class ChatResponse(BaseModel):
    question: str
    answer: str
    citations: list[Citation] = Field(default_factory=list)
    latency_ms: LatencyMs


class HealthResponse(BaseModel):
    status: str
    chunks: int
