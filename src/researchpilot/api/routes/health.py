from fastapi import APIRouter, Depends

from researchpilot.api.deps import Components, get_components
from researchpilot.api.schemas import HealthResponse

router = APIRouter()


@router.get("/health", response_model=HealthResponse)
def health(
    components: Components = Depends(get_components),
) -> HealthResponse:
    return HealthResponse(
        status="ok",
        chunks=components.chunk_count,
    )
