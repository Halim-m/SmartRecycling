from fastapi import APIRouter
from api.schemas.prediction import HealthResponse

router = APIRouter()


@router.get(
    "/health",
    response_model=HealthResponse,
    summary="Sağlık kontrolü",
    description="API servisinin çalışıp çalışmadığını kontrol eder"
)
async def health_check():
    from api.main import predictor

    model_loaded = predictor is not None and predictor.model is not None

    return HealthResponse(
        status="healthy",
        model_loaded=model_loaded,
        version="1.0.0"
    )


@router.get(
    "/",
    summary="Root endpoint",
    description="API hoş geldin mesajı"
)
async def root():
    return {
        "message": "Smart Recycling API",
        "description": "AI-powered waste classification system",
        "docs_url": "/docs",
        "version": "1.0.0"
    }
