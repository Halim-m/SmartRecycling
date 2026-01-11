from fastapi import APIRouter, UploadFile, File, HTTPException, Query
from PIL import Image
import io
from typing import Optional

from api.schemas.prediction import (
    PredictionResponse,
    ModelInfo,
    CategoriesResponse,
    CategoryInfo,
    TopKPrediction
)

router = APIRouter()


@router.post(
    "/classify",
    response_model=PredictionResponse,
    summary="Atık sınıflandır",
    description="Yüklenen görüntüyü analiz ederek atık türünü belirler"
)
async def classify_waste(
    file: UploadFile = File(..., description="Sınıflandırılacak görüntü dosyası"),
    top_k: int = Query(3, ge=1, le=9, description="Döndürülecek en yüksek K tahmin")
):
    from api.main import predictor

    if predictor is None or predictor.model is None:
        raise HTTPException(
            status_code=503,
            detail="Model yüklenmedi. Lütfen daha sonra tekrar deneyin."
        )

    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(
            status_code=400,
            detail="Geçersiz dosya tipi. Lütfen bir görüntü dosyası yükleyin."
        )

    try:
        contents = await file.read()
        image = Image.open(io.BytesIO(contents))

        result = predictor.predict(image, top_k=top_k)

        return PredictionResponse(
            predicted_class=result['predicted_class'],
            predicted_class_tr=result['predicted_class_tr'],
            confidence=result['confidence'],
            recycling_info=result['recycling_info'],
            top_k_predictions=[
                TopKPrediction(
                    **{"class": p['class']},
                    class_tr=p['class_tr'],
                    confidence=p['confidence']
                )
                for p in result['top_k_predictions']
            ],
            all_probabilities=result['all_probabilities'],
            processing_time_ms=result['processing_time_ms']
        )

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Sınıflandırma hatası: {str(e)}"
        )


@router.get(
    "/model/info",
    response_model=ModelInfo,
    summary="Model bilgisi",
    description="Yüklü model hakkında bilgi döndürür"
)
async def get_model_info():
    from api.main import predictor

    if predictor is None:
        return ModelInfo(status="not_initialized")

    info = predictor.get_model_info()
    return ModelInfo(**info)


@router.get(
    "/categories",
    response_model=CategoriesResponse,
    summary="Kategoriler",
    description="Desteklenen atık kategorilerini listeler"
)
async def get_categories():
    from api.main import predictor

    if predictor is None:
        raise HTTPException(
            status_code=503,
            detail="Servis hazır değil"
        )

    categories = []
    for class_name in predictor.classes:
        recycling = predictor.recycling_info.get(class_name, {})
        categories.append(
            CategoryInfo(
                name=class_name,
                name_tr=predictor.class_names_tr.get(class_name, class_name),
                bin_color=recycling.get('bin_color', 'bilinmiyor'),
                recyclable=recycling.get('recyclable', False),
                instructions=recycling.get('instructions', '')
            )
        )

    return CategoriesResponse(
        total=len(categories),
        categories=categories
    )
