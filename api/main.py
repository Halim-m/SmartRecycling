from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
from pathlib import Path
import yaml

from api.routes import classification, health
from src.inference.predictor import WastePredictor

predictor: WastePredictor = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global predictor

    print("="*50)
    print("Smart Recycling API Başlatılıyor...")
    print("="*50)

    config_path = Path("configs/config.yaml")
    if config_path.exists():
        with open(config_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
        model_path = config['api']['model_path']
    else:
        model_path = "saved_models/waste_classifier_final.keras"

    model_file = Path(model_path)
    if model_file.exists():
        try:
            predictor = WastePredictor(model_path=str(model_file))
            print(f"Model yüklendi: {model_path}")
        except Exception as e:
            print(f"Model yükleme hatası: {e}")
            predictor = WastePredictor.__new__(WastePredictor)
            predictor.model = None
            predictor.config = config if config_path.exists() else {}
            predictor.classes = config.get('data', {}).get('classes', [])
            predictor.class_names_tr = config.get('data', {}).get('class_names_tr', {})
            predictor.recycling_info = config.get('recycling_info', {})
    else:
        print(f"Model dosyası bulunamadı: {model_path}")
        print("API model olmadan başlatılıyor...")

        if config_path.exists():
            with open(config_path, 'r', encoding='utf-8') as f:
                config = yaml.safe_load(f)
            predictor = WastePredictor.__new__(WastePredictor)
            predictor.model = None
            predictor.config = config
            predictor.classes = config.get('data', {}).get('classes', [])
            predictor.class_names_tr = config.get('data', {}).get('class_names_tr', {})
            predictor.recycling_info = config.get('recycling_info', {})
            predictor.image_size = tuple(config.get('data', {}).get('image_size', [224, 224]))
        else:
            predictor = None

    print("API hazır!")
    print("="*50)

    yield

    print("API kapatılıyor...")


app = FastAPI(
    title="Smart Recycling API",
    description="""
    ## AI Destekli Atık Sınıflandırma Sistemi

    Bu API, görüntü tabanlı atık sınıflandırma hizmeti sunar.
    EfficientNetV2-S transfer learning modeli kullanılarak eğitilmiştir.

    ### Desteklenen Kategoriler
    * Plastik, Kağıt, Karton, Cam, Metal
    * Organik, E-Atık, Tekstil, Diğer

    ### Kullanım
    1. `/api/v1/classify` endpoint'ine görüntü yükleyin
    2. Tahmin sonucu ve geri dönüşüm bilgisi alın

    ---
    BM5113 Bilgisayarla Görme Dersi Projesi
    """,
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
    lifespan=lifespan
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(
    health.router,
    prefix="/api/v1",
    tags=["Health"]
)

app.include_router(
    classification.router,
    prefix="/api/v1",
    tags=["Classification"]
)


@app.get("/", tags=["Root"])
async def root():
    return {
        "name": "Smart Recycling API",
        "version": "1.0.0",
        "description": "AI-powered waste classification system",
        "docs": "/docs",
        "health": "/api/v1/health"
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "api.main:app",
        host="0.0.0.0",
        port=8000,
        reload=True
    )
