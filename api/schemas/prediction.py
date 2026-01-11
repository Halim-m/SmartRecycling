from pydantic import BaseModel, Field
from typing import Dict, List, Optional


class RecyclingInfo(BaseModel):
    bin_color: str = Field(..., description="Atık kutusu rengi")
    recyclable: bool = Field(..., description="Geri dönüştürülebilir mi")
    instructions: str = Field(..., description="Geri dönüşüm talimatları")
    environmental_impact: str = Field(..., description="Çevresel etki bilgisi")


class TopKPrediction(BaseModel):
    class_name: str = Field(..., alias="class", description="Sınıf ismi (İngilizce)")
    class_tr: str = Field(..., description="Sınıf ismi (Türkçe)")
    confidence: float = Field(..., ge=0, le=1, description="Güven skoru")

    class Config:
        populate_by_name = True


class PredictionResponse(BaseModel):
    predicted_class: str = Field(..., description="Tahmin edilen sınıf (İngilizce)")
    predicted_class_tr: str = Field(..., description="Tahmin edilen sınıf (Türkçe)")
    confidence: float = Field(..., ge=0, le=1, description="Güven skoru")
    recycling_info: RecyclingInfo = Field(..., description="Geri dönüşüm bilgisi")
    top_k_predictions: List[TopKPrediction] = Field(..., description="En yüksek K tahmin")
    all_probabilities: Dict[str, float] = Field(..., description="Tüm sınıf olasılıkları")
    processing_time_ms: float = Field(..., description="İşlem süresi (ms)")


class HealthResponse(BaseModel):
    status: str = Field(..., description="Servis durumu")
    model_loaded: bool = Field(..., description="Model yüklü mü")
    version: str = Field(..., description="API versiyonu")


class ModelInfo(BaseModel):
    status: str = Field(..., description="Model durumu")
    model_path: Optional[str] = Field(None, description="Model dosya yolu")
    input_shape: Optional[tuple] = Field(None, description="Giriş boyutu")
    num_classes: Optional[int] = Field(None, description="Sınıf sayısı")
    classes: Optional[List[str]] = Field(None, description="Sınıf isimleri")
    total_params: Optional[int] = Field(None, description="Toplam parametre sayısı")


class CategoryInfo(BaseModel):
    name: str = Field(..., description="Kategori ismi (İngilizce)")
    name_tr: str = Field(..., description="Kategori ismi (Türkçe)")
    bin_color: str = Field(..., description="Kutu rengi")
    recyclable: bool = Field(..., description="Geri dönüştürülebilir")
    instructions: str = Field(..., description="Talimatlar")


class CategoriesResponse(BaseModel):
    total: int = Field(..., description="Toplam kategori sayısı")
    categories: List[CategoryInfo] = Field(..., description="Kategori listesi")


class ErrorResponse(BaseModel):
    detail: str = Field(..., description="Hata detayı")
    error_code: Optional[str] = Field(None, description="Hata kodu")
