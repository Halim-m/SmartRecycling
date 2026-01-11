import tensorflow as tf
import numpy as np
from PIL import Image
from pathlib import Path
from typing import Dict, Any, Union, Optional, List
import yaml
import time


class WastePredictor:
    def __init__(
        self,
        model_path: Optional[str] = None,
        config_path: str = "configs/config.yaml"
    ):
        with open(config_path, 'r', encoding='utf-8') as f:
            self.config = yaml.safe_load(f)

        self.image_size = tuple(self.config['data']['image_size'])
        self.classes = self.config['data']['classes']
        self.class_names_tr = self.config['data']['class_names_tr']
        self.recycling_info = self.config['recycling_info']

        if model_path is None:
            model_path = self.config['api']['model_path']

        self.model_path = model_path
        self.model = None

        if Path(model_path).exists():
            self.load_model(model_path)

    def load_model(self, model_path: str) -> None:
        print(f"Model yükleniyor: {model_path}")
        self.model = tf.keras.models.load_model(model_path)
        print("Model başarıyla yüklendi.")

        dummy_input = np.zeros((1, *self.image_size, 3), dtype=np.float32)
        self.model.predict(dummy_input, verbose=0)

    def preprocess(
        self,
        image: Union[Image.Image, np.ndarray, str]
    ) -> np.ndarray:
        if isinstance(image, str):
            image = Image.open(image)

        if isinstance(image, np.ndarray):
            image = Image.fromarray(image.astype(np.uint8))

        if image.mode != 'RGB':
            image = image.convert('RGB')

        image = image.resize(self.image_size, Image.Resampling.LANCZOS)

        img_array = np.array(image, dtype=np.float32)

        img_array = tf.keras.applications.efficientnet_v2.preprocess_input(img_array)

        return img_array

    def predict(
        self,
        image: Union[Image.Image, np.ndarray, str],
        top_k: int = 3
    ) -> Dict[str, Any]:
        if self.model is None:
            raise ValueError("Model yüklenmedi.")

        start_time = time.time()

        processed = self.preprocess(image)
        input_tensor = np.expand_dims(processed, axis=0)

        predictions = self.model.predict(input_tensor, verbose=0)[0]

        predicted_idx = int(np.argmax(predictions))
        predicted_class = self.classes[predicted_idx]
        confidence = float(predictions[predicted_idx])

        all_probabilities = {
            self.classes[i]: float(predictions[i])
            for i in range(len(self.classes))
        }

        top_k_indices = np.argsort(predictions)[-top_k:][::-1]
        top_k_predictions = [
            {
                'class': self.classes[i],
                'class_tr': self.class_names_tr.get(self.classes[i], self.classes[i]),
                'confidence': float(predictions[i])
            }
            for i in top_k_indices
        ]

        recycling = self.recycling_info.get(predicted_class, {})

        processing_time = (time.time() - start_time) * 1000

        return {
            'predicted_class': predicted_class,
            'predicted_class_tr': self.class_names_tr.get(predicted_class, predicted_class),
            'confidence': confidence,
            'recycling_info': {
                'bin_color': recycling.get('bin_color', 'bilinmiyor'),
                'recyclable': recycling.get('recyclable', False),
                'instructions': recycling.get('instructions', ''),
                'environmental_impact': recycling.get('environmental_impact', '')
            },
            'top_k_predictions': top_k_predictions,
            'all_probabilities': all_probabilities,
            'processing_time_ms': round(processing_time, 2)
        }

    def predict_batch(
        self,
        images: List[Union[Image.Image, np.ndarray, str]]
    ) -> List[Dict[str, Any]]:
        if self.model is None:
            raise ValueError("Model yüklenmedi.")

        processed_images = np.array([
            self.preprocess(img) for img in images
        ])

        start_time = time.time()

        predictions = self.model.predict(processed_images, verbose=0)

        total_time = (time.time() - start_time) * 1000
        per_image_time = total_time / len(images)

        results = []
        for i, preds in enumerate(predictions):
            predicted_idx = int(np.argmax(preds))
            predicted_class = self.classes[predicted_idx]
            confidence = float(preds[predicted_idx])

            recycling = self.recycling_info.get(predicted_class, {})

            results.append({
                'predicted_class': predicted_class,
                'predicted_class_tr': self.class_names_tr.get(predicted_class, predicted_class),
                'confidence': confidence,
                'recycling_info': {
                    'bin_color': recycling.get('bin_color', 'bilinmiyor'),
                    'recyclable': recycling.get('recyclable', False),
                    'instructions': recycling.get('instructions', ''),
                    'environmental_impact': recycling.get('environmental_impact', '')
                },
                'processing_time_ms': round(per_image_time, 2)
            })

        return results

    def get_model_info(self) -> Dict[str, Any]:
        if self.model is None:
            return {'status': 'not_loaded'}

        return {
            'status': 'loaded',
            'model_path': str(self.model_path),
            'input_shape': self.image_size,
            'num_classes': len(self.classes),
            'classes': self.classes,
            'total_params': self.model.count_params()
        }
