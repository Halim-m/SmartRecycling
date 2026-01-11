import tensorflow as tf
from pathlib import Path
from typing import Dict, Any, Optional, Tuple
import yaml
import json
from datetime import datetime
from .callbacks import get_callbacks


class Trainer:
    def __init__(
        self,
        model: tf.keras.Model,
        config: Optional[Dict[str, Any]] = None,
        config_path: str = "configs/config.yaml"
    ):
        if config is None:
            with open(config_path, 'r', encoding='utf-8') as f:
                config = yaml.safe_load(f)

        self.config = config
        self.model = model
        self.history = {}

        self.log_dir = Path(config['paths']['logs'])
        self.model_dir = Path(config['paths']['models'])
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.model_dir.mkdir(parents=True, exist_ok=True)

    def train(
        self,
        train_dataset: tf.data.Dataset,
        val_dataset: tf.data.Dataset,
        epochs: int,
        learning_rate: float,
        phase: str = 'feature_extraction',
        class_weights: Optional[Dict[int, float]] = None
    ) -> tf.keras.callbacks.History:
        print(f"\n{'='*50}")
        print(f"Eğitim Aşaması: {phase.upper()}")
        print(f"Epochs: {epochs}")
        print(f"Learning Rate: {learning_rate}")
        print(f"{'='*50}\n")

        optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)

        self.model.compile(
            optimizer=optimizer,
            loss='categorical_crossentropy',
            metrics=[
                'accuracy',
                tf.keras.metrics.Precision(name='precision'),
                tf.keras.metrics.Recall(name='recall')
            ]
        )

        callbacks = get_callbacks(
            config=self.config,
            phase=phase,
            log_dir=str(self.log_dir),
            model_dir=str(self.model_dir)
        )

        history = self.model.fit(
            train_dataset,
            validation_data=val_dataset,
            epochs=epochs,
            callbacks=callbacks,
            class_weight=class_weights,
            verbose=1
        )

        self.history[phase] = history.history
        self._save_history(phase)

        return history

    def train_two_phase(
        self,
        train_dataset: tf.data.Dataset,
        val_dataset: tf.data.Dataset,
        classifier,
        class_weights: Optional[Dict[int, float]] = None
    ) -> Dict[str, Any]:
        training_config = self.config['training']

        print("\n" + "="*60)
        print("AŞAMA 1: FEATURE EXTRACTION")
        print("="*60)

        classifier.freeze_base_model()

        history1 = self.train(
            train_dataset=train_dataset,
            val_dataset=val_dataset,
            epochs=training_config['epochs_phase1'],
            learning_rate=training_config['learning_rate'],
            phase='feature_extraction',
            class_weights=class_weights
        )

        print("\n" + "="*60)
        print("AŞAMA 2: FINE-TUNING")
        print("="*60)

        classifier.unfreeze_for_fine_tuning(unfreeze_ratio=0.3)

        history2 = self.train(
            train_dataset=train_dataset,
            val_dataset=val_dataset,
            epochs=training_config['epochs_phase2'],
            learning_rate=training_config['fine_tune_learning_rate'],
            phase='fine_tuning',
            class_weights=class_weights
        )

        final_path = self.model_dir / 'waste_classifier_final.keras'
        self.model.save(final_path)
        print(f"\nFinal model kaydedildi: {final_path}")

        return {
            'feature_extraction': history1.history,
            'fine_tuning': history2.history
        }

    def evaluate(
        self,
        test_dataset: tf.data.Dataset
    ) -> Dict[str, float]:
        print("\n" + "="*50)
        print("MODEL DEĞERLENDİRME")
        print("="*50 + "\n")

        results = self.model.evaluate(test_dataset, verbose=1)

        metrics = {}
        for name, value in zip(self.model.metrics_names, results):
            metrics[name] = float(value)
            print(f"{name}: {value:.4f}")

        return metrics

    def _save_history(self, phase: str) -> None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        history_path = self.log_dir / f"history_{phase}_{timestamp}.json"

        with open(history_path, 'w') as f:
            json.dump(self.history[phase], f, indent=2)

        print(f"History kaydedildi: {history_path}")

    def get_combined_history(self) -> Dict[str, list]:
        combined = {}

        for phase, history in self.history.items():
            for metric, values in history.items():
                if metric not in combined:
                    combined[metric] = []
                combined[metric].extend(values)

        return combined
