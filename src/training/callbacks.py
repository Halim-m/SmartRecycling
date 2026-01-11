import tensorflow as tf
from pathlib import Path
from typing import Dict, Any, List
from datetime import datetime


def get_callbacks(
    config: Dict[str, Any],
    phase: str = 'feature_extraction',
    log_dir: str = 'logs',
    model_dir: str = 'saved_models'
) -> List[tf.keras.callbacks.Callback]:
    callbacks = []
    training_config = config.get('training', {})
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    checkpoint_path = Path(model_dir) / f"checkpoint_{phase}_{timestamp}.keras"
    callbacks.append(
        tf.keras.callbacks.ModelCheckpoint(
            filepath=str(checkpoint_path),
            monitor='val_accuracy',
            mode='max',
            save_best_only=True,
            save_weights_only=False,
            verbose=1
        )
    )

    callbacks.append(
        tf.keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=training_config.get('early_stopping_patience', 5),
            restore_best_weights=True,
            verbose=1
        )
    )

    callbacks.append(
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=training_config.get('reduce_lr_factor', 0.5),
            patience=training_config.get('reduce_lr_patience', 3),
            min_lr=1e-7,
            verbose=1
        )
    )

    tensorboard_dir = Path(log_dir) / f"tensorboard_{phase}_{timestamp}"
    callbacks.append(
        tf.keras.callbacks.TensorBoard(
            log_dir=str(tensorboard_dir),
            histogram_freq=1,
            write_graph=True,
            write_images=False,
            update_freq='epoch',
            profile_batch=0
        )
    )

    csv_path = Path(log_dir) / f"training_{phase}_{timestamp}.csv"
    callbacks.append(
        tf.keras.callbacks.CSVLogger(
            filename=str(csv_path),
            separator=',',
            append=False
        )
    )

    return callbacks


class TrainingProgressCallback(tf.keras.callbacks.Callback):
    def __init__(self, total_epochs: int):
        super().__init__()
        self.total_epochs = total_epochs

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}

        print(f"\n{'='*60}")
        print(f"Epoch {epoch + 1}/{self.total_epochs} Tamamlandı")
        print(f"{'='*60}")

        for metric, value in logs.items():
            if 'val_' in metric:
                continue
            val_metric = f'val_{metric}'
            train_val = f"{value:.4f}"
            if val_metric in logs:
                train_val += f" / {logs[val_metric]:.4f}"
            print(f"  {metric}: {train_val}")

        if hasattr(self.model.optimizer, 'lr'):
            lr = float(tf.keras.backend.get_value(self.model.optimizer.lr))
            print(f"  learning_rate: {lr:.2e}")

        print()


class GradCAMCallback(tf.keras.callbacks.Callback):
    def __init__(
        self,
        validation_data,
        output_dir: str,
        frequency: int = 5
    ):
        super().__init__()
        self.validation_data = validation_data
        self.output_dir = Path(output_dir)
        self.frequency = frequency
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def on_epoch_end(self, epoch, logs=None):
        if (epoch + 1) % self.frequency != 0:
            return
        pass
