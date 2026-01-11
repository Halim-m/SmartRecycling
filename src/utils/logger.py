import logging
import sys
from pathlib import Path
from datetime import datetime
from typing import Optional


def setup_logger(
    name: str = 'smart_recycling',
    log_dir: str = 'logs',
    level: int = logging.INFO,
    log_to_file: bool = True
) -> logging.Logger:
    logger = logging.getLogger(name)
    logger.setLevel(level)

    logger.handlers = []

    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )

    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(level)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    if log_to_file:
        log_path = Path(log_dir)
        log_path.mkdir(parents=True, exist_ok=True)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        file_handler = logging.FileHandler(
            log_path / f'{name}_{timestamp}.log',
            encoding='utf-8'
        )
        file_handler.setLevel(level)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    return logger


class TrainingLogger:
    def __init__(
        self,
        log_dir: str = 'logs',
        experiment_name: Optional[str] = None
    ):
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)

        if experiment_name is None:
            experiment_name = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.experiment_name = experiment_name

        self.logger = setup_logger(
            name=f'training_{experiment_name}',
            log_dir=str(self.log_dir)
        )

        self.metrics_history = []

    def log_epoch(
        self,
        epoch: int,
        metrics: dict,
        phase: str = 'train'
    ) -> None:
        metrics_str = ' | '.join([
            f'{k}: {v:.4f}' for k, v in metrics.items()
        ])
        self.logger.info(f"[{phase.upper()}] Epoch {epoch}: {metrics_str}")

        self.metrics_history.append({
            'epoch': epoch,
            'phase': phase,
            **metrics
        })

    def log_info(self, message: str) -> None:
        self.logger.info(message)

    def log_warning(self, message: str) -> None:
        self.logger.warning(message)

    def log_error(self, message: str) -> None:
        self.logger.error(message)

    def log_model_summary(self, model) -> None:
        self.logger.info("Model Summary:")
        model.summary(print_fn=self.logger.info)
