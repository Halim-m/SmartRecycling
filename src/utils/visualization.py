import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import List, Dict, Any, Optional
import tensorflow as tf


def plot_training_history(
    history: Dict[str, List[float]],
    save_path: Optional[str] = None,
    figsize: tuple = (14, 5)
) -> plt.Figure:
    fig, axes = plt.subplots(1, 2, figsize=figsize)

    ax1 = axes[0]
    if 'accuracy' in history:
        ax1.plot(history['accuracy'], label='Eğitim', linewidth=2)
    if 'val_accuracy' in history:
        ax1.plot(history['val_accuracy'], label='Doğrulama', linewidth=2)
    ax1.set_title('Model Doğruluğu', fontsize=14, fontweight='bold')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Doğruluk')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    ax2 = axes[1]
    if 'loss' in history:
        ax2.plot(history['loss'], label='Eğitim', linewidth=2)
    if 'val_loss' in history:
        ax2.plot(history['val_loss'], label='Doğrulama', linewidth=2)
    ax2.set_title('Model Kaybı', fontsize=14, fontweight='bold')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Kayıp')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Grafik kaydedildi: {save_path}")

    return fig


def plot_confusion_matrix(
    cm: np.ndarray,
    class_names: List[str],
    save_path: Optional[str] = None,
    figsize: tuple = (12, 10),
    normalize: bool = True
) -> plt.Figure:
    if normalize:
        cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        cm_display = cm_normalized
        fmt = '.2f'
        title = 'Normalize Edilmiş Confusion Matrix'
    else:
        cm_display = cm
        fmt = 'd'
        title = 'Confusion Matrix'

    fig, ax = plt.subplots(figsize=figsize)

    sns.heatmap(
        cm_display,
        annot=True,
        fmt=fmt,
        cmap='Blues',
        xticklabels=class_names,
        yticklabels=class_names,
        ax=ax,
        cbar_kws={'label': 'Oran' if normalize else 'Sayı'}
    )

    ax.set_title(title, fontsize=14, fontweight='bold', pad=20)
    ax.set_xlabel('Tahmin Edilen Sınıf', fontsize=12)
    ax.set_ylabel('Gerçek Sınıf', fontsize=12)

    plt.setp(ax.get_xticklabels(), rotation=45, ha='right')
    plt.setp(ax.get_yticklabels(), rotation=0)

    plt.tight_layout()

    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Confusion matrix kaydedildi: {save_path}")

    return fig


def plot_class_distribution(
    class_counts: Dict[str, int],
    save_path: Optional[str] = None,
    figsize: tuple = (12, 6)
) -> plt.Figure:
    fig, ax = plt.subplots(figsize=figsize)

    classes = list(class_counts.keys())
    counts = list(class_counts.values())

    colors = plt.cm.viridis(np.linspace(0, 0.8, len(classes)))

    bars = ax.bar(classes, counts, color=colors, edgecolor='black', linewidth=0.5)

    ax.set_title('Veri Seti Sınıf Dağılımı', fontsize=14, fontweight='bold')
    ax.set_xlabel('Sınıf')
    ax.set_ylabel('Görüntü Sayısı')

    for bar, count in zip(bars, counts):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + max(counts) * 0.01,
            str(count),
            ha='center',
            va='bottom',
            fontsize=10
        )

    plt.setp(ax.get_xticklabels(), rotation=45, ha='right')
    ax.grid(axis='y', alpha=0.3)

    plt.tight_layout()

    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Dağılım grafiği kaydedildi: {save_path}")

    return fig


def plot_sample_predictions(
    images: List[np.ndarray],
    true_labels: List[str],
    pred_labels: List[str],
    confidences: List[float],
    save_path: Optional[str] = None,
    figsize: tuple = (15, 10)
) -> plt.Figure:
    n = len(images)
    cols = min(4, n)
    rows = (n + cols - 1) // cols

    fig, axes = plt.subplots(rows, cols, figsize=figsize)
    if rows == 1 and cols == 1:
        axes = [[axes]]
    elif rows == 1:
        axes = [axes]
    elif cols == 1:
        axes = [[ax] for ax in axes]

    for i, (img, true_label, pred_label, conf) in enumerate(
        zip(images, true_labels, pred_labels, confidences)
    ):
        row = i // cols
        col = i % cols
        ax = axes[row][col]

        if img.max() <= 1.0:
            img = (img * 255).astype(np.uint8)
        ax.imshow(img)
        ax.axis('off')

        correct = true_label == pred_label
        color = 'green' if correct else 'red'
        title = f"Gerçek: {true_label}\nTahmin: {pred_label}\nGüven: {conf:.2%}"
        ax.set_title(title, fontsize=10, color=color)

    for i in range(n, rows * cols):
        row = i // cols
        col = i % cols
        axes[row][col].axis('off')

    plt.suptitle('Örnek Tahminler', fontsize=14, fontweight='bold')
    plt.tight_layout()

    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Örnek tahminler kaydedildi: {save_path}")

    return fig


def generate_gradcam(
    model: tf.keras.Model,
    image: np.ndarray,
    class_idx: int,
    layer_name: Optional[str] = None
) -> np.ndarray:
    if layer_name is None:
        for layer in reversed(model.layers):
            if 'conv' in layer.name.lower():
                layer_name = layer.name
                break

    if layer_name is None:
        raise ValueError("Konvolüsyon katmanı bulunamadı")

    grad_model = tf.keras.Model(
        inputs=model.input,
        outputs=[model.get_layer(layer_name).output, model.output]
    )

    with tf.GradientTape() as tape:
        conv_output, predictions = grad_model(np.expand_dims(image, axis=0))
        class_output = predictions[:, class_idx]

    grads = tape.gradient(class_output, conv_output)

    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

    conv_output = conv_output[0]
    heatmap = tf.reduce_sum(conv_output * pooled_grads, axis=-1)

    heatmap = tf.maximum(heatmap, 0) / tf.math.reduce_max(heatmap)

    return heatmap.numpy()


def plot_gradcam(
    image: np.ndarray,
    heatmap: np.ndarray,
    alpha: float = 0.4,
    save_path: Optional[str] = None
) -> plt.Figure:
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    axes[0].imshow(image)
    axes[0].set_title('Orijinal Görüntü')
    axes[0].axis('off')

    axes[1].imshow(heatmap, cmap='jet')
    axes[1].set_title('Grad-CAM Isı Haritası')
    axes[1].axis('off')

    heatmap_resized = tf.image.resize(
        heatmap[..., np.newaxis],
        (image.shape[0], image.shape[1])
    ).numpy()[:, :, 0]

    axes[2].imshow(image)
    axes[2].imshow(heatmap_resized, cmap='jet', alpha=alpha)
    axes[2].set_title('Grad-CAM Overlay')
    axes[2].axis('off')

    plt.tight_layout()

    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Grad-CAM kaydedildi: {save_path}")

    return fig
