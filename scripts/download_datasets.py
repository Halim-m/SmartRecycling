#!/usr/bin/env python3
"""
Çoklu atık veri seti indirme ve birleştirme scripti.

Bu script Kaggle'dan birden fazla veri setini indirir ve
bizim 9 sınıflı formatımıza dönüştürür.

Kullanım (Colab'da):
    1. Kaggle API key'inizi yükleyin
    2. Bu scripti çalıştırın

Veri Setleri:
    1. RealWaste (~4,752 görüntü) - Zaten var
    2. Garbage Classification (~15,150 görüntü)
    3. TrashNet (~2,527 görüntü)

Toplam: ~22,000+ görüntü
"""

import os
import shutil
from pathlib import Path
import random

# Hedef sınıflar (bizim 9 sınıfımız)
TARGET_CLASSES = [
    'cardboard',
    'food_organics',
    'glass',
    'metal',
    'miscellaneous_trash',
    'paper',
    'plastic',
    'textile_trash',
    'vegetation'
]

# Veri seti sınıf eşleştirmeleri
# Her veri setindeki sınıfların bizim sınıflarımıza nasıl eşleneceği

GARBAGE_CLASSIFICATION_MAPPING = {
    # Garbage Classification (12 sınıf) -> Bizim 9 sınıf
    'cardboard': 'cardboard',
    'glass': 'glass',
    'metal': 'metal',
    'paper': 'paper',
    'plastic': 'plastic',
    'trash': 'miscellaneous_trash',
    'battery': 'miscellaneous_trash',  # e-atık -> misc
    'biological': 'food_organics',
    'brown-glass': 'glass',
    'green-glass': 'glass',
    'white-glass': 'glass',
    'clothes': 'textile_trash',
    'shoes': 'textile_trash',
}

TRASHNET_MAPPING = {
    # TrashNet (6 sınıf) -> Bizim 9 sınıf
    'cardboard': 'cardboard',
    'glass': 'glass',
    'metal': 'metal',
    'paper': 'paper',
    'plastic': 'plastic',
    'trash': 'miscellaneous_trash',
}

REALWASTE_MAPPING = {
    # RealWaste zaten bizim formatımızda
    'cardboard': 'cardboard',
    'food_organics': 'food_organics',
    'glass': 'glass',
    'metal': 'metal',
    'miscellaneous_trash': 'miscellaneous_trash',
    'paper': 'paper',
    'plastic': 'plastic',
    'textile_trash': 'textile_trash',
    'vegetation': 'vegetation',
}


def setup_kaggle():
    """Kaggle API'yi ayarla."""
    print("Kaggle API kontrol ediliyor...")

    # Colab'da kaggle.json yüklenmişse
    if os.path.exists('/content/kaggle.json'):
        os.makedirs('/root/.kaggle', exist_ok=True)
        shutil.copy('/content/kaggle.json', '/root/.kaggle/kaggle.json')
        os.chmod('/root/.kaggle/kaggle.json', 0o600)
        print("Kaggle API key ayarlandı.")
    elif os.path.exists(os.path.expanduser('~/.kaggle/kaggle.json')):
        print("Kaggle API key mevcut.")
    else:
        print("UYARI: Kaggle API key bulunamadı!")
        print("Lütfen kaggle.json dosyasını yükleyin.")
        return False

    return True


def download_datasets(base_dir):
    """Kaggle veri setlerini indir."""
    import subprocess

    datasets_dir = Path(base_dir) / 'datasets_raw'
    datasets_dir.mkdir(parents=True, exist_ok=True)

    datasets = [
        # (kaggle_dataset_id, local_folder_name)
        ('techsash/waste-classification-data', 'garbage_classification'),
        ('mostafaabla/garbage-classification', 'trashnet'),
    ]

    for dataset_id, folder_name in datasets:
        output_dir = datasets_dir / folder_name
        if output_dir.exists():
            print(f"{folder_name} zaten indirilmiş, atlanıyor...")
            continue

        print(f"\n{dataset_id} indiriliyor...")
        try:
            subprocess.run([
                'kaggle', 'datasets', 'download', '-d', dataset_id,
                '-p', str(datasets_dir), '--unzip'
            ], check=True)
            print(f"{folder_name} indirildi.")
        except Exception as e:
            print(f"HATA: {dataset_id} indirilemedi: {e}")

    return datasets_dir


def merge_datasets(datasets_dir, output_dir, realwaste_dir=None):
    """
    Tüm veri setlerini birleştir.

    Args:
        datasets_dir: İndirilen veri setlerinin dizini
        output_dir: Birleştirilmiş veri setinin kaydedileceği dizin
        realwaste_dir: RealWaste veri setinin dizini (opsiyonel)
    """
    output_path = Path(output_dir)
    datasets_path = Path(datasets_dir)

    # Hedef sınıf dizinlerini oluştur
    for cls in TARGET_CLASSES:
        (output_path / cls).mkdir(parents=True, exist_ok=True)

    stats = {cls: 0 for cls in TARGET_CLASSES}

    # 1. RealWaste (eğer varsa)
    if realwaste_dir and Path(realwaste_dir).exists():
        print("\n1. RealWaste işleniyor...")
        realwaste_path = Path(realwaste_dir)

        for src_class, dst_class in REALWASTE_MAPPING.items():
            src_dir = realwaste_path / src_class
            if src_dir.exists():
                for img in src_dir.glob('*'):
                    if img.suffix.lower() in ['.jpg', '.jpeg', '.png']:
                        dst_name = f"realwaste_{src_class}_{img.name}"
                        shutil.copy(img, output_path / dst_class / dst_name)
                        stats[dst_class] += 1

        print(f"RealWaste: {sum(stats.values())} görüntü eklendi")

    # 2. Garbage Classification
    print("\n2. Garbage Classification işleniyor...")
    gc_paths = [
        datasets_path / 'DATASET',
        datasets_path / 'garbage_classification' / 'DATASET',
        datasets_path / 'Garbage classification' / 'Garbage classification',
    ]

    gc_path = None
    for p in gc_paths:
        if p.exists():
            gc_path = p
            break

    if gc_path:
        prev_count = sum(stats.values())
        for src_class, dst_class in GARBAGE_CLASSIFICATION_MAPPING.items():
            # Train ve test klasörlerini kontrol et
            for split in ['train', 'test', '']:
                src_dir = gc_path / split / src_class if split else gc_path / src_class
                if src_dir.exists():
                    for img in src_dir.glob('*'):
                        if img.suffix.lower() in ['.jpg', '.jpeg', '.png']:
                            dst_name = f"gc_{src_class}_{img.name}"
                            shutil.copy(img, output_path / dst_class / dst_name)
                            stats[dst_class] += 1

        print(f"Garbage Classification: {sum(stats.values()) - prev_count} görüntü eklendi")
    else:
        print("Garbage Classification bulunamadı")

    # 3. TrashNet
    print("\n3. TrashNet işleniyor...")
    trashnet_paths = [
        datasets_path / 'trashnet' / 'Garbage classification' / 'Garbage classification',
        datasets_path / 'Garbage classification' / 'Garbage classification',
        datasets_path / 'trashnet',
    ]

    trashnet_path = None
    for p in trashnet_paths:
        if p.exists():
            trashnet_path = p
            break

    if trashnet_path:
        prev_count = sum(stats.values())
        for src_class, dst_class in TRASHNET_MAPPING.items():
            src_dir = trashnet_path / src_class
            if src_dir.exists():
                for img in src_dir.glob('*'):
                    if img.suffix.lower() in ['.jpg', '.jpeg', '.png']:
                        dst_name = f"trashnet_{src_class}_{img.name}"
                        # Duplikasyonu önle
                        dst_file = output_path / dst_class / dst_name
                        if not dst_file.exists():
                            shutil.copy(img, dst_file)
                            stats[dst_class] += 1

        print(f"TrashNet: {sum(stats.values()) - prev_count} görüntü eklendi")
    else:
        print("TrashNet bulunamadı")

    # İstatistikleri yazdır
    print("\n" + "="*50)
    print("BİRLEŞTİRİLMİŞ VERİ SETİ İSTATİSTİKLERİ")
    print("="*50)

    total = 0
    for cls in TARGET_CLASSES:
        count = stats[cls]
        total += count
        print(f"{cls:25} : {count:5} görüntü")

    print("-"*50)
    print(f"{'TOPLAM':25} : {total:5} görüntü")
    print("="*50)

    return stats


def create_splits(merged_dir, output_dir, train_ratio=0.7, val_ratio=0.15):
    """
    Birleştirilmiş veri setini train/val/test olarak böl.
    """
    merged_path = Path(merged_dir)
    output_path = Path(output_dir)

    # Çıktı dizinlerini oluştur
    for split in ['train', 'val', 'test']:
        for cls in TARGET_CLASSES:
            (output_path / split / cls).mkdir(parents=True, exist_ok=True)

    print("\nVeri seti bölünüyor...")

    for cls in TARGET_CLASSES:
        cls_dir = merged_path / cls
        if not cls_dir.exists():
            continue

        images = list(cls_dir.glob('*'))
        random.shuffle(images)

        n_total = len(images)
        n_train = int(n_total * train_ratio)
        n_val = int(n_total * val_ratio)

        train_images = images[:n_train]
        val_images = images[n_train:n_train + n_val]
        test_images = images[n_train + n_val:]

        # Kopyala
        for img in train_images:
            shutil.copy(img, output_path / 'train' / cls / img.name)
        for img in val_images:
            shutil.copy(img, output_path / 'val' / cls / img.name)
        for img in test_images:
            shutil.copy(img, output_path / 'test' / cls / img.name)

        print(f"{cls}: train={len(train_images)}, val={len(val_images)}, test={len(test_images)}")

    print("\nVeri seti hazır!")


def main():
    """Ana fonksiyon - Colab'da çalıştırılacak."""

    # Colab çalışma dizini
    BASE_DIR = '/content'
    DATASETS_RAW = f'{BASE_DIR}/datasets_raw'
    MERGED_DIR = f'{BASE_DIR}/merged_dataset'
    FINAL_DIR = f'{BASE_DIR}/waste_dataset'

    # RealWaste dizini (eğer önceden yüklenmişse)
    REALWASTE_DIR = f'{BASE_DIR}/RealWaste'  # veya None

    print("="*60)
    print("ÇOKLU VERİ SETİ İNDİRME VE BİRLEŞTİRME")
    print("="*60)

    # 1. Kaggle'ı ayarla
    if not setup_kaggle():
        print("\nKaggle API ayarlanamadı. Manuel indirme gerekebilir.")
        return

    # 2. Veri setlerini indir
    print("\n" + "-"*60)
    print("VERİ SETLERİ İNDİRİLİYOR...")
    print("-"*60)
    download_datasets(DATASETS_RAW)

    # 3. Veri setlerini birleştir
    print("\n" + "-"*60)
    print("VERİ SETLERİ BİRLEŞTİRİLİYOR...")
    print("-"*60)

    # RealWaste'i kontrol et
    realwaste = REALWASTE_DIR if os.path.exists(REALWASTE_DIR) else None
    merge_datasets(DATASETS_RAW, MERGED_DIR, realwaste)

    # 4. Train/Val/Test olarak böl
    print("\n" + "-"*60)
    print("VERİ SETİ BÖLÜNÜYOR...")
    print("-"*60)
    create_splits(MERGED_DIR, FINAL_DIR)

    print("\n" + "="*60)
    print("TAMAMLANDI!")
    print(f"Veri seti hazır: {FINAL_DIR}")
    print("="*60)


if __name__ == "__main__":
    main()
