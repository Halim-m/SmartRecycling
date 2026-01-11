#!/usr/bin/env python3
import sys
import argparse
import zipfile
import shutil
from pathlib import Path
import urllib.request
from tqdm import tqdm
import random



DATASETS = {
    'realwaste': {
        'url': 'https://archive.org/download/realwaste-image-classification/RealWaste.zip',
        'description': 'RealWaste - 4,752 images, 9 classes',
        'kaggle': 'realwaste-image-classification'
    },
    'trashnet': {
        'url': 'https://github.com/garythung/trashnet/raw/master/data/dataset-resized.zip',
        'description': 'TrashNet - 2,527 images, 6 classes',
    }
}


class DownloadProgressBar(tqdm):
    """İndirme ilerleme çubuğu."""

    def update_to(self, b=1, bsize=1, tsize=None):
        if tsize is not None:
            self.total = tsize
        self.update(b * bsize - self.n)


def download_file(url: str, output_path: str) -> None:
    """
    URL'den dosya indir.

    Args:
        url: İndirilecek URL
        output_path: Kayıt yolu
    """
    print(f"İndiriliyor: {url}")

    with DownloadProgressBar(unit='B', unit_scale=True, miniters=1, desc=output_path) as t:
        urllib.request.urlretrieve(url, filename=output_path, reporthook=t.update_to)


def extract_zip(zip_path: str, extract_to: str) -> None:
    """
    ZIP dosyasını çıkart.

    Args:
        zip_path: ZIP dosya yolu
        extract_to: Çıkartılacak dizin
    """
    print(f"Çıkartılıyor: {zip_path}")

    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(extract_to)


def split_dataset(
    source_dir: str,
    dest_dir: str,
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    seed: int = 42
) -> dict:
    """
    Veri setini train/val/test olarak böl.

    Args:
        source_dir: Kaynak dizin
        dest_dir: Hedef dizin
        train_ratio: Eğitim oranı
        val_ratio: Doğrulama oranı
        test_ratio: Test oranı
        seed: Random seed

    Returns:
        Bölme istatistikleri
    """
    random.seed(seed)

    source_path = Path(source_dir)
    dest_path = Path(dest_dir)

    stats = {'train': {}, 'val': {}, 'test': {}}

    # Her sınıf için
    for class_dir in source_path.iterdir():
        if not class_dir.is_dir():
            continue

        class_name = class_dir.name.lower()

        # Görüntüleri listele
        images = list(class_dir.glob('*'))
        images = [img for img in images if img.suffix.lower() in ['.jpg', '.jpeg', '.png', '.bmp']]

        if not images:
            continue

        random.shuffle(images)

        # Bölme indeksleri
        n = len(images)
        train_end = int(n * train_ratio)
        val_end = train_end + int(n * val_ratio)

        splits = {
            'train': images[:train_end],
            'val': images[train_end:val_end],
            'test': images[val_end:]
        }

        # Dosyaları kopyala
        for split_name, split_images in splits.items():
            split_class_dir = dest_path / split_name / class_name
            split_class_dir.mkdir(parents=True, exist_ok=True)

            for img in split_images:
                shutil.copy2(img, split_class_dir / img.name)

            stats[split_name][class_name] = len(split_images)

    return stats


def download_realwaste(data_dir: str) -> None:
    """
    RealWaste veri setini indir ve hazırla.

    Args:
        data_dir: Veri dizini
    """
    raw_dir = Path(data_dir) / 'raw'
    processed_dir = Path(data_dir) / 'processed'

    raw_dir.mkdir(parents=True, exist_ok=True)
    processed_dir.mkdir(parents=True, exist_ok=True)

    zip_path = raw_dir / 'realwaste.zip'

    # İndir
    if not zip_path.exists():
        print("\n=== RealWaste veri seti indiriliyor ===")
        print("Not: Bu veri seti Kaggle'dan indirilebilir:")
        print("kaggle datasets download -d joebeachcapital/realwaste")
        print("\nAlternatif olarak manuel indirme yapabilirsiniz:")
        print("1. https://www.kaggle.com/datasets/joebeachcapital/realwaste adresine gidin")
        print("2. 'Download' butonuna tıklayın")
        print(f"3. ZIP dosyasını {zip_path} konumuna kaydedin")
        print("\nVeri seti hazır olduğunda scripti tekrar çalıştırın.")
        return

    # Çıkart
    extract_dir = raw_dir / 'realwaste'
    if not extract_dir.exists():
        extract_zip(str(zip_path), str(raw_dir))

    # Bölme
    print("\n=== Veri seti bölünüyor ===")
    source = raw_dir / 'RealWaste'

    if not source.exists():
        # Farklı dizin yapısı olabilir
        possible_sources = list(raw_dir.glob('**/RealWaste'))
        if possible_sources:
            source = possible_sources[0]
        else:
            print(f"Hata: Veri seti dizini bulunamadı: {source}")
            return

    stats = split_dataset(str(source), str(processed_dir))

    # İstatistikleri yazdır
    print("\n=== Veri Seti İstatistikleri ===")
    for split_name, classes in stats.items():
        total = sum(classes.values())
        print(f"\n{split_name.upper()}: {total} görüntü")
        for class_name, count in sorted(classes.items()):
            print(f"  {class_name}: {count}")


def download_trashnet(data_dir: str) -> None:
    """
    TrashNet veri setini indir ve hazırla.

    Args:
        data_dir: Veri dizini
    """
    raw_dir = Path(data_dir) / 'raw'
    processed_dir = Path(data_dir) / 'processed'

    raw_dir.mkdir(parents=True, exist_ok=True)
    processed_dir.mkdir(parents=True, exist_ok=True)

    zip_path = raw_dir / 'trashnet.zip'

    # İndir
    if not zip_path.exists():
        try:
            download_file(DATASETS['trashnet']['url'], str(zip_path))
        except Exception as e:
            print(f"İndirme hatası: {e}")
            print("\nManuel indirme:")
            print("1. https://github.com/garythung/trashnet adresine gidin")
            print("2. data/dataset-resized.zip dosyasını indirin")
            print(f"3. {zip_path} konumuna kaydedin")
            return

    # Çıkart
    extract_dir = raw_dir / 'trashnet'
    if not extract_dir.exists():
        extract_zip(str(zip_path), str(raw_dir))

    # Bölme
    print("\n=== Veri seti bölünüyor ===")
    source = raw_dir / 'dataset-resized'

    if not source.exists():
        possible_sources = list(raw_dir.glob('**/dataset-resized'))
        if possible_sources:
            source = possible_sources[0]
        else:
            print(f"Hata: Veri seti dizini bulunamadı: {source}")
            return

    stats = split_dataset(str(source), str(processed_dir))

    # İstatistikleri yazdır
    print("\n=== Veri Seti İstatistikleri ===")
    for split_name, classes in stats.items():
        total = sum(classes.values())
        print(f"\n{split_name.upper()}: {total} görüntü")
        for class_name, count in sorted(classes.items()):
            print(f"  {class_name}: {count}")


def main():
    parser = argparse.ArgumentParser(description='Atık sınıflandırma veri seti indir')
    parser.add_argument(
        '--dataset',
        type=str,
        choices=['realwaste', 'trashnet'],
        default='realwaste',
        help='İndirilecek veri seti'
    )
    parser.add_argument(
        '--data-dir',
        type=str,
        default='data',
        help='Veri dizini'
    )

    args = parser.parse_args()

    print(f"\n{'='*50}")
    print(f"Smart Recycling - Veri Seti Hazırlama")
    print(f"{'='*50}")
    print(f"\nSeçilen veri seti: {args.dataset}")
    print(f"Veri dizini: {args.data_dir}")

    if args.dataset == 'realwaste':
        download_realwaste(args.data_dir)
    elif args.dataset == 'trashnet':
        download_trashnet(args.data_dir)

    print("\n=== Tamamlandı ===")


if __name__ == '__main__':
    main()
