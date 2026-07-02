import zipfile
import urllib.request
from pathlib import Path


def download_movielens_100k(raw_dir: str = "data/raw") -> Path:
    url = "https://files.grouplens.org/datasets/movielens/ml-100k.zip"
    raw_path = Path(raw_dir)
    raw_path.mkdir(parents=True, exist_ok=True)

    extracted = raw_path / "ml-100k"
    if extracted.exists():
        return extracted

    zip_path = raw_path / "ml-100k.zip"
    print(f"Downloading MovieLens 100K to {zip_path} ...")
    urllib.request.urlretrieve(url, zip_path)

    print("Extracting ...")
    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(raw_path)

    zip_path.unlink()
    return extracted
