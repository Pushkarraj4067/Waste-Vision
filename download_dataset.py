
from pathlib import Path
from urllib.request import urlopen, Request
from zipfile import ZipFile
import shutil
import sys

ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"
DATASET_DIR = DATA_DIR / "dataset-resized"
ZIP_PATH = DATA_DIR / "dataset-resized.zip"

DATASET_URL = (
    "https://huggingface.co/datasets/garythung/"
    "trashnet/resolve/main/dataset-resized.zip"
)

CLASSES = ["cardboard", "glass", "metal", "paper", "plastic", "trash"]


def dataset_is_ready():
    return all(
        (DATASET_DIR / category).is_dir()
        and any((DATASET_DIR / category).iterdir())
        for category in CLASSES
    )


def download_dataset():
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    if dataset_is_ready():
        print("Dataset already exists. Skipping download.")
        return

    print("Downloading TrashNet dataset...")

    request = Request(
        DATASET_URL,
        headers={"User-Agent": "WasteVision-Dataset-Downloader"}
    )

    try:
        with urlopen(request, timeout=60) as response:
            with open(ZIP_PATH, "wb") as file:
                shutil.copyfileobj(response, file)

        print("Download complete. Extracting dataset...")

        with ZipFile(ZIP_PATH, "r") as archive:
            for member in archive.infolist():
                destination = (DATA_DIR / member.filename).resolve()
                if not destination.is_relative_to(DATA_DIR.resolve()):
                    raise ValueError("Unsafe file path in ZIP archive.")

            archive.extractall(DATA_DIR)

        if not dataset_is_ready():
            raise RuntimeError("Dataset extraction failed.")

        print("Dataset downloaded and extracted successfully!")

    except Exception as error:
        print(f"Dataset download failed: {error}")
        sys.exit(1)

    finally:
        if ZIP_PATH.exists():
            ZIP_PATH.unlink()


if __name__ == "__main__":
    download_dataset()
